import os
import json
import time
import shutil
import logging
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import lightgbm as lgb

# --- Konfiguracja ---
INPUT_DIR = "CENY_IN"
OUTPUT_DIR = "CENY_OUT"
PROCESSED_DIR = "CENY_PROCESSED"
ERROR_DIR = os.path.join(PROCESSED_DIR, "errors")
ARTIFACTS_DIR = "artifacts_ppm_routing"
CHECK_INTERVAL_SECONDS = 5

# Progi pewności dla reguł dynamicznych
MEGA_T, PEWNY_T, CHYBA_T = 0.85, 0.70, 0.55
ALPHA_MEGA, ALPHA_PEWNY, ALPHA_CHYBA = 0.15, 0.10, 0.05

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories():
    logging.info("Sprawdzanie i tworzenie katalogów...")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ERROR_DIR, exist_ok=True)
    
    # Sprawdź artefakty modelu
    required_artifacts = [
        'lgb_q10.txt', 'lgb_q50.txt', 'lgb_q90.txt',
        'isotonic_calibrator_ratio_fixed_v4.pkl',
        'logreg_calibrator_ratio_fixed_v4.pkl'
    ]
    missing = [f for f in required_artifacts if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))]
    if missing:
        raise RuntimeError(f"Brakujące artefakty modelu v11 w '{ARTIFACTS_DIR}': {missing}")
    
    logging.info(f"Katalogi i artefakty gotowe.")

def prepare_data_for_prediction(df_input):
    """Przygotowuje dane do predykcji modelem v11"""
    df = df_input.copy()
    
    # 1. Konwersja numerycznych
    numeric_cols = ['Area', 'NumberOfRooms', 'Floor', 'Floors', 'BuiltYear']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. Wiek budynku
    if 'BuiltYear' in df.columns:
        built_year = pd.to_numeric(df['BuiltYear'], errors='coerce').fillna(2000)
        built_year = built_year.clip(1800, datetime.now().year + 1)
        df['BuildingAge'] = (datetime.now().year - built_year).astype(int)
    else:
        df['BuildingAge'] = 60
    
    # 3. Wypełnienie brakujących numerycznych medianą (uproszczone)
    num_features = ['Area', 'NumberOfRooms', 'Floor', 'Floors', 'BuildingAge']
    for col in num_features:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 50)
    
    # 4. Kategoryczne - mapowanie na oczekiwane nazwy v11
    cat_mapping = {
        'Predict_State': 'Predict_State',
        'Predicted_Loc': 'Predicted_Loc',
        'BuildingType': 'BuildingType',
        'TypeOfMarket': 'TypeOfMarket', 
        'Type': 'Type',
        'OfferFrom': 'OfferFrom',
        'city_slug': 'city_slug'
    }
    
    for expected_col, source_col in cat_mapping.items():
        if source_col in df.columns:
            df[expected_col] = df[source_col].astype(str).fillna('unknown')
        elif expected_col == 'Predicted_Loc' and 'Predict_Loc' in df.columns:
            df[expected_col] = df['Predict_Loc'].astype(str).fillna('unknown')
        else:
            df[expected_col] = 'unknown'
    
    # 5. Konwersja kategorycznych na category (wymagane przez LightGBM)
    cat_features = ['Predict_State', 'Predicted_Loc', 'BuildingType', 'TypeOfMarket', 'Type', 'OfferFrom', 'city_slug']
    for col in cat_features:
        df[col] = df[col].astype('category')
    
    return df

def predict_with_v11_model(df, models, calibrators):
    """Wykonuje predykcję z regułami dynamicznymi"""
    b10, b50, b90 = models
    iso_cal, log_cal = calibrators
    scaler = log_cal['scaler']
    logreg = log_cal['logreg']
    
    # Cechy dla modelu
    num_cols = ['Area', 'NumberOfRooms', 'Floor', 'Floors', 'BuildingAge']
    cat_cols = ['Predict_State', 'Predicted_Loc', 'BuildingType', 'TypeOfMarket', 'Type', 'OfferFrom', 'city_slug']
    X = df[num_cols + cat_cols]
    
    # Predykcje kwantylowe
    p10 = np.asarray(b10.predict(X)).reshape(-1)
    p50 = np.asarray(b50.predict(X)).reshape(-1) 
    p90 = np.asarray(b90.predict(X)).reshape(-1)
    
    # Korekta monotonii
    p10_fix = np.minimum.reduce([p10, p50, p90])
    p90_fix = np.maximum.reduce([p10, p50, p90])
    p50_fix = np.clip(p50, p10_fix, p90_fix)
    
    # Wskaźniki szerokości
    width_abs = np.maximum(0.0, p90_fix - p10_fix)
    width_ratio = width_abs / np.maximum(1.0, p50_fix)
    
    # Probability z kalibracji
    prob_iso = iso_cal.predict(width_ratio)
    X_feat = np.c_[width_ratio, width_abs, p50_fix, np.log1p(p50_fix)]
    Xs = scaler.transform(X_feat)
    prob_log = logreg.predict_proba(Xs)[:, 1]
    probability = (0.5 * prob_iso + 0.5 * prob_log).clip(0, 1)
    
    # Cena bazowa (z oferty) i predykcja surowa
    base_price = df['Price'].iloc[0]
    predicted_ppm = p50_fix[0]
    area = df['Area'].iloc[0]
    raw_predicted_price = predicted_ppm * area
    
    # Reguły dynamiczne
    prob = probability[0]
    if prob >= MEGA_T:
        alpha_used = ALPHA_MEGA
        conf_bucket = 'mega'
    elif prob >= PEWNY_T:
        alpha_used = ALPHA_PEWNY
        conf_bucket = 'pewny'
    elif prob >= CHYBA_T:
        alpha_used = ALPHA_CHYBA
        conf_bucket = 'chyba'
    else:
        alpha_used = 0.0
        conf_bucket = 'brak'
    
    # Cena finalna z ograniczeniem ±alpha
    if alpha_used > 0:
        min_allowed = base_price * (1 - alpha_used)
        max_allowed = base_price * (1 + alpha_used)
        final_price = np.clip(raw_predicted_price, min_allowed, max_allowed)
    else:
        final_price = base_price
    
    return {
        'raw_predicted_price': float(raw_predicted_price),
        'final_price': float(final_price),
        'probability': float(prob),
        'confidence_bucket': conf_bucket,
        'alpha_used': float(alpha_used),
        'predicted_ppm': float(predicted_ppm)
    }

def process_json_file(filepath, filename, models, calibrators):
    logging.info(f"Przetwarzam plik: {filename}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        
        # Sprawdź wymagane pola
        required_keys = ['Price', 'Area']
        missing_keys = [key for key in required_keys if key not in data_json or data_json[key] is None]
        
        if missing_keys:
            error_message = f"Plik {filename} jest niekompletny. Brakuje kluczowych pól: {', '.join(missing_keys)}"
            logging.error(error_message)
            error_path = os.path.join(ERROR_DIR, filename)
            shutil.move(filepath, error_path)
            return
        
        # Przygotuj dane
        df_to_predict = pd.DataFrame([data_json])
        prepared_df = prepare_data_for_prediction(df_to_predict)
        
        # Predykcja
        results = predict_with_v11_model(prepared_df, models, calibrators)
        
        # Wyniki
        original_price = data_json.get('Price', 0)
        
        logging.info(f"--- WYNIKI DLA PLIKU: {filename} ---")
        logging.info(f" > Cena ofertowa: {original_price:,.2f} PLN")
        logging.info(f" > Surowa predykcja: {results['raw_predicted_price']:,.2f} PLN")
        logging.info(f" > Cena finalna (z regułami): {results['final_price']:,.2f} PLN")
        logging.info(f" > Pewność modelu: {results['probability']:.3f}")
        logging.info(f" > Koszyk pewności: {results['confidence_bucket']} (±{results['alpha_used']*100:.0f}%)")
        logging.info(f" > Przewidziane ppm: {results['predicted_ppm']:,.2f} PLN/m²")
        logging.info("--------------------------------------------------")
        
        # Dodaj wyniki do JSON-a
        data_json.update({
            'RawPredictedPrice': results['raw_predicted_price'],
            'FinalPrice': results['final_price'],
            'Probability': results['probability'],
            'ConfidenceBucket': results['confidence_bucket'],
            'AlphaUsed': results['alpha_used'],
            'PredictedPPM': results['predicted_ppm']
        })
        
        # Zapisz wynik
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_json, f, ensure_ascii=False, indent=4)
        
        logging.info(f"Wynik zapisano w: {output_path}")
        
    except Exception as e:
        logging.error(f"BŁĄD podczas przetwarzania pliku {filename}: {e}", exc_info=True)
        error_path = os.path.join(ERROR_DIR, filename)
        try:
            shutil.move(filepath, error_path)
        except Exception as move_error:
            logging.error(f"Nie udało się przenieść błędnego pliku: {move_error}")

def main():
    setup_directories()
    
    try:
        # Wczytaj modele LightGBM
        logging.info("Wczytywanie modeli LightGBM...")
        b10 = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, 'lgb_q10.txt'))
        b50 = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, 'lgb_q50.txt'))
        b90 = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, 'lgb_q90.txt'))
        models = (b10, b50, b90)
        
        # Wczytaj kalibratory
        logging.info("Wczytywanie kalibratorów...")
        with open(os.path.join(ARTIFACTS_DIR, 'isotonic_calibrator_ratio_fixed_v4.pkl'), 'rb') as f:
            iso_cal = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'logreg_calibrator_ratio_fixed_v4.pkl'), 'rb') as f:
            log_cal = pickle.load(f)
        calibrators = (iso_cal, log_cal)
        
        logging.info("Modele wczytane pomyślnie.")
        
    except Exception as e:
        logging.error(f"BŁĄD podczas wczytywania modeli: {e}")
        return
    
    logging.info(f"Rozpoczynam nasłuchiwanie na nowe pliki w katalogu '{INPUT_DIR}'...")
    
    while True:
        try:
            files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
            
            if not files_to_process:
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            
            for filename in files_to_process:
                source_path = os.path.join(INPUT_DIR, filename)
                processed_path = os.path.join(PROCESSED_DIR, filename)
                
                try:
                    shutil.move(source_path, processed_path)
                    logging.info(f"Znaleziono nowy plik: {filename}")
                    process_json_file(processed_path, filename, models, calibrators)
                except FileNotFoundError:
                    logging.warning(f"Plik {filename} zniknął przed przetworzeniem.")
                    continue
                except Exception as e:
                    logging.error(f"Błąd przetwarzania pliku {filename}: {e}")
                    
        except KeyboardInterrupt:
            logging.info("Otrzymano sygnał zatrzymania. Kończę pracę.")
            break
        except Exception as e:
            logging.error(f"Nieoczekiwany błąd w głównej pętli: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL_SECONDS * 2)

if __name__ == "__main__":
    main()
