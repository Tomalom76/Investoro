import os
import json
import time
import shutil
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pycaret.regression import load_model, predict_model

# --- Konfiguracja ---
INPUT_DIR = "CENY_IN"
OUTPUT_DIR = "CENY_OUT"
PROCESSED_DIR = "CENY_PROCESSED"
ERROR_DIR = os.path.join(PROCESSED_DIR, "errors")

MODEL_PATH = "final_price_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

CHECK_INTERVAL_SECONDS = 5
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories():
    logging.info("Sprawdzanie i tworzenie katalogów...")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ERROR_DIR, exist_ok=True)
    logging.info(f"Katalogi gotowe: '{INPUT_DIR}', '{OUTPUT_DIR}', '{PROCESSED_DIR}', '{ERROR_DIR}'")

# --- KOREKTA: Zaktualizowana funkcja prepare_data_for_prediction ---
def prepare_data_for_prediction(df_input, vectorizer):
    """
    Kompleksowo przygotowuje dane do predykcji zgodnie z POPRAWIONĄ logiką z notatnika.
    """
    df = df_input.copy()
    
    # 1. Inflacja (bez zmian)
    def adjust_price(row):
        price = pd.to_numeric(row.get('Price'), errors='coerce')
        if pd.isna(price): return np.nan
        date_str = row.get('NewestDate') if pd.notna(row.get('NewestDate')) else row.get('DateAdded')
        if date_str is None or date_str == 'NULL': return price
        try:
            offer_date = pd.to_datetime(date_str, errors='coerce')
            if pd.isna(offer_date): return price
            years_diff = (datetime.now() - offer_date).days / 365.25
            return round(price * (1.05**years_diff), 0) if years_diff > 0 else price
        except: return price
    adjusted_price_value = df.apply(adjust_price, axis=1).iloc[0]
    df['AdjustedPrice'] = adjusted_price_value

    # 2. Konwersja typów i wiek budynku (bez zmian)
    for col in ['Area', 'NumberOfRooms', 'Floor', 'Floors']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'BuiltYear' in df.columns:
        built_year_numeric = pd.to_numeric(df['BuiltYear'], errors='coerce').fillna(1980) 
        df['BuildingAge'] = datetime.now().year - built_year_numeric
        df['BuiltYear'] = pd.to_datetime(built_year_numeric, format='%Y', errors='coerce')

    # 3. KOREKTA: Poprawne przygotowanie cech kategorycznych
    # Usunięto błędne tworzenie 'Unified_Location'.
    if 'Predict_State' not in df.columns:
        df['Predict_State'] = 'Brak Danych'
    else:
        df['Predict_State'].fillna('Brak Danych', inplace=True) # Stan budynku
        
    if 'Predict_Loc' not in df.columns:
        df['Predict_Loc'] = 'Brak Danych'
    else:
        df['Predict_Loc'].fillna('Brak Danych', inplace=True) # Lokalizacja

    # 4. TF-IDF (bez zmian)
    description_clean = df['Description'].fillna('').str.lower()
    tfidf_features = vectorizer.transform(description_clean)
    df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=[f"tfidf_{name}" for name in vectorizer.get_feature_names_out()])
    df_tfidf.index = df.index
    df = pd.concat([df, df_tfidf], axis=1)

    return df, adjusted_price_value

def process_json_file(filepath, filename, model, vectorizer):
    logging.info(f"Przetwarzam plik: {filename}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data_json = json.load(f)

        # KOREKTA: Zaktualizowane kluczowe pola
        required_keys = ['Price', 'Description', 'Area', 'BuiltYear', 'Predict_State', 'Predict_Loc']
        missing_keys = [key for key in required_keys if key not in data_json or data_json[key] is None]

        if missing_keys:
            error_message = f"Plik {filename} jest niekompletny. Brakuje kluczowych pól: {', '.join(missing_keys)}"
            logging.error(error_message)
            error_path = os.path.join(ERROR_DIR, filename)
            shutil.move(filepath, error_path)
            logging.warning(f"Plik powodujący błąd został przeniesiony do: {error_path}")
            return

        df_to_predict = pd.DataFrame([data_json])
        prepared_df, adjusted_price = prepare_data_for_prediction(df_to_predict, vectorizer)
        predictions = predict_model(model, data=prepared_df)
        
        original_price = data_json.get('Price', 0)
        predicted_price = round(predictions['prediction_label'].iloc[0], 2)
        predicted_condition = prepared_df['Predict_State'].iloc[0] # Poprawna nazwa
        predicted_location = prepared_df['Predict_Loc'].iloc[0]   # Poprawna nazwa

        logging.info(f"--- WYNIKI DLA PLIKU: {filename} ---")
        logging.info(f"  > Oryginalna cena ofertowa (Price):        {original_price:,.2f} PLN")
        logging.info(f"  > Cena po waloryzacji (AdjustedPrice):     {adjusted_price:,.2f} PLN")
        logging.info(f"  > Przewidziana cena (PredictedPrice):      {predicted_price:,.2f} PLN")
        logging.info(f"  > Przewidziany STAN BUDYNKU (Predict_State): {predicted_condition}")
        logging.info(f"  > Przewidziana LOKALIZACJA (Predict_Loc):    {predicted_location}")
        logging.info("--------------------------------------------------")
        
        data_json['PredictedPrice'] = predicted_price
        data_json['AdjustedPrice'] = adjusted_price
        data_json['PredictedCondition'] = predicted_condition # Używam bardziej opisowej nazwy w wyjściu
        data_json['PredictedLocation'] = predicted_location

        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_json, f, ensure_ascii=False, indent=4)
        logging.info(f"Wynik zapisano w: {output_path}. Przetwarzanie zakończone pomyślnie.")

    except Exception as e:
        logging.error(f"KRYTYCZNY BŁĄD podczas przetwarzania pliku {filename}: {e}", exc_info=True)
        error_path = os.path.join(ERROR_DIR, filename)
        try:
            shutil.move(filepath, error_path)
            logging.warning(f"Plik powodujący błąd został przeniesiony do: {error_path}")
        except Exception as move_error:
            logging.error(f"Nie udało się przenieść błędnego pliku {filepath} do katalogu error: {move_error}")

def main():
    setup_directories()
    try:
        logging.info("Wczytywanie wytrenowanego modelu PyCaret...")
        model = load_model(MODEL_PATH)
        logging.info("Model wczytany pomyślnie.")
        logging.info("Wczytywanie wektoryzatora TF-IDF...")
        vectorizer = joblib.load(VECTORIZER_PATH)
        logging.info("Wektoryzator wczytany pomyślnie.")
    except FileNotFoundError as e:
        logging.error(f"BŁĄD KRYTYCZNY: Nie znaleziono wymaganego pliku: {e.filename}. Czy na pewno ponownie wytrenowałeś model po poprawkach?")
        return
    except Exception as e:
        logging.error(f"Wystąpił błąd podczas wczytywania plików startowych: {e}")
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
                    logging.info(f"Znaleziono i przeniesiono do przetwarzania nowy plik: {filename}")
                    process_json_file(processed_path, filename, model, vectorizer)
                except FileNotFoundError:
                    logging.warning(f"Plik {filename} zniknął przed przetworzeniem. Ignoruję.")
                    continue
                except Exception as e:
                    logging.error(f"Nie udało się przenieść lub rozpocząć przetwarzania pliku {filename}: {e}")
        
        except KeyboardInterrupt:
            logging.info("Otrzymano sygnał zatrzymania. Kończę pracę.")
            break
        except Exception as e:
            logging.error(f"Wystąpił nieoczekiwany błąd w głównej pętli: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL_SECONDS * 2)

if __name__ == "__main__":
    main()