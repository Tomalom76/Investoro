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

# --- Ścieżki do plików niezbędnych do predykcji ---
MODEL_PATH = "final_price_model.pkl"
SLOWNIK_PATH = "slownik_finalny_z_hierarchia.csv"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

CHECK_INTERVAL_SECONDS = 5

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories():
    """Tworzy wymagane katalogi, jeśli nie istnieją."""
    logging.info("Sprawdzanie i tworzenie katalogów...")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ERROR_DIR, exist_ok=True)
    logging.info(f"Katalogi gotowe: '{INPUT_DIR}', '{OUTPUT_DIR}', '{PROCESSED_DIR}', '{ERROR_DIR}'")

def prepare_data_for_prediction(df_input, slownik_df, vectorizer):
    """
    Kompleksowo przygotowuje pojedynczy wiersz danych (w DataFrame) do predykcji.
    """
    df = df_input.copy()
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
    for col in ['Area', 'NumberOfRooms', 'Floor', 'Floors', 'StreetNumber']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'BuiltYear' in df.columns:
        built_year_numeric = pd.to_numeric(df['BuiltYear'], errors='coerce').fillna(1970)
        df['BuildingAge'] = datetime.now().year - built_year_numeric
        df['BuiltYear'] = pd.to_datetime(df['BuiltYear'], format='%Y', errors='coerce')
    if 'StreetNumber' in df.columns and 'UlicaID' in slownik_df.columns:
        df.rename(columns={'StreetNumber': 'UlicaID'}, inplace=True)
        df = pd.merge(df, slownik_df, on='UlicaID', how='left')
        for col in ['Dzielnica_Name', 'Ulica_Name']:
             if col in df.columns:
                df[col].fillna('Brak Danych', inplace=True)
    if 'Predict_State' in df.columns:
        df['Predict_State'].fillna('Brak Danych', inplace=True)
    if 'Predict_Loc' not in df.columns:
        df['Predict_Loc'] = np.nan
    dzielnica = df['Dzielnica_Name'].iloc[0] if 'Dzielnica_Name' in df.columns else 'Brak Danych'
    ulica = df['Ulica_Name'].iloc[0] if 'Ulica_Name' in df.columns else 'Brak Danych'
    location_str_fallback = f"{dzielnica} > {ulica}"
    df['Unified_Location'] = np.where(df['Predict_Loc'].notna(), df['Predict_Loc'], location_str_fallback)
    description_clean = df['Description'].fillna('').str.lower()
    tfidf_features = vectorizer.transform(description_clean)
    df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=[f"tfidf_{name}" for name in vectorizer.get_feature_names_out()])
    df_tfidf.index = df.index
    df = pd.concat([df, df_tfidf], axis=1)
    return df, adjusted_price_value

def process_json_file(filepath, filename, model, slownik_df, vectorizer):
    """
    Wersja z walidacją: Sprawdza, czy kluczowe pola istnieją, zanim rozpocznie przetwarzanie.
    """
    logging.info(f"Przetwarzam plik: {filename}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data_json = json.load(f)

        # --- BLOK WALIDACJI DANYCH WEJŚCIOWYCH ---
        required_keys = ['StreetNumber', 'Price', 'Description', 'Area', 'BuiltYear']
        missing_keys = [key for key in required_keys if key not in data_json]

        if missing_keys:
            # Jeśli brakuje jakiegokolwiek kluczowego pola, odrzuć plik
            error_message = f"Plik {filename} jest niekompletny. Brakuje kluczowych pól: {', '.join(missing_keys)}"
            logging.error(error_message)
            error_path = os.path.join(ERROR_DIR, filename)
            shutil.move(filepath, error_path)
            logging.warning(f"Plik powodujący błąd został przeniesiony do: {error_path}")
            return # Zakończ dalsze przetwarzanie tego pliku
        # --- KONIEC BLOKU WALIDACJI ---

        df_to_predict = pd.DataFrame([data_json])
        prepared_df, adjusted_price = prepare_data_for_prediction(df_to_predict, slownik_df, vectorizer)
        predictions = predict_model(model, data=prepared_df)
        
        original_price = data_json.get('Price', 0)
        predicted_price = round(predictions['prediction_label'].iloc[0], 2)
        predict_state = prepared_df['Predict_State'].iloc[0]
        predict_loc = prepared_df['Predict_Loc'].iloc[0] if pd.notna(prepared_df['Predict_Loc'].iloc[0]) else "Brak"

        logging.info(f"--- WYNIKI DLA PLIKU: {filename} ---")
        logging.info(f"  > Oryginalna cena ofertowa (Price):      {original_price:,.2f} PLN")
        logging.info(f"  > Cena po waloryzacji (Adjusted_Price):  {adjusted_price:,.2f} PLN")
        logging.info(f"  > Przewidziana cena (Predict_Price):     {predicted_price:,.2f} PLN")
        logging.info(f"  > Przewidziany stan (Predict_State):     {predict_state}")
        logging.info(f"  > Przewidziana lokalizacja (Predict_Loc): {predict_loc}")
        logging.info("--------------------------------------------------")
        
        data_json['Predict_Price'] = predicted_price
        data_json['Adjusted_Price'] = adjusted_price
        data_json['Predict_State'] = predict_state
        data_json['Predict_Loc'] = predict_loc
        
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
    """Główna funkcja skryptu - pętla nasłuchująca."""
    setup_directories()
    try:
        logging.info("Wczytywanie wytrenowanego modelu PyCaret...")
        model = load_model(MODEL_PATH)
        logging.info("Model wczytany pomyślnie.")
        logging.info("Wczytywanie słownika lokalizacji...")
        slownik_df = pd.read_csv(SLOWNIK_PATH, sep=';')
        logging.info("Słownik wczytany pomyślnie.")
        logging.info("Wczytywanie wektoryzatora TF-IDF...")
        vectorizer = joblib.load(VECTORIZER_PATH)
        logging.info("Wektoryzator wczytany pomyślnie.")
    except FileNotFoundError as e:
        logging.error(f"BŁĄD KRYTYCZNY: Nie znaleziono wymaganego pliku: {e.filename}")
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
                    process_json_file(processed_path, filename, model, slownik_df, vectorizer)
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