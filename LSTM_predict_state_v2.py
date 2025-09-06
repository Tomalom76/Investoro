# predict_state_from_json.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json
import time
import os
import re
import datetime as dt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Konfiguracja ścieżek do wszystkich 8 artefaktów z v5 ---
MODEL_PATH = 'model_lstm_stan.keras'
TOKENIZER_PATH = 'tokenizer.json'
PREPROCESSOR_PATH = 'preprocessor.joblib'
SVD_PATH = 'svd_256.joblib'
LABEL_MAPPING_PATH = 'label_mapping.json'
COLS_FOR_PRED_PATH = 'columns_for_prediction.joblib'
TOP_K_PATH = 'top_k_params.joblib'
WINSOR_PATH = 'winsor_params.joblib'
MEDIAN_YEAR_PATH = 'median_year.joblib'

# --- Konfiguracja folderów ---
INPUT_DIR = '2JSON_STATE'
OUTPUT_DIR = '2JSON_STATE_OUT'
ERROR_DIR = os.path.join(INPUT_DIR, 'error')

# --- Parametry modelu ---
MAX_LEN = 250

# --- Funkcje pomocnicze do preprocessingu (identyczne jak w notebooku) ---
def clean_text(s: str) -> str:
    s = (s or "").lower()
    patterns = [r'oferta nie stanowi.*?oferty w rozumieniu kodeksu cywilnego', r'prosz[ąa] o kontakt.*', r'tylko u nas.*', r'nie pobieramy prowizji.*']
    for p in patterns: s = re.sub(p, ' ', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3,4}\b', ' ', s)
    s = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', s)
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_year(val):
    s = str(val)
    m = re.search(r'(?:18|19|20)\d{2}', s)
    if m:
        y = int(m.group(0))
        if 1800 <= y <= dt.datetime.now().year + 1: return y
    return np.nan

def norm_market(v):
    v = (v or '').lower()
    if 'pierwot' in v: return 'pierwotny'
    if 'wtór' in v or 'wtorn' in v: return 'wtórny'
    return v if v else 'unknown'

class PredictionHandler(FileSystemEventHandler):
    def __init__(self, artifacts):
        self.artifacts = artifacts
        print("Handler gotowy do pracy.")

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            print(f"\n--- Nowy plik wykryty: {os.path.basename(event.src_path)} ---")
            time.sleep(1) # Czekamy na pełne zapisanie pliku
            self.process_file(event.src_path)

    def process_file(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
            
            # Konwersja do DataFrame, nawet jeśli jest to pojedynczy słownik
            infer_df = pd.DataFrame(data_json if isinstance(data_json, list) else [data_json])
            original_data_for_display = infer_df.copy()
            
            print(f"Wczytano {len(infer_df)} rekordów.")

            # --- KROK 1: PRZYGOTOWANIE DANYCH (SPÓJNE Z NOTEBOOKIEM) ---
            print("Rozpoczynam spójny preprocessing...")
            
            # Czyszczenie tekstu
            infer_df['Description'] = infer_df['Description'].fillna('').astype(str).apply(clean_text)
            
            # Konwersja kolumn numerycznych
            num_cols_to_convert = ['Area','Price','NumberOfRooms','Floor','Floors']
            for col in num_cols_to_convert:
                infer_df[col] = pd.to_numeric(infer_df.get(col), errors='coerce')

            # Przetwarzanie roku i imputacja medianą z treningu
            infer_df['year'] = infer_df.get('BuiltYear').apply(extract_year)
            infer_df['year'] = infer_df['year'].fillna(self.artifacts['median_year'])

            # Inżynieria cech
            infer_df['price_per_m2'] = np.where((infer_df['Area']>0) & (infer_df['Price']>0), infer_df['Price']/infer_df['Area'], np.nan)
            
            # Winsoryzacja z użyciem parametrów z treningu
            for col, params in self.artifacts['winsor_params'].items():
                if col in infer_df.columns:
                    infer_df[col] = infer_df[col].clip(lower=params['lower'], upper=params['upper'])
            
            # Cechy logarytmiczne
            infer_df['log_area']  = np.log1p(infer_df['Area'])
            infer_df['log_price'] = np.log1p(infer_df['Price'])
            infer_df['log_ppm']   = np.log1p(infer_df['price_per_m2'])

            # Przetwarzanie kategorii
            cat_cols_to_process = ['BuildingType','OfferFrom','TypeOfMarket']
            for col in cat_cols_to_process:
                infer_df[col] = infer_df.get(col).fillna('unknown').astype(str).str.strip().str.lower()
            
            infer_df['TypeOfMarket'] = infer_df['TypeOfMarket'].apply(norm_market)
            
            # Redukcja rzadkich kategorii (top-K) z użyciem parametrów z treningu
            for col, top_list in self.artifacts['top_k_params'].items():
                if col in infer_df.columns:
                    infer_df[col] = infer_df[col].where(infer_df[col].isin(top_list), 'other')
            
            print("Preprocessing zakończony.")

            # --- KROK 2: TRANSFORMACJA I PREDYKCJA ---
            print("Transformacja danych i predykcja...")
            
            # Tekst
            sequences = self.artifacts['tokenizer'].texts_to_sequences(infer_df['Description'])
            X_text = pad_sequences(sequences, maxlen=MAX_LEN)
            
            # Tabela
            X_tab_sparse = self.artifacts['preprocessor'].transform(infer_df[self.artifacts['columns_for_prediction']])
            X_tabular = self.artifacts['svd'].transform(X_tab_sparse)
            
            # Predykcja
            probabilities = self.artifacts['model'].predict([X_text, X_tabular])
            label_indices = np.argmax(probabilities, axis=1)
            predicted_conditions = [self.artifacts['label_mapping'].get(str(i), 'UNKNOWN') for i in label_indices]
            
            original_data_for_display['Predict_State'] = predicted_conditions
            original_data_for_display['Predict_Score'] = np.max(probabilities, axis=1).tolist() # tolist() dla JSON
            print("Predykcje zakończone.")

            # --- KROK 3: PREZENTACJA I ZAPIS WYNIKÓW ---
            print("\n--- WYNIKI PREDYKCJI ---")
            display_cols = [col for col in ['Area', 'Price', 'Location', 'Predict_State', 'Predict_Score'] if col in original_data_for_display.columns]
            print(original_data_for_display[display_cols].to_string())

            base_filename = os.path.basename(json_path)
            output_filename = os.path.splitext(base_filename)[0] + '_processed.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            results_to_save = original_data_for_display.to_dict(orient='records')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, ensure_ascii=False, indent=4)
            print(f"\nWyniki zapisano do pliku: {output_path}")

            os.remove(json_path)
            print(f"Usunięto oryginalny plik: {base_filename}")

        except Exception as e:
            print(f"BŁĄD: {e}")
            self.move_to_error(json_path)
        
        print("\n--- Nasłuchiwanie na kolejne pliki... ---")
    
    def move_to_error(self, file_path):
        import shutil
        os.makedirs(ERROR_DIR, exist_ok=True)
        try:
            shutil.move(file_path, os.path.join(ERROR_DIR, os.path.basename(file_path)))
            print(f"Plik przeniesiono do folderu error: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Nie udało się przenieść pliku do folderu error: {e}")

def load_artifacts():
    print("Wczytywanie artefaktów...")
    artifacts = {}
    try:
        artifacts['model'] = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            artifacts['tokenizer'] = tokenizer_from_json(json.dumps(tokenizer_data) if isinstance(tokenizer_data, dict) else tokenizer_data)
        artifacts['preprocessor'] = joblib.load(PREPROCESSOR_PATH)
        artifacts['svd'] = joblib.load(SVD_PATH)
        with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f: artifacts['label_mapping'] = json.load(f)
        artifacts['columns_for_prediction'] = joblib.load(COLS_FOR_PRED_PATH)
        artifacts['top_k_params'] = joblib.load(TOP_K_PATH)
        artifacts['winsor_params'] = joblib.load(WINSOR_PATH)
        artifacts['median_year'] = joblib.load(MEDIAN_YEAR_PATH)
        print("Wszystkie artefakty wczytane pomyślnie.")
        return artifacts
    except FileNotFoundError as e:
        print(f"KRYTYCZNY BŁĄD: Brak jednego z plików modelu: {e}. Upewnij się, że wszystkie 9 artefaktów jest w tym samym folderze co skrypt.")
        return None

def start_watcher():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ERROR_DIR, exist_ok=True)
    
    artifacts = load_artifacts()
    if artifacts is None:
        return
        
    event_handler = PredictionHandler(artifacts)
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    
    print(f"\n--- Nasłuchuję na nowe pliki .json w folderze '{INPUT_DIR}' ---")
    print("--- Naciśnij CTRL+C, aby zakończyć. ---")
    
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()