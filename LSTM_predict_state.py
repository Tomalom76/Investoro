import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Konfiguracja ścieżek ---
MODEL_PATH = 'model_lstm_stan.keras'
TOKENIZER_PATH = 'tokenizer.json'
PREPROCESSOR_PATH = 'preprocessor.joblib'
LABEL_MAPPING_PATH = 'label_mapping.json'
INPUT_DIR = '2JSON_STATE'
OUTPUT_DIR = '2JSON_STATE_OUT'

# --- Definicje kolumn (muszą być takie same jak podczas treningu) ---
NUMERIC_FEATURES = ['Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'year']
CATEGORICAL_FEATURES = ['BuildingType', 'OfferFrom', 'TypeOfMarket']
MAX_LEN = 200

class PredictionHandler(FileSystemEventHandler):
    def __init__(self, model, tokenizer, preprocessor, label_mapping):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.label_mapping = label_mapping
        print("Handler gotowy do pracy.")

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            print(f"\n--- Nowy plik wykryty: {os.path.basename(event.src_path)} ---")
            time.sleep(1)
            self.process_file(event.src_path)

    def process_file(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
            data_to_predict = pd.DataFrame(data_json)
            original_data_for_display = data_to_predict.copy()
            
            print(f"Wczytano {len(data_to_predict)} rekordów.")

            # --- PRZYGOTOWANIE DANYCH ---
            data_to_predict['Description'] = data_to_predict.get('Description', '').astype(str).fillna('brak opisu')
            sequences = self.tokenizer.texts_to_sequences(data_to_predict['Description'])
            X_text = pad_sequences(sequences, maxlen=MAX_LEN)
            
            # --- KLUCZOWA POPRAWKA: Tworzymy 'year' PRZED użyciem listy numeric_features ---
            data_to_predict['year'] = pd.to_datetime(data_to_predict.get('BuiltYear'), errors='coerce').dt.year

            # Uzupełnianie braków
            for col in NUMERIC_FEATURES:
                data_to_predict[col] = pd.to_numeric(data_to_predict.get(col), errors='coerce').fillna(0)
            for col in CATEGORICAL_FEATURES:
                data_to_predict[col] = data_to_predict.get(col, 'BRAK').fillna('BRAK')
            
            # Transformacja
            X_tabular = self.preprocessor.transform(data_to_predict[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

            # --- Predykcja ---
            print("Wykonuję predykcje...")
            probabilities = self.model.predict([X_text, X_tabular])
            label_indices = np.argmax(probabilities, axis=1)
            predicted_conditions = [self.label_mapping[str(i)] for i in label_indices]

            original_data_for_display['Predict_State'] = predicted_conditions
            original_data_for_display['Predict_Score'] = np.max(probabilities, axis=1)
            print("Predykcje zakończone.")

            # --- ZMIANA 1: WYŚWIETLANIE W KONSOLI ---
            print("\n--- WYNIKI PREDYKCJI ---")
            display_cols = [col for col in ['Area', 'Price', 'Location', 'Predict_State', 'Predict_Score'] if col in original_data_for_display.columns]
            print(original_data_for_display[display_cols].to_string())

            # --- ZMIANA 2: ZAPIS DO JSON ---
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
        error_dir = os.path.join(INPUT_DIR, 'error')
        os.makedirs(error_dir, exist_ok=True)
        try:
            shutil.move(file_path, os.path.join(error_dir, os.path.basename(file_path)))
        except Exception as e:
            print(f"Nie udało się przenieść pliku do folderu error: {e}")

def start_watcher():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        print("Wczytuję model Keras...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Wczytuję tokenizer...")
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f: tokenizer = tokenizer_from_json(json.load(f))
        print("Wczytuję preprocessor...")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Wczytuję mapowanie etykiet...")
        with open(LABEL_MAPPING_PATH, 'r') as f: label_mapping = json.load(f)
    except FileNotFoundError as e:
        print(f"KRYTYCZNY BŁĄD: Brak jednego z plików modelu: {e}. Upewnij się, że wszystkie pliki (keras, json, joblib) są w tym samym folderze co skrypt.")
        return
        
    print("\nModel i wszystkie komponenty wczytane pomyślnie.")
    
    event_handler = PredictionHandler(model, tokenizer, preprocessor, label_mapping)
    
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    
    print(f"--- Nasłuchuję na nowe pliki .json w folderze '{INPUT_DIR}' ---")
    print("Naciśnij CTRL+C, aby zakończyć.")
    
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()