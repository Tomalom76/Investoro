import time
import os
import json
import pandas as pd
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import sys
import joblib
from pycaret.regression import load_model as pycaret_load_model

# --- Konfiguracja ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_IN_DIR = os.path.join(BASE_DIR, "1JSON_PREDICT_IN")
JSON_OUT_DIR = os.path.join(BASE_DIR, "1JSON_PREDICT_OUT")
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "DATA_OUT")
MODEL_FILENAME = "0_best_LGBM_model_via_script_watch"
VECTORIZER_FILENAME = "location_vectorizer.joblib"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# --- Globalne artefakty ---
final_estimator = None
location_vectorizer = None
FINAL_MODEL_FEATURES = []

def load_artifacts():
    """Ładuje wszystkie niezbędne artefakty: finalny model, vectorizer i listę cech."""
    global final_estimator, location_vectorizer, FINAL_MODEL_FEATURES
    
    model_path_pkl = os.path.join(MODEL_ARTIFACTS_DIR, MODEL_FILENAME + ".pkl")
    vectorizer_path = os.path.join(MODEL_ARTIFACTS_DIR, VECTORIZER_FILENAME)

    if not os.path.exists(model_path_pkl) or not os.path.exists(vectorizer_path):
        logging.critical(f"Brak pliku modelu ({model_path_pkl}) lub vectorizera ({vectorizer_path})!")
        return False

    try:
        # Krok 1: Załaduj pipeline PyCaret
        logging.info(f"Ładowanie pipeline'u z: {model_path_pkl}...")
        temp_pipeline = pycaret_load_model(os.path.join(MODEL_ARTIFACTS_DIR, MODEL_FILENAME), verbose=False)
        
        # Krok 2: Wyciągnij finalny estymator (np. ExtraTreesRegressor)
        final_estimator = temp_pipeline.steps[-1][1]
        
        # Krok 3 (KLUCZOWA POPRAWKA): Wyciągnij listę cech BEZPOŚREDNIO z finalnego modelu
        if hasattr(final_estimator, 'feature_names_in_'):
            FINAL_MODEL_FEATURES = final_estimator.feature_names_in_.tolist()
        else:
            logging.critical("Nie można pobrać listy cech (feature_names_in_) z finalnego modelu!")
            return False
            
        logging.info(f"Wyekstrahowano finalny estymator: {type(final_estimator)}")
        logging.info(f"Pobrano {len(FINAL_MODEL_FEATURES)} nazw cech oczekiwanych przez model.")

        # Krok 4: Załaduj zewnętrzny TfidfVectorizer
        logging.info(f"Ładowanie TfidfVectorizer z: {vectorizer_path}")
        location_vectorizer = joblib.load(vectorizer_path)
        logging.info("Wszystkie artefakty załadowane pomyślnie.")
        return True

    except Exception as e:
        logging.critical(f"Błąd podczas ładowania artefaktów: {e}", exc_info=True)
        return False

def create_features_for_prediction(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Tworzy ramkę danych z cechami gotową do predykcji, idealnie dopasowaną do modelu."""
    df = df_raw.copy()
    
    # DataFrame, do którego będziemy dodawać przetworzone cechy
    df_transformed = pd.DataFrame(index=df.index)

    # === Cechy numeryczne i proste mapowania ===
    offer_from_map = {'Osoba prywatna': 0.0, 'Biuro nieruchomości': 1.0, 'Deweloper': 2.0} 
    type_of_market_map = {'Wtórny': 0.0, 'Pierwotny': 1.0}

    for col_name in FINAL_MODEL_FEATURES:
        if col_name in df.columns:
            if col_name == 'OfferFrom':
                df_transformed[col_name] = df[col_name].map(offer_from_map).fillna(-1.0)
            elif col_name == 'TypeOfMarket':
                df_transformed[col_name] = df[col_name].map(type_of_market_map).fillna(-1.0)
            else:
                df_transformed[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(-1.0)

    # === Cechy z daty (BuiltYear) ===
    if 'BuiltYear' in df.columns:
        dt_series = pd.to_datetime(df['BuiltYear'], errors='coerce', format='%Y')
        if 'BuiltYear_year' in FINAL_MODEL_FEATURES:
            df_transformed['BuiltYear_year'] = dt_series.dt.year.fillna(dt_series.dt.year.median())
        if 'BuiltYear_month' in FINAL_MODEL_FEATURES:
            df_transformed['BuiltYear_month'] = dt_series.dt.month.fillna(dt_series.dt.month.median())

    # === Cechy TF-IDF z lokalizacji ===
    if location_vectorizer and 'Location' in df.columns:
        tfidf_features = location_vectorizer.transform(df['Location'].fillna(''))
        tfidf_feature_names = ['loc_tfidf_' + name for name in location_vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), index=df.index, columns=tfidf_feature_names)
        df_transformed = pd.concat([df_transformed, tfidf_df], axis=1)

    # === Cechy z One-Hot Encoding ===
    categorical_cols_to_ohe = list(set([f.split('_')[0] for f in FINAL_MODEL_FEATURES if '_' in f and not f.startswith('loc_tfidf_') and not f.startswith('BuiltYear_')]))
    
    for cat_col in categorical_cols_to_ohe:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, prefix_sep='_').astype(int)
            df_transformed = pd.concat([df_transformed, dummies], axis=1)

    # === OSTATECZNE I NAJWAŻNIEJSZE DOPASOWANIE ===
    df_aligned = df_transformed.reindex(columns=FINAL_MODEL_FEATURES, fill_value=0)
    
    logging.info(f"Przygotowano ostateczny zbiór cech do predykcji o kształcie: {df_aligned.shape}")
    return df_aligned

def process_single_json_file(json_file_path: str):
    if not all([final_estimator, location_vectorizer, FINAL_MODEL_FEATURES]):
        logging.error("Kluczowe artefakty niezaładowane. Pomijam plik.")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_json_content = json.load(f)
    except Exception as e:
        logging.error(f"Błąd wczytywania pliku JSON {json_file_path}: {e}")
        return

    data_records_list, original_json_structure_key, is_single_item = [], None, False
    if isinstance(raw_json_content, dict):
        if len(raw_json_content) == 1 and isinstance(list(raw_json_content.values())[0], list):
            original_json_structure_key = list(raw_json_content.keys())[0]
            data_records_list = raw_json_content[original_json_structure_key]
        else: data_records_list, is_single_item = [raw_json_content], True
    elif isinstance(raw_json_content, list): data_records_list = raw_json_content
    else: logging.error(f"Nieobsługiwany format JSON."); return

    if not data_records_list: logging.warning(f"Plik JSON pusty."); return
    
    df_initial = pd.DataFrame(data_records_list)
    logging.info(f"Utworzono DataFrame z JSON o kształcie: {df_initial.shape}.")
        
    final_json_to_save = {}
    try:
        data_for_prediction = create_features_for_prediction(df_initial)
        
        logging.info("Wykonywanie predykcji...")
        predicted_values = final_estimator.predict(data_for_prediction)
        
        df_initial['PredictedPrice_raw'] = predicted_values

        output_records_final = []
        for index, row in df_initial.iterrows():
            original_record = data_records_list[index]
            new_record_items = list(original_record.items())
            
            raw_pred = row['PredictedPrice_raw']
            predicted_price = f"{float(raw_pred):,.0f}" if pd.notna(raw_pred) else "N/A"
            
            price_key_index = -1
            for i, (key, val) in enumerate(new_record_items):
                if str(key).lower() == 'price':
                    price_key_index = i
                    break
            
            prediction_tuple = ('PredictedPriceByModel', predicted_price)
            if price_key_index != -1:
                new_record_items.insert(price_key_index + 1, prediction_tuple)
            else:
                new_record_items.append(prediction_tuple)
            
            output_records_final.append(dict(new_record_items))

        if output_records_final:
             logging.info(f"--- PRZEWIDZIANA CENA: {output_records_final[0].get('PredictedPriceByModel', 'B/D')} ---")

        if original_json_structure_key: final_json_to_save = {original_json_structure_key: output_records_final}
        elif is_single_item: final_json_to_save = output_records_final[0] if output_records_final else {}
        else: final_json_to_save = output_records_final

    except Exception as e:
        logging.error(f"Błąd podczas predykcji/składania wyniku dla {json_file_path}: {e}", exc_info=True)
        final_json_to_save = {"error": f"Błąd przetwarzania: {str(e)}"}

    base_name, ext_name = os.path.splitext(os.path.basename(json_file_path))
    output_file_path = os.path.join(JSON_OUT_DIR, f"{base_name}_out{ext_name}")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(final_json_to_save, f_out, ensure_ascii=False, indent=4)
        logging.info(f"Wynik zapisano do: {output_file_path}")
    except Exception as e:
        logging.error(f"Błąd zapisu wyniku JSON {output_file_path}: {e}")

class JsonFileChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            logging.info(f"Wykryto nowy plik JSON: {event.src_path}")
            time.sleep(1.5)
            if not os.path.exists(event.src_path) or os.path.getsize(event.src_path) == 0:
                logging.warning(f"Plik {event.src_path} nie istnieje/pusty. Pomijam.")
                return
            process_single_json_file(event.src_path)

if __name__ == "__main__":
    logging.info("Inicjalizacja skryptu predykcyjnego.")
    os.makedirs(JSON_IN_DIR, exist_ok=True)
    os.makedirs(JSON_OUT_DIR, exist_ok=True)
    
    if not load_artifacts():
        logging.critical("Nie załadowano artefaktów. Koniec działania.")
        sys.exit(1)
    
    event_handler = JsonFileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, JSON_IN_DIR, recursive=False)
    logging.info(f"Nasłuchiwanie na .json w: {os.path.abspath(JSON_IN_DIR)}")
    observer.start()
    try:
        while True: time.sleep(5)
    except KeyboardInterrupt: logging.info("Zatrzymano przez użytkownika.")
    finally:
        observer.stop()
        observer.join()
        logging.info("Skrypt zakończył działanie.")