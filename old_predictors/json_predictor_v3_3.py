import time
import os
import json
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import sys
import mlflow 
from pycaret.regression import load_model as pycaret_load_model
from pycaret.regression import predict_model as pycaret_predict_model
import numpy as np
import joblib # Do ładowania TfidfVectorizer

# --- Konfiguracja ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_IN_DIR = os.path.join(BASE_DIR, "1JSON_PREDICT_IN") # Zmieniona nazwa dla jasności
JSON_OUT_DIR = os.path.join(BASE_DIR, "1JSON_PREDICT_OUT") # Zmieniona nazwa dla jasności

# Użyj ścieżki lokalnej, jeśli model i vectorizer są zapisywane lokalnie przez watcher.py
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "DATA_OUT") # Katalog, gdzie watcher zapisuje artefakty
MODEL_FILENAME_FROM_WATCHER = "0_best_LGBM_model_via_script_watch" # Nazwa modelu z watchera
VECTORIZER_FILENAME = "location_vectorizer.joblib" # Nazwa zapisana przez watchera

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

loaded_model_global = None
loaded_vectorizer_global = None

# === Schemat danych wejściowych z JSON - dostosowany do modelu z notebooka EDA_Projekt1_v6_3FULL ===
# To są kolumny, które chcesz przyjmować z JSON. 
# Pipeline PyCaret zajmie się resztą (np. ignorowaniem nieużywanych kolumn, jeśli są obecne).
# Najważniejsze są te, które są faktycznie używane w `setup()` jako cechy.
EXPECTED_INPUT_SCHEMA = {
    # Kluczowe cechy używane w `setup` z notebooka
    'SaleId': pd.Int64Dtype(), # Dla identyfikacji, może być ignorowana przez model
    'Area': float,
    'NumberOfRooms': pd.Int64Dtype(),
    'Floor': pd.Int64Dtype(),
    'Floors': pd.Int64Dtype(),
    'CommunityScore': float,
    'BuiltYear': object, # Wczytamy jako object, potem konwersja na datetime
    'Location': str,     # Kluczowe dla TF-IDF

    # Cechy kategoryczne zdefiniowane w notebooku
    'BuildingType': str, # PyCaret przekształci na category
    'BuildingCondition': str,
    'TypeOfMarket': str,
    'OwnerType': str,
    'Type': str, # Ogólny typ nieruchomości
    'OfferFrom': str,
    
    # Numeryczne/identyfikatory przekształcane na string w notebooku
    'VoivodeshipNumber': str,
    'CountyNumber': str,
    'CommunityNumber': str,
    'KindNumber': str,
    'RegionNumber': str,
    'StreetNumber': str, # Dodane w notebooku do konwersji na string

    # Inne potencjalnie przydatne kolumny, które mogą być w JSON
    'Title': str,
    'Description': str, # Chociaż może być ignorowana, JSON może ją zawierać
    
    # Te kolumny są na liście `ignore_features` w notebooku, więc model ich nie użyje,
    # ale JSON może je dostarczyć. Wpisujemy je, aby poprawnie je sparsować,
    # a pipeline PyCaret je zignoruje, jeśli są na liście ignorowanych.
    'OriginalId': str, 
    'PortalId': pd.Int64Dtype(),
    'OfferPrice': object, 
    'RealPriceAfterRenovation': object, 
    'OriginalPrice': object, 
    'PricePerSquareMeter': float, # Może być przydatna do walidacji, ale model jej nie bierze
    'DateAddedToDatabase': str, 
    'DateAdded': str, 
    'DateLastModification': str, 
    'DateLastRaises': str, 
    'NewestDate': str,
    'AvailableFrom': str, 
    'Link': str, 
    'Phone': object, 
    'MainImage': str, 
    'OtherImages': str,
    'NumberOfDuplicates': pd.Int64Dtype(), 
    'NumberOfRaises': pd.Int64Dtype(), 
    'NumberOfModifications': pd.Int64Dtype(), 
    'IsDuplicatePriceLower': pd.BooleanDtype(), 
    'IsDuplicatePrivateOwner': pd.BooleanDtype(), 
    'Score': pd.Int64Dtype(), 
    'ScorePrecision': pd.Int64Dtype(), 
    'NumberOfCommunityComments': pd.Int64Dtype(), 
    'NumberOfCommunityOpinions': pd.Int64Dtype(),
    'Archive': str, 
    'SubRegionNumber': str, 
    'EncryptedId': str,
}
# Usuwamy 'Price' ze schematu wejściowego, bo to będziemy przewidywać
# if 'Price' in EXPECTED_INPUT_SCHEMA: del EXPECTED_INPUT_SCHEMA['Price'] 
# W tym przypadku 'Price' nie ma w schemacie, co jest poprawne.

def normalize_column_name(col_name_from_json):
    # Ta funkcja wydaje się już dość dobra i mapuje wiele popularnych wariantów.
    # Sprawdź, czy pokrywa wszystkie klucze, których możesz się spodziewać w JSON.
    col_name = str(col_name_from_json).lower()
    mapping = {
        "saleid": "SaleId", "originalid": "OriginalId", "portalid": "PortalId", "title": "Title",
        "description": "Description", "area": "Area", "offerprice": "OfferPrice",
        "realpriceafterrenovation": "RealPriceAfterRenovation", "originalprice": "OriginalPrice",
        "pricepersquaremeter": "PricePerSquareMeter", "numberofrooms": "NumberOfRooms",
        "builtyear": "BuiltYear", "type": "Type", "buildingtype": "BuildingType",
        "buildingcondition": "BuildingCondition", "offerfrom": "OfferFrom", "floor": "Floor",
        "floors": "Floors", "typeofmarket": "TypeOfMarket", "ownertype": "OwnerType",
        "dateaddedtodatabase": "DateAddedToDatabase", "dateadded": "DateAdded",
        "datelastmodification": "DateLastModification", "datelastraises": "DateLastRaises",
        "newestdate": "NewestDate", "availablefrom": "AvailableFrom", "link": "Link",
        "phone": "Phone", "mainimage": "MainImage", "otherimages": "OtherImages",
        "numberofduplicates": "NumberOfDuplicates", "numberofraises": "NumberOfRaises",
        "numberofmodifications": "NumberOfModifications",
        "isduplicatepricelower": "IsDuplicatePriceLower",
        "isduplicateprivateowner": "IsDuplicatePrivateOwner", "score": "Score",
        "scoreprecision": "ScorePrecision", "communityscore": "CommunityScore",
        "numberofcommunitycomments": "NumberOfCommunityComments",
        "numberofcommunityopinions": "NumberOfCommunityOpinions", "archive": "Archive",
        "location": "Location", "voivodeshipnumber": "VoivodeshipNumber",
        "countynumber": "CountyNumber", "communitynumber": "CommunityNumber",
        "kindnumber": "KindNumber", "regionnumber": "RegionNumber",
        "subregionnumber": "SubRegionNumber", "streetnumber": "StreetNumber",
        "encryptedid": "EncryptedId",
    }
    if col_name in mapping: return mapping[col_name]
    # Jeśli nazwa jest już w poprawnym formacie (PascalCase), zwróć ją
    if col_name_from_json in EXPECTED_INPUT_SCHEMA: return col_name_from_json
    # Domyślna próba konwersji snake_case lub alllower na PascalCase
    if "_" in col_name_from_json: return "".join(word.capitalize() for word in col_name_from_json.split('_'))
    if col_name_from_json.islower(): return col_name_from_json.capitalize() # Prosty capitalize dla alllower
    return col_name_from_json # Zwróć oryginalną, jeśli nic nie pasuje

def load_local_pycaret_model_once():
    global loaded_model_global
    if loaded_model_global is None:
        model_path = os.path.join(MODEL_ARTIFACTS_DIR, MODEL_FILENAME_FROM_WATCHER)
        if not os.path.exists(model_path + ".pkl"): # PyCaret zapisuje z .pkl
            logging.error(f"Plik modelu {model_path}.pkl nie został znaleziony!")
            return None
        try:
            logging.info(f"Ładowanie lokalnego modelu PyCaret z: {model_path}")
            loaded_model_global = pycaret_load_model(model_path, verbose=False)
            logging.info("Lokalny model PyCaret załadowany pomyślnie.")
        except Exception as e:
            logging.error(f"Nie udało się załadować lokalnego modelu PyCaret z {model_path}: {e}")
            import traceback; logging.error(traceback.format_exc())
            loaded_model_global = None
    return loaded_model_global

def load_location_vectorizer_once():
    global loaded_vectorizer_global
    if loaded_vectorizer_global is None:
        vectorizer_path = os.path.join(MODEL_ARTIFACTS_DIR, VECTORIZER_FILENAME)
        if not os.path.exists(vectorizer_path):
            logging.error(f"Plik TfidfVectorizer {vectorizer_path} nie został znaleziony!")
            return None
        try:
            logging.info(f"Ładowanie TfidfVectorizer z: {vectorizer_path}")
            loaded_vectorizer_global = joblib.load(vectorizer_path)
            logging.info("TfidfVectorizer załadowany pomyślnie.")
        except Exception as e:
            logging.error(f"Nie udało się załadować TfidfVectorizer z {vectorizer_path}: {e}")
            loaded_vectorizer_global = None
    return loaded_vectorizer_global

# Funkcja skopiowana z notebooka/watchera, dostosowana
def convert_column_types_for_prediction(df_to_convert):
    df_copy = df_to_convert.copy()
    str_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']
    for col in str_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).replace({'nan': pd.NA, '<NA>': pd.NA, 'None': pd.NA})
    
    if 'BuiltYear' in df_copy.columns:
        # Konwertuj na numeryczny (rok), a potem na datetime, obsługa błędów
        # Zapewnij, że jest to float przed próbą konwersji na int, jeśli są NaN
        df_copy['BuiltYear'] = pd.to_numeric(df_copy['BuiltYear'], errors='coerce')
        # Dla tych, co są liczbami, spróbuj formatu %Y
        # Te, które nie są, pozostaną NaT
        df_copy['BuiltYear'] = pd.to_datetime(df_copy['BuiltYear'], format='%Y', errors='coerce')
    return df_copy

def create_dirs_if_not_exist():
    os.makedirs(JSON_IN_DIR, exist_ok=True)
    os.makedirs(JSON_OUT_DIR, exist_ok=True)
    logging.info(f"Katalogi '{JSON_IN_DIR}' i '{JSON_OUT_DIR}' gotowe.")

def process_json_file(json_file_path):
    model_pipeline = load_local_pycaret_model_once()
    location_vectorizer = load_location_vectorizer_once()

    if model_pipeline is None:
        logging.error("Model PyCaret nie jest załadowany, pomijanie predykcji dla pliku: %s", json_file_path)
        return
    if location_vectorizer is None:
        logging.error("TfidfVectorizer nie jest załadowany, pomijanie predykcji dla pliku: %s", json_file_path)
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            input_json_data_raw = json.load(f)
        logging.info(f"Pomyślnie wczytano plik JSON: {json_file_path}")
    except Exception as e:
        logging.error(f"Błąd podczas wczytywania lub parsowania pliku JSON {json_file_path}: {e}")
        return

    is_single_object_under_key = False
    is_single_top_level_object = False
    original_structure_key = None

    if isinstance(input_json_data_raw, dict) and len(input_json_data_raw) == 1:
        original_structure_key = list(input_json_data_raw.keys())[0]
        if isinstance(input_json_data_raw[original_structure_key], list):
            input_json_data_list = input_json_data_raw[original_structure_key]
        elif isinstance(input_json_data_raw[original_structure_key], dict):
             input_json_data_list = [input_json_data_raw[original_structure_key]]
             is_single_object_under_key = True
        else: 
            input_json_data_list = [input_json_data_raw]
            is_single_top_level_object = True
    elif isinstance(input_json_data_raw, dict):
        input_json_data_list = [input_json_data_raw]
        is_single_top_level_object = True
    elif isinstance(input_json_data_raw, list):
        input_json_data_list = input_json_data_raw
    else:
        logging.error(f"Nieobsługiwany główny format danych w pliku JSON: {json_file_path}.")
        return
    
    if not input_json_data_list:
        logging.warning(f"Plik JSON {json_file_path} nie zawiera danych do przetworzenia.")
        return

    try:
        input_df_raw = pd.DataFrame(input_json_data_list)
        input_df_raw.columns = [normalize_column_name(col) for col in input_df_raw.columns]
        logging.info(f"Nazwy kolumn po normalizacji: {input_df_raw.columns.tolist()}")
        
        df_for_processing = pd.DataFrame()

        # Stosowanie schematu i konwersji typów
        for col_expected, expected_dtype in EXPECTED_INPUT_SCHEMA.items():
            if col_expected in input_df_raw.columns:
                current_col_data = input_df_raw[col_expected]
                # Wstępne czyszczenie dla typów liczbowych/boolowskich
                if expected_dtype in [float, pd.Int64Dtype(), pd.Float64Dtype(), pd.BooleanDtype()]:
                    current_col_data = current_col_data.replace(['', 'null', None, 'None'], np.nan)
                
                try:
                    if col_expected == 'BuiltYear': # Specjalna obsługa dla BuiltYear (już object, konwersja później)
                        df_for_processing[col_expected] = current_col_data 
                    elif expected_dtype == str:
                        df_for_processing[col_expected] = current_col_data.astype(str).fillna("")
                    elif expected_dtype == pd.Int64Dtype():
                        df_for_processing[col_expected] = pd.to_numeric(current_col_data, errors='coerce').astype(pd.Int64Dtype())
                    elif expected_dtype == float: # lub pd.Float64Dtype()
                        df_for_processing[col_expected] = pd.to_numeric(current_col_data, errors='coerce').astype(float)
                    elif expected_dtype == pd.BooleanDtype():
                        bool_map = {'true': True, '1': True, 1: True, 'false': False, '0': False, 0: False, 
                                    '': pd.NA, 'nan': pd.NA, np.nan: pd.NA} # Dodano 'nan'
                        df_for_processing[col_expected] = current_col_data.fillna('').astype(str).str.lower().map(bool_map).astype(pd.BooleanDtype())
                    else: # Domyślna próba konwersji
                        df_for_processing[col_expected] = current_col_data.astype(expected_dtype)
                except Exception as e_conv:
                    logging.warning(f"Problem z konwersją kolumny '{col_expected}' do {expected_dtype}. Ustawiam na pd.NA/None. Błąd: {e_conv}")
                    df_for_processing[col_expected] = pd.Series([pd.NA] * len(input_df_raw), dtype=object) # dtype=object dla bezpieczeństwa
            else:
                logging.warning(f"Brakująca oczekiwana kolumna '{col_expected}' w JSON. Dodaję jako pustą/pd.NA.")
                df_for_processing[col_expected] = pd.Series([pd.NA] * len(input_df_raw), dtype=object if expected_dtype == str else expected_dtype)

        # Zapewnienie istnienia wszystkich oczekiwanych kolumn (choć pipeline PyCaret powinien to obsłużyć)
        for col_final in EXPECTED_INPUT_SCHEMA.keys():
            if col_final not in df_for_processing.columns:
                expected_dtype = EXPECTED_INPUT_SCHEMA[col_final]
                df_for_processing[col_final] = pd.Series([pd.NA] * len(df_for_processing), dtype=object if expected_dtype == str else expected_dtype)
        
        # --- POCZĄTEK PRZETWARZANIA DANYCH (jak w notebooku przed setup) ---
        # 1. Konwersja typów (VoivodeshipNumber na str, BuiltYear na datetime itp.)
        df_for_processing = convert_column_types_for_prediction(df_for_processing)
        logging.info("Zastosowano convert_column_types_for_prediction.")

        # 2. Przetwarzanie TF-IDF dla kolumny 'Location'
        if 'Location' in df_for_processing.columns:
            location_clean = df_for_processing['Location'].fillna('').astype(str)
            location_tfidf_features = location_vectorizer.transform(location_clean)
            
            try:
                tfidf_feature_names = location_vectorizer.get_feature_names_out()
            except AttributeError:
                tfidf_feature_names = location_vectorizer.get_feature_names_()
            
            location_tfidf_df = pd.DataFrame(
                location_tfidf_features.toarray(),
                columns=['loc_tfidf_' + name for name in tfidf_feature_names],
                index=df_for_processing.index
            )
            
            # Połącz cechy TF-IDF, usuń oryginalną 'Location'
            df_for_model_input = pd.concat(
                [df_for_processing.drop(columns=['Location'], errors='ignore'), location_tfidf_df],
                axis=1
            )
            logging.info(f"Dodano {location_tfidf_df.shape[1]} cech TF-IDF. Rozmiar df_for_model_input: {df_for_model_input.shape}")
        else:
            logging.warning("Brak kolumny 'Location' w danych wejściowych, pomijam TF-IDF.")
            df_for_model_input = df_for_processing.copy() # Użyj danych bez TF-IDF

        # Upewnij się, że wszystkie kolumny oczekiwane przez model są obecne, nawet jeśli puste
        # Pipeline PyCaret powinien obsłużyć brakujące kolumny, jeśli były w treningu,
        # ale dla pewności można dodać te, których używa pipeline, a nie ma w df_for_model_input.
        # Na tym etapie zakładamy, że df_for_model_input ma już poprawny zestaw kolumn
        # po dodaniu TF-IDF i usunięciu 'Location'.

        logging.info(f"Dane przekazywane do modelu (df_for_model_input) dla pliku {os.path.basename(json_file_path)} (pierwsze 5 wierszy):\n{df_for_model_input.head().to_string(max_rows=5, max_cols=15)}")
        logging.info(f"Typy danych w df_for_model_input:\n{df_for_model_input.dtypes.to_string()}")
        
        # --- KONIEC PRZETWARZANIA DANYCH ---

        predictions_df = pycaret_predict_model(model_pipeline, data=df_for_model_input)
        
        output_json_data_list_modified = []
        for i, original_record_dict in enumerate(input_json_data_list):
            new_record_output = original_record_dict.copy() 
            predicted_value_raw = predictions_df.loc[i, 'prediction_label'] if i < len(predictions_df) and 'prediction_label' in predictions_df.columns else None
            
            formatted_prediction = ""
            if predicted_value_raw is not None:
                try: formatted_prediction = f"{float(predicted_value_raw):,.0f}"
                except (ValueError, TypeError): formatted_prediction = str(predicted_value_raw) # Lepsza obsługa błędów
            
            # Wstaw PredictedPriceByModel obok klucza 'Price' lub 'price' jeśli istnieje
            price_key_found = None
            for key in new_record_output.keys():
                if key.lower() == 'price':
                    price_key_found = key
                    break
            
            if price_key_found:
                items = list(new_record_output.items())
                try:
                    idx = [item[0].lower() for item in items].index('price')
                    items.insert(idx + 1, ('PredictedPriceByModel', formatted_prediction))
                    new_record_output = dict(items)
                except ValueError: 
                    new_record_output['PredictedPriceByModel'] = formatted_prediction # Fallback
            else: 
                new_record_output['PredictedPriceByModel'] = formatted_prediction
            output_json_data_list_modified.append(new_record_output)
        
        # Logowanie predykcji
        if predictions_df is not None and 'prediction_label' in predictions_df.columns:
            log_msg_predictions = []
            for idx in predictions_df.index: # Użyj indeksu z predictions_df
                sale_id_for_log = df_for_model_input.loc[idx, 'SaleId'] if 'SaleId' in df_for_model_input.columns else f"Rekord_idx_{idx}"
                raw_pred_val = predictions_df.loc[idx, 'prediction_label']
                formatted_pred_for_log = ""
                if raw_pred_val is not None:
                    try: formatted_pred_for_log = f"{float(raw_pred_val):,.0f}"
                    except (ValueError, TypeError): formatted_pred_for_log = str(raw_pred_val)
                log_msg_predictions.append(f"SaleId '{sale_id_for_log}': {formatted_pred_for_log} (surowa: {raw_pred_val})")
            logging.info(f"Przewidziane ceny dla pliku {os.path.basename(json_file_path)}:\n" + "\n".join(log_msg_predictions))


        if is_single_object_under_key and original_structure_key:
            output_data_to_save = {original_structure_key: output_json_data_list_modified[0]}
        elif is_single_top_level_object:
            output_data_to_save = output_json_data_list_modified[0]
        else: 
            output_data_to_save = output_json_data_list_modified

    except Exception as e:
        logging.error(f"Błąd podczas przygotowywania danych lub predykcji dla pliku {json_file_path}: {e}")
        import traceback; logging.error(traceback.format_exc())
        return

    original_basename_no_ext, ext = os.path.splitext(os.path.basename(json_file_path))
    output_filename_with_suffix = f"{original_basename_no_ext}_out{ext}"
    output_json_path = os.path.join(JSON_OUT_DIR, output_filename_with_suffix)
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(output_data_to_save, f_out, ensure_ascii=False, indent=4)
        logging.info(f"Pomyślnie przetworzono i zapisano wynik do: {output_json_path}")
    except Exception as e:
        logging.error(f"Błąd podczas zapisywania wynikowego pliku JSON {output_json_path}: {e}")

class JsonFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            logging.info(f"Zdarzenie! Wykryto nowy plik JSON: {event.src_path}")
            time.sleep(2)
            if not os.path.exists(event.src_path) or os.path.getsize(event.src_path) == 0:
                logging.warning(f"Plik {event.src_path} nie istnieje lub jest pusty po chwili. Pomijanie.")
                return
            process_json_file(event.src_path)

if __name__ == "__main__":
    create_dirs_if_not_exist()
    # Jednorazowe ładowanie modeli przy starcie skryptu
    load_local_pycaret_model_once()
    load_location_vectorizer_once()

    if loaded_model_global is None or loaded_vectorizer_global is None:
        logging.critical("Nie udało się załadować modelu lub TfidfVectorizer. Skrypt nie będzie działać poprawnie. Sprawdź logi.")
        # Można tu dodać sys.exit(1), jeśli chcesz, aby skrypt się zakończył
    else:
        event_handler = JsonFileHandler()
        observer = Observer()
        observer.schedule(event_handler, JSON_IN_DIR, recursive=False)
        logging.info(f"Nasłuchiwanie na pliki .json w katalogu: {JSON_IN_DIR}")
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logging.info("Zatrzymano nasłuchiwanie na pliki JSON.")
        observer.join()