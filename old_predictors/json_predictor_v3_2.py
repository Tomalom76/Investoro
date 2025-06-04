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

# --- Konfiguracja ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_IN_DIR = os.path.join(BASE_DIR, "1JSON")
JSON_OUT_DIR = os.path.join(BASE_DIR, "1JSON_OUT")

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
LOGGED_MODEL_URI_FROM_MLFLOW = 'runs:/daa9b673ba654fccbfdffb6f05a6ef75/model' 

MODEL_OUTPUT_DIR_FROM_WATCHER = os.path.join(BASE_DIR, "DATA_OUT") 
MODEL_FILENAME_FROM_WATCHER = "0_best_price_modelLGBM_auto" 

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

loaded_model_global = None

EXPECTED_INPUT_SCHEMA = {
    'SaleId': pd.Int64Dtype(), 'Area': float, 'NumberOfRooms': pd.Int64Dtype(), 
    'BuiltYear': pd.Int64Dtype(), 
    'BuildingType': pd.CategoricalDtype(categories=['Pozostałe', 'Blok', 'Apartametowiec', 'Kamienica'], ordered=True), 
    'BuildingCondition': pd.CategoricalDtype(categories=['For_Renovation', 'Good', 'After_Renovation', 'Developer_State'], ordered=True), 
    'Floor': pd.Int64Dtype(), 'Floors': pd.Int64Dtype(), 'Location': str, 'Description': str,
    'OriginalId': str, 'PortalId': pd.Int64Dtype(), 'Title': str, 'Type': str, 
    'OfferFrom': str, 'TypeOfMarket': str, 'OwnerType': str, 'DateAddedToDatabase': str, 
    'DateAdded': str, 'DateLastModification': str, 'DateLastRaises': str, 'NewestDate': str,
    'AvailableFrom': str, 'Link': str, 'Phone': object, 'MainImage': str, 'OtherImages': str,
    'NumberOfDuplicates': pd.Int64Dtype(), 'NumberOfRaises': pd.Int64Dtype(), 
    'NumberOfModifications': pd.Int64Dtype(), 'IsDuplicatePriceLower': pd.BooleanDtype(), 
    'IsDuplicatePrivateOwner': pd.BooleanDtype(), 'Score': pd.Int64Dtype(), 
    'ScorePrecision': pd.Int64Dtype(), 'CommunityScore': pd.Float64Dtype(),
    'NumberOfCommunityComments': pd.Int64Dtype(), 'NumberOfCommunityOpinions': pd.Int64Dtype(),
    'Archive': str, 'VoivodeshipNumber': pd.Float64Dtype(), 'CountyNumber': pd.Float64Dtype(),
    'CommunityNumber': pd.Float64Dtype(), 'KindNumber': pd.Float64Dtype(), 'RegionNumber': pd.Float64Dtype(),
    'SubRegionNumber': str, 'StreetNumber': pd.Float64Dtype(), 'EncryptedId': str,
    'OfferPrice': object, 'RealPriceAfterRenovation': object, 'OriginalPrice': object, 
    'PricePerSquareMeter': pd.Float64Dtype()
}
if 'Price' in EXPECTED_INPUT_SCHEMA: del EXPECTED_INPUT_SCHEMA['Price']

def normalize_column_name(col_name_from_json):
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
        "encryptedid": "EncryptedId", "predicted_state": "PredictedState",
        "column1": "Column1", "unnamed: 0": "Unnamed: 0"
    }
    if col_name in mapping: return mapping[col_name]
    if "_" in col_name_from_json: return "".join(word.capitalize() for word in col_name_from_json.split('_'))
    return col_name_from_json.capitalize()

def load_local_pycaret_model_once():
    global loaded_model_global
    if loaded_model_global is None:
        model_path = os.path.join(MODEL_OUTPUT_DIR_FROM_WATCHER, MODEL_FILENAME_FROM_WATCHER)
        if not os.path.exists(model_path + ".pkl"):
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

def create_dirs_if_not_exist():
    os.makedirs(JSON_IN_DIR, exist_ok=True)
    os.makedirs(JSON_OUT_DIR, exist_ok=True)
    logging.info(f"Katalogi 1JSON ('{JSON_IN_DIR}') i 1JSON_OUT ('{JSON_OUT_DIR}') gotowe.")

def process_json_file(json_file_path):
    model_pipeline = load_local_pycaret_model_once()
    if model_pipeline is None:
        logging.error("Model nie jest załadowany, pomijanie predykcji dla pliku: %s", json_file_path)
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            input_json_data_raw = json.load(f)
        logging.info(f"Pomyślnie wczytano plik JSON: {json_file_path}")
    except Exception as e:
        logging.error(f"Błąd podczas wczytywania lub parsowania pliku JSON {json_file_path}: {e}")
        return

    # Obsługa różnych struktur JSON (pojedynczy obiekt, lista, obiekt z listą pod kluczem)
    # oraz zachowanie oryginalnej struktury dla zapisu
    is_single_object_under_key = False
    is_single_top_level_object = False

    if isinstance(input_json_data_raw, dict) and len(input_json_data_raw) == 1:
        first_key = list(input_json_data_raw.keys())[0]
        if isinstance(input_json_data_raw[first_key], list):
            input_json_data_list = input_json_data_raw[first_key]
        elif isinstance(input_json_data_raw[first_key], dict):
             input_json_data_list = [input_json_data_raw[first_key]]
             is_single_object_under_key = True # Zapamiętaj, że był pojedynczy obiekt pod kluczem
        else: 
            # Jeśli klucz nie zawiera listy ani słownika, traktuj cały obiekt jako pojedynczy rekord
            input_json_data_list = [input_json_data_raw]
            is_single_top_level_object = True
    elif isinstance(input_json_data_raw, dict): # Pojedynczy obiekt JSON bez zagnieżdżenia
        input_json_data_list = [input_json_data_raw]
        is_single_top_level_object = True
    elif isinstance(input_json_data_raw, list): # Lista obiektów JSON
        input_json_data_list = input_json_data_raw
    else:
        logging.error(f"Nieobsługiwany główny format danych w pliku JSON: {json_file_path}.")
        return
    
    if not input_json_data_list:
        logging.warning(f"Plik JSON {json_file_path} nie zawiera danych do przetworzenia.")
        return

    try:
        input_df_raw = pd.DataFrame(input_json_data_list) # Tworzymy DF z listy rekordów
        input_df_raw.columns = [normalize_column_name(col) for col in input_df_raw.columns]
        logging.info(f"Nazwy kolumn po normalizacji: {input_df_raw.columns.tolist()}")
        
        df_for_model_input = pd.DataFrame()

        for col_expected, expected_dtype in EXPECTED_INPUT_SCHEMA.items():
            if col_expected in input_df_raw.columns:
                current_col_data = input_df_raw[col_expected]
                if isinstance(expected_dtype, (pd.Int64Dtype, pd.Float64Dtype, pd.CategoricalDtype, pd.BooleanDtype)) or expected_dtype in [int, float, bool]:
                    current_col_data = current_col_data.replace('', np.nan) 
                try:
                    if isinstance(expected_dtype, pd.CategoricalDtype):
                        df_for_model_input[col_expected] = pd.Categorical(current_col_data, categories=expected_dtype.categories, ordered=expected_dtype.ordered)
                    elif expected_dtype == str or expected_dtype == object:
                        df_for_model_input[col_expected] = current_col_data.astype(str).fillna("")
                    elif expected_dtype == int or isinstance(expected_dtype, pd.Int64Dtype):
                        df_for_model_input[col_expected] = pd.to_numeric(current_col_data, errors='coerce').astype(pd.Int64Dtype())
                    elif expected_dtype == float or isinstance(expected_dtype, pd.Float64Dtype):
                        df_for_model_input[col_expected] = pd.to_numeric(current_col_data, errors='coerce').astype(float)
                    elif expected_dtype == bool or isinstance(expected_dtype, pd.BooleanDtype):
                        bool_map = {'true': True, '1': True, 1: True, 'false': False, '0': False, 0: False, '': pd.NA, None: pd.NA, np.nan: pd.NA}
                        # Upewnij się, że konwertujesz na string przed .str.lower()
                        df_for_model_input[col_expected] = current_col_data.fillna('').astype(str).str.lower().map(bool_map).astype(pd.BooleanDtype())
                    else:
                        df_for_model_input[col_expected] = current_col_data.astype(expected_dtype)
                except Exception as e_conv:
                    logging.warning(f"Problem z konwersją kolumny '{col_expected}' do {expected_dtype}. Ustawiam na NaN/None. Błąd: {e_conv}")
                    df_for_model_input[col_expected] = pd.Series([None] * len(input_df_raw), dtype=object)
            else:
                logging.warning(f"Brakująca oczekiwana kolumna '{col_expected}' w JSON. Dodaję jako pustą/NaN.")
                if isinstance(expected_dtype, pd.CategoricalDtype): df_for_model_input[col_expected] = pd.Series([None] * len(input_df_raw), dtype=expected_dtype)
                elif isinstance(expected_dtype, (pd.Int64Dtype, pd.Float64Dtype, pd.BooleanDtype)): df_for_model_input[col_expected] = pd.Series([pd.NA] * len(input_df_raw), dtype=expected_dtype)
                else: df_for_model_input[col_expected] = pd.Series([""] * len(input_df_raw) if expected_dtype == str else [None] * len(input_df_raw), dtype=object)
        
        final_cols_ordered = list(EXPECTED_INPUT_SCHEMA.keys())
        for col_final in final_cols_ordered:
            if col_final not in df_for_model_input.columns:
                expected_dtype = EXPECTED_INPUT_SCHEMA[col_final]
                if isinstance(expected_dtype, pd.CategoricalDtype): df_for_model_input[col_final] = pd.Series([None] * len(df_for_model_input), dtype=expected_dtype)
                elif isinstance(expected_dtype, (pd.Int64Dtype, pd.Float64Dtype, pd.BooleanDtype)): df_for_model_input[col_final] = pd.Series([pd.NA] * len(df_for_model_input), dtype=expected_dtype)
                else: df_for_model_input[col_final] = pd.Series([""] * len(df_for_model_input) if expected_dtype == str else [None] * len(df_for_model_input), dtype=object)
        
        df_for_model_input = df_for_model_input.reindex(columns=final_cols_ordered)

        logging.info(f"Dane przekazywane do modelu (df_for_model_input) dla pliku {os.path.basename(json_file_path)}:\n{df_for_model_input.head().to_string(max_rows=5, max_cols=10)}")
        logging.info(f"Typy danych w df_for_model_input:\n{df_for_model_input.dtypes.to_string()}")
        
        predictions_df = pycaret_predict_model(model_pipeline, data=df_for_model_input)
        
        output_json_data_list_modified = []
        raw_predictions_for_log = [] 

        for i, original_record_dict in enumerate(input_json_data_list):
            new_record_output = original_record_dict.copy() 
            predicted_value_raw = predictions_df.loc[i, 'prediction_label'] if i < len(predictions_df) and 'prediction_label' in predictions_df.columns else None
            raw_predictions_for_log.append(predicted_value_raw)

            formatted_prediction = ""
            if predicted_value_raw is not None:
                try: formatted_prediction = f"{float(predicted_value_raw):,.0f}"
                except ValueError: formatted_prediction = str(predicted_value_raw)
            
            items = list(new_record_output.items())
            price_key_original_case = next((key for key in new_record_output if key.lower() == 'price'), None)
            if price_key_original_case:
                try:
                    idx = [item[0] for item in items].index(price_key_original_case)
                    items.insert(idx + 1, ('PredictedPriceByModel', formatted_prediction))
                    new_record_output = dict(items)
                except ValueError: 
                    new_record_output['PredictedPriceByModel'] = formatted_prediction
            else: 
                new_record_output['PredictedPriceByModel'] = formatted_prediction
            output_json_data_list_modified.append(new_record_output)
        
        if raw_predictions_for_log:
            log_msg_predictions = []
            for i, pred_val in enumerate(raw_predictions_for_log):
                # Używamy SaleId z oryginalnego rekordu, jeśli dostępne
                sale_id_for_log = input_json_data_list[i].get('saleid', input_json_data_list[i].get('SaleId', f"Rekord_{i+1}"))
                formatted_pred_for_log = output_json_data_list_modified[i].get('PredictedPriceByModel', 'Brak')
                log_msg_predictions.append(f"SaleId '{sale_id_for_log}': {formatted_pred_for_log} (surowa: {pred_val})")
            logging.info(f"Przewidziane ceny dla pliku {os.path.basename(json_file_path)}:\n" + "\n".join(log_msg_predictions))

        # Zapisz wynik zgodnie z oryginalną strukturą wejściową
        if is_single_object_under_key:
            first_key = list(input_json_data_raw.keys())[0]
            output_data_to_save = {first_key: output_json_data_list_modified[0]}
        elif is_single_top_level_object:
            output_data_to_save = output_json_data_list_modified[0]
        else: # Domyślnie lista
            output_data_to_save = output_json_data_list_modified

    except Exception as e:
        logging.error(f"Błąd podczas predykcji dla pliku {json_file_path}: {e}")
        import traceback; logging.error(traceback.format_exc())
        return

    # Tworzenie nowej nazwy pliku wyjściowego
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