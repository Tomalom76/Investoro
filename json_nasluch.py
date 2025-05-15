import time
import os
import json
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import sys
import mlflow

# --- Konfiguracja ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_IN_DIR = os.path.join(BASE_DIR, "1JSON")
JSON_OUT_DIR = os.path.join(BASE_DIR, "1JSON_OUT")

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
LOGGED_MODEL_URI_FROM_MLFLOW = 'runs:/daa9b673ba654fccbfdffb6f05a6ef75/model' 

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

loaded_model_global = None

# --- ZDEFINIUJ LISTĘ OCZEKIWANYCH KOLUMN WEJŚCIOWYCH I ICH TYPÓW (PASCALCASE) ---
# Te nazwy MUSZĄ odpowiadać nazwom kolumn, jakich oczekuje pipeline PyCaret (z setupu)
# Typy danych powinny odpowiadać typom w DataFrame przekazanym do setup() PyCaret.
EXPECTED_INPUT_SCHEMA = {
    # Kluczowe kolumny z setupu (dostosuj typy DOKŁADNIE do df_price2 przed setupem)
    'SaleId': pd.Int64Dtype(),                   # Z keep_features, w JSON 'saleid'
    'Area': float,                              # Numeryczna, również w keep_features, w JSON 'area'
    'BuildingType': pd.CategoricalDtype(categories=['Pozostałe', 'Blok', 'Apartametowiec', 'Kamienica'], ordered=True), # w JSON 'buildingtype'
    'BuildingCondition': pd.CategoricalDtype(categories=['For_Renovation', 'Good', 'After_Renovation', 'Developer_State'], ordered=True), # w JSON 'buildingcondition'
    'NumberOfRooms': pd.Int64Dtype(),           # w JSON 'numberofrooms'
    'BuiltYear': pd.Int64Dtype(),               # w JSON 'builtyear'
    'Floor': pd.Int64Dtype(),                   # w JSON 'floor'
    'Floors': pd.Int64Dtype(),                  # w JSON 'floors'
    'Location': str,                            # Z keep_features, w JSON 'location'
    
    # Pozostałe kolumny, które były w df_price2 i pipeline może ich oczekiwać (nawet jeśli są ignorowane)
    # Nazwy w PascalCase, typy jak w df_price2
    'OriginalId': str,                          # w JSON 'originalid'
    'PortalId': pd.Int64Dtype(),                # w JSON 'portalid'
    'Title': str,                               # w JSON 'title'
    'Description': str,                         # w JSON 'description'
    'OfferPrice': object,                       # Może być puste, string lub float; object jest bezpieczny
    'RealPriceAfterRenovation': object,
    'OriginalPrice': object,                    # W JSON jest null
    'PricePerSquareMeter': pd.Float64Dtype(),   # w JSON 'pricepersquaremeter'
    'Type': str,                                # w JSON 'Type' (tutaj wielka litera)
    'OfferFrom': str,                           # w JSON 'offerfrom'
    'TypeOfMarket': str,                        # w JSON 'typeofmarket'
    'OwnerType': str,                           # w JSON 'ownertype'
    'DateAddedToDatabase': str,                 # Można rozważyć konwersję na datetime, jeśli model tego używał
    'DateAdded': str,
    'DateLastModification': str,
    'DateLastRaises': str,
    'NewestDate': str,
    'AvailableFrom': str,
    'Link': str,
    'Phone': object,                            # W JSON jest null
    'MainImage': str,
    'OtherImages': str,
    'NumberOfDuplicates': pd.Int64Dtype(),
    'NumberOfRaises': pd.Int64Dtype(),
    'NumberOfModifications': pd.Int64Dtype(),
    'IsDuplicatePriceLower': bool, # lub pd.BooleanDtype() jeśli może być NaN
    'IsDuplicatePrivateOwner': bool,
    'Score': pd.Int64Dtype(),
    'ScorePrecision': pd.Int64Dtype(),
    'CommunityScore': pd.Float64Dtype(),
    'NumberOfCommunityComments': pd.Int64Dtype(),
    'NumberOfCommunityOpinions': pd.Int64Dtype(),
    'Archive': str,
    'VoivodeshipNumber': pd.Float64Dtype(),     # Były float, bo mogły mieć NaN
    'CountyNumber': pd.Float64Dtype(),
    'CommunityNumber': pd.Float64Dtype(),
    'KindNumber': pd.Float64Dtype(),
    'RegionNumber': pd.Float64Dtype(),
    'SubRegionNumber': object,                  # W JSON jest ""
    'StreetNumber': pd.Float64Dtype(),
    'EncryptedId': str,
    # 'predicted_state' - tej kolumny prawdopodobnie nie było w danych treningowych dla modelu cen
}
# Usuwamy 'Price' (target) z oczekiwanego schematu, jeśli przez pomyłkę został dodany
if 'Price' in EXPECTED_INPUT_SCHEMA:
    del EXPECTED_INPUT_SCHEMA['Price']


def normalize_column_name(col_name):
    """Konwertuje nazwę kolumny z JSON (małe litery) na PascalCase używany w schemacie."""
    if not isinstance(col_name, str):
        return str(col_name) # Na wszelki wypadek
    
    # Bezpośrednie mapowania dla przypadków, które nie pasują do prostego capitalize()
    # lub gdzie chcemy mieć pewność
    mapping = {
        "saleid": "SaleId",
        "originalid": "OriginalId",
        "portalid": "PortalId",
        "pricepersquaremeter": "PricePerSquareMeter",
        "numberofrooms": "NumberOfRooms",
        "builtyear": "BuiltYear",
        "buildingtype": "BuildingType",
        "buildingcondition": "BuildingCondition",
        "offerfrom": "OfferFrom",
        "typeofmarket": "TypeOfMarket",
        "ownertype": "OwnerType",
        "dateaddedtodatabase": "DateAddedToDatabase",
        "dateadded": "DateAdded",
        "datelastmodification": "DateLastModification",
        "datelastraises": "DateLastRaises",
        "newestdate": "NewestDate",
        "availablefrom": "AvailableFrom",
        "mainimage": "MainImage",
        "otherimages": "OtherImages",
        "numberofduplicates": "NumberOfDuplicates",
        "numberofraises": "NumberOfRaises",
        "numberofmodifications": "NumberOfModifications",
        "isduplicatepricelower": "IsDuplicatePriceLower",
        "isduplicateprivateowner": "IsDuplicatePrivateOwner",
        "scoreprecision": "ScorePrecision",
        "communityscore": "CommunityScore",
        "numberofcommunitycomments": "NumberOfCommunityComments",
        "numberofcommunityopinions": "NumberOfCommunityOpinions",
        "voivodeshipnumber": "VoivodeshipNumber",
        "countynumber": "CountyNumber",
        "communitynumber": "CommunityNumber",
        "kindnumber": "KindNumber",
        "regionnumber": "RegionNumber",
        "subregionnumber": "SubRegionNumber",
        "streetnumber": "StreetNumber",
        "encryptedid": "EncryptedId",
        "realpriceafterrenovation": "RealPriceAfterRenovation",
        "originalprice": "OriginalPrice",
        "offerprice": "OfferPrice",
        # Kolumny już w PascalCase lub z wielką literą na początku (zgodnie z obrazkiem)
        "Type": "Type", # Z obrazka
        "column1": "Column1", # Z obrazka (jeśli ma być używana)
        "Unnamed: 0": "Unnamed: 0" # Z obrazka (jeśli ma być używana)
    }
    if col_name.lower() in mapping: # Sprawdź wersję z małymi literami
        return mapping[col_name.lower()]
    
    # Ogólna próba PascalCase dla pozostałych (jeśli są np. snake_case)
    # lub po prostu capitalize jeśli to pojedyncze słowo
    if "_" in col_name:
        return "".join(word.capitalize() for word in col_name.split('_'))
    else:
        return col_name.capitalize() # Dla jednowyrazowych jak 'area', 'price', 'link', 'phone' itp.

def load_mlflow_model_once():
    global loaded_model_global
    if loaded_model_global is None:
        try:
            logging.info(f"Ładowanie modelu MLflow z URI: {LOGGED_MODEL_URI_FROM_MLFLOW}")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            loaded_model_global = mlflow.pyfunc.load_model(LOGGED_MODEL_URI_FROM_MLFLOW)
            logging.info("Model MLflow załadowany pomyślnie.")
        except Exception as e:
            logging.error(f"Nie udało się załadować modelu MLflow z {LOGGED_MODEL_URI_FROM_MLFLOW}: {e}")
            import traceback; logging.error(traceback.format_exc())
            loaded_model_global = None
    return loaded_model_global

def create_dirs_if_not_exist():
    os.makedirs(JSON_IN_DIR, exist_ok=True)
    os.makedirs(JSON_OUT_DIR, exist_ok=True)
    logging.info(f"Katalogi 1JSON ('{JSON_IN_DIR}') i 1JSON_OUT ('{JSON_OUT_DIR}') gotowe.")

def process_json_file(json_file_path):
    model = load_mlflow_model_once()
    if model is None:
        logging.error("Model nie jest załadowany, pomijanie predykcji dla pliku: %s", json_file_path)
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            input_json_data_raw = json.load(f)
        logging.info(f"Pomyślnie wczytano plik JSON: {json_file_path}")
    except Exception as e:
        logging.error(f"Błąd podczas wczytywania lub parsowania pliku JSON {json_file_path}: {e}")
        return

    # Obsługa zagnieżdżonego JSONa
    # Zakładamy, że dane są listą pod pierwszym kluczem słownika
    if isinstance(input_json_data_raw, dict) and len(input_json_data_raw) >= 1:
        first_key = list(input_json_data_raw.keys())[0]
        if isinstance(input_json_data_raw[first_key], list):
            input_json_data_list = input_json_data_raw[first_key]
        elif isinstance(input_json_data_raw[first_key], dict): # Jeśli pod kluczem jest pojedynczy obiekt
             input_json_data_list = [input_json_data_raw[first_key]]
        else: # Jeśli sam główny obiekt jest rekordem
            input_json_data_list = [input_json_data_raw]
    elif isinstance(input_json_data_raw, list):
        input_json_data_list = input_json_data_raw
    else:
        logging.error(f"Nieobsługiwany główny format danych w pliku JSON: {json_file_path}.")
        return
    
    if not input_json_data_list:
        logging.warning(f"Plik JSON {json_file_path} nie zawiera danych do przetworzenia.")
        return

    try:
        input_df_raw_from_list = pd.DataFrame(input_json_data_list)

        # Normalizacja nazw kolumn
        input_df_raw_from_list.columns = [normalize_column_name(col) for col in input_df_raw_from_list.columns]
        logging.info(f"Nazwy kolumn po normalizacji: {input_df_raw_from_list.columns.tolist()}")
        
        df_for_model_input = pd.DataFrame()

        for col_expected, expected_dtype in EXPECTED_INPUT_SCHEMA.items():
            if col_expected in input_df_raw_from_list.columns:
                current_col_data = input_df_raw_from_list[col_expected]
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
                        # Konwersja 0/1 na bool, puste stringi/NaN na False (lub pd.NA)
                        bool_map = {'true': True, '1': True, 1: True, 'false': False, '0': False, 0: False}
                        # Najpierw do string, potem mapowanie, potem na typ bool z obsługą NA
                        df_for_model_input[col_expected] = current_col_data.astype(str).str.lower().map(bool_map).astype(pd.BooleanDtype())
                    else:
                        df_for_model_input[col_expected] = current_col_data.astype(expected_dtype)
                except Exception as e_conv:
                    logging.warning(f"Problem z konwersją kolumny '{col_expected}' do {expected_dtype}. Ustawiam na NaN/None. Błąd: {e_conv}")
                    df_for_model_input[col_expected] = pd.Series([None] * len(input_df_raw_from_list), dtype=object)
            else:
                logging.warning(f"Brakująca oczekiwana kolumna '{col_expected}' w JSON. Dodaję jako pustą/NaN.")
                if isinstance(expected_dtype, pd.CategoricalDtype):
                    df_for_model_input[col_expected] = pd.Series([None] * len(input_df_raw_from_list), dtype=expected_dtype)
                elif isinstance(expected_dtype, (pd.Int64Dtype, pd.Float64Dtype, pd.BooleanDtype)):
                     df_for_model_input[col_expected] = pd.Series([pd.NA] * len(input_df_raw_from_list), dtype=expected_dtype)
                else: # str, object
                    df_for_model_input[col_expected] = pd.Series([""] * len(input_df_raw_from_list) if expected_dtype == str else [None] * len(input_df_raw_from_list), dtype=object)
        
        final_cols_ordered = list(EXPECTED_INPUT_SCHEMA.keys())
        df_for_model_input = df_for_model_input.reindex(columns=final_cols_ordered)

        logging.info(f"Dane przekazywane do modelu (df_for_model_input) dla pliku {os.path.basename(json_file_path)}:\n{df_for_model_input.head().to_string(max_rows=5, max_cols=10)}")
        logging.info(f"Typy danych w df_for_model_input:\n{df_for_model_input.dtypes.to_string()}")
        
        predictions_array = model.predict(df_for_model_input) 
        
        output_json_data_list = []
        for i, original_record_dict in enumerate(input_json_data_list):
            new_record = original_record_dict.copy() 
            predicted_value = predictions_array[i] if i < len(predictions_array) else None
            formatted_prediction = ""
            if predicted_value is not None:
                try: formatted_prediction = f"{float(predicted_value):,.0f}"
                except ValueError: formatted_prediction = str(predicted_value)
            
            items = list(new_record.items())
            price_key_original_case = next((key for key in new_record if key.lower() == 'price'), None)
            if price_key_original_case:
                try:
                    price_idx = [item[0] for item in items].index(price_key_original_case)
                    items.insert(price_idx + 1, ('PredictedPriceByModel', formatted_prediction))
                except ValueError: items.append(('PredictedPriceByModel', formatted_prediction))
            else: items.append(('PredictedPriceByModel', formatted_prediction))
            new_record_ordered = dict(items)
            output_json_data_list.append(new_record_ordered)
        
        # Zwróć strukturę zgodną z wejściem (pojedynczy obiekt lub lista)
        if isinstance(input_json_data_raw, dict) and len(input_json_data_raw) >= 1:
            first_key = list(input_json_data_raw.keys())[0]
            if isinstance(input_json_data_raw[first_key], dict): # Jeśli pod kluczem był pojedynczy obiekt
                output_data_to_save = {first_key: output_json_data_list[0]}
            else: # Jeśli pod kluczem była lista
                output_data_to_save = {first_key: output_json_data_list}
        else: # Jeśli na wejściu była lista lub pojedynczy obiekt bez klucza głównego
            output_data_to_save = output_json_data_list[0] if isinstance(input_json_data_raw, dict) else output_json_data_list


    except Exception as e:
        logging.error(f"Błąd podczas predykcji dla pliku {json_file_path}: {e}")
        import traceback; logging.error(traceback.format_exc())
        return

    base_filename = os.path.basename(json_file_path)
    output_json_path = os.path.join(JSON_OUT_DIR, base_filename)
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