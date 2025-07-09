import time
import os
import json
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import sys
from pycaret.regression import load_model as pycaret_load_model
# predict_model NIE BĘDZIE JUŻ UŻYWANE BEZPOŚREDNIO, jeśli ominiemy pipeline
# from pycaret.regression import predict_model as pycaret_predict_model
import numpy as np
import joblib

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

loaded_final_estimator = None
loaded_location_vectorizer = None
FINAL_ESTIMATOR_FEATURES = [] 

EXPECTED_COLUMNS_SCHEMA = { 
    'SaleId': pd.Int64Dtype(), 'OriginalId': object, 'PortalId': pd.Int64Dtype(),
    'Title': object, 'Description': object, 'Area': float, 'Price': float, 
    'OfferPrice': object, 'RealPriceAfterRenovation': object, 'OriginalPrice': object,
    'PricePerSquareMeter': float, 'NumberOfRooms': pd.Int64Dtype(),
    'BuiltYear': object, 'Type': object, 'BuildingType': object, 'BuildingCondition': object,
    'OfferFrom': object, 'Floor': pd.Int64Dtype(), 'Floors': pd.Int64Dtype(),
    'TypeOfMarket': object, 'OwnerType': object,
    'DateAddedToDatabase': object, 'DateAdded': object, 'DateLastModification': object,
    'DateLastRaises': object, 'NewestDate': object, 'AvailableFrom': object,
    'Link': object, 'Phone': object, 'MainImage': object, 'OtherImages': object,
    'NumberOfDuplicates': pd.Int64Dtype(), 'NumberOfRaises': pd.Int64Dtype(),
    'NumberOfModifications': pd.Int64Dtype(),
    'IsDuplicatePriceLower': pd.BooleanDtype(), 'IsDuplicatePrivateOwner': pd.BooleanDtype(),
    'Score': pd.Int64Dtype(), 'ScorePrecision': pd.Int64Dtype(), 'CommunityScore': float,
    'NumberOfCommunityComments': pd.Int64Dtype(), 'NumberOfCommunityOpinions': pd.Int64Dtype(),
    'Archive': object, 'Location': object, 'VoivodeshipNumber': object,
    'CountyNumber': object, 'CommunityNumber': object, 'KindNumber': object,
    'RegionNumber': object, 'SubRegionNumber': object, 'StreetNumber': object,
    'EncryptedId': object
}
ORIGINAL_COLS_THAT_GET_ONE_HOT_ENCODED = [
    'Type', 'BuildingType', 'BuildingCondition', 'OwnerType', 
    'VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber'
]

def normalize_json_column_name(col_name_from_json: str, schema_keys: list) -> str:
    col_name_lower = str(col_name_from_json).lower()
    direct_mapping = {"saleid": "SaleId", "builtyear": "BuiltYear", "price": "Price", "location": "Location", "area": "Area"}
    if col_name_lower in direct_mapping: return direct_mapping[col_name_lower]
    for sk in schema_keys:
        if col_name_lower == sk.lower(): return sk
    if col_name_from_json in schema_keys: return col_name_from_json
    return col_name_from_json

def load_final_estimator_and_vectorizer():
    global loaded_final_estimator, loaded_location_vectorizer, FINAL_ESTIMATOR_FEATURES
    success = True
    model_path_pkl = os.path.join(MODEL_ARTIFACTS_DIR, MODEL_FILENAME + ".pkl")
    vectorizer_path = os.path.join(MODEL_ARTIFACTS_DIR, VECTORIZER_FILENAME)

    debug_final_file = os.path.join(BASE_DIR, "DEBUG_final_estimator_features.txt")
    if os.path.exists(debug_final_file) and not FINAL_ESTIMATOR_FEATURES:
        try:
            with open(debug_final_file, "r") as f:
                FINAL_ESTIMATOR_FEATURES = [line.strip() for line in f if line.strip()]
            if FINAL_ESTIMATOR_FEATURES:
                 logging.info(f"Wczytano {len(FINAL_ESTIMATOR_FEATURES)} cech z DEBUG_final_estimator_features.txt")
            else: logging.warning("Plik DEBUG_final_estimator_features.txt jest pusty.")
        except Exception as e_read_debug: logging.error(f"Błąd odczytu DEBUG_final_estimator_features.txt: {e_read_debug}")

    if loaded_final_estimator is None:
        if not os.path.exists(model_path_pkl): logging.error(f"Model {model_path_pkl} nie znaleziony!"); success = False
        else:
            try:
                logging.info(f"Ładowanie pełnego pipeline'u PyCaret z: {model_path_pkl}...")
                temp_pipeline = pycaret_load_model(os.path.join(MODEL_ARTIFACTS_DIR, MODEL_FILENAME), verbose=False)
                if hasattr(temp_pipeline, 'steps'):
                    loaded_final_estimator = temp_pipeline.steps[-1][1]
                    logging.info(f"Wyekstrahowano finalny estymator: {type(loaded_final_estimator)}.")
                    if not FINAL_ESTIMATOR_FEATURES and hasattr(loaded_final_estimator, 'feature_names_in_'):
                        raw_features = loaded_final_estimator.feature_names_in_
                        FINAL_ESTIMATOR_FEATURES = list(raw_features.tolist() if hasattr(raw_features, 'tolist') else raw_features)
                        logging.info(f"Automatycznie uzyskano {len(FINAL_ESTIMATOR_FEATURES)} cech dla finalnego estymatora.")
                        try:
                            with open(debug_final_file, "w") as f:
                                for fn in FINAL_ESTIMATOR_FEATURES: f.write(f"{fn}\n")
                            logging.info(f"Zapisano/nadpisano cechy finalnego estymatora do {debug_final_file}")
                        except Exception as e_w: logging.error(f"Błąd zapisu {debug_final_file}: {e_w}")
                    elif not FINAL_ESTIMATOR_FEATURES:
                         logging.warning("Finalny estymator nie ma `feature_names_in_`, a plik DEBUG nie istniał lub był pusty.")
                else:
                    logging.error("Załadowany obiekt nie jest pipeline'em sklearn. Nie można wyekstrahować finalnego estymatora."); success = False
            except Exception as e: logging.error(f"Błąd ładowania pipeline'u/ekstrakcji estymatora: {e}", exc_info=True); success = False
    
    if loaded_location_vectorizer is None:
        if not os.path.exists(vectorizer_path): logging.error(f"Vectorizer {vectorizer_path} nie znaleziony!"); success = False
        else:
            try:
                logging.info(f"Ładowanie TfidfVectorizer z: {vectorizer_path}")
                loaded_location_vectorizer = joblib.load(vectorizer_path)
                logging.info("TfidfVectorizer załadowany.")
            except Exception as e: logging.error(f"Błąd ładowania vectorizera: {e}", exc_info=True); success = False
    
    if not FINAL_ESTIMATOR_FEATURES and success:
        logging.error("KRYTYCZNE: FINAL_ESTIMATOR_FEATURES jest pusta."); success = False
    if loaded_final_estimator is None and success:
        logging.error("KRYTYCZNE: Nie załadowano finalnego estymatora."); success = False
    return success

def create_features_for_final_estimator(df_input: pd.DataFrame, vectorizer: joblib.memory.MemorizedResult, 
                                       final_feature_list: list) -> pd.DataFrame:
    df = df_input.copy()
    logging.info(f"Dane do create_features_for_final_estimator: {df.shape}")
    
    required_for_nan = ['Area', 'Location']; df.dropna(subset=required_for_nan, inplace=True)
    if df.empty: logging.warning("DataFrame pusty po dropna."); return pd.DataFrame()
    if "Area" in df.columns and not df["Area"].dropna().empty:
        Q1, Q3 = df["Area"].quantile(0.25), df["Area"].quantile(0.75); IQR = Q3 - Q1
        df = df[~((df["Area"] < (Q1 - 1.5 * IQR)) | (df["Area"] > (Q3 + 1.5 * IQR)))]
    if df.empty: logging.warning("DataFrame pusty po usunięciu outlierów Area."); return pd.DataFrame()

    cols_to_process_types = {
        'Area': float, 'NumberOfRooms': pd.Int64Dtype(), 'Floor': pd.Int64Dtype(), 
        'Floors': pd.Int64Dtype(), 'CommunityScore': float, 'RegionNumber': float,
    }
    for col, dtype_info in cols_to_process_types.items():
        if col in df.columns:
            try:
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col].dtype):
                    df[col] = pd.to_numeric(df[col].replace(['', None, 'None', 'null', 'nan', 'NaN'], np.nan), errors='coerce')
                if df[col].notna().any():
                    if str(dtype_info).endswith("Dtype"): df[col] = df[col].astype(dtype_info)
                    elif dtype_info == float and df[col].dtype != float : df[col] = df[col].astype(float)
                elif str(dtype_info).endswith("Dtype"): df[col] = df[col].astype(dtype_info)
                elif dtype_info == float: df[col] = df[col].astype(float)
            except Exception as e_conv_num:
                logging.warning(f"Problem z konwersją numerycznej '{col}' na {dtype_info}: {e_conv_num}")

    offer_from_map = {'Osoba prywatna': 0.0, 'Biuro nieruchomości': 1.0, 'Deweloper': 2.0} 
    if 'OfferFrom' in df.columns:
        df['OfferFrom_original_str'] = df['OfferFrom'] 
        df['OfferFrom'] = df['OfferFrom'].map(offer_from_map).astype(float)
        if df['OfferFrom'].isnull().any() and df['OfferFrom_original_str'].notna().any() :
            unmapped = df.loc[df['OfferFrom'].isnull() & df['OfferFrom_original_str'].notna(), 'OfferFrom_original_str'].unique()
            logging.warning(f"Niektóre wartości w 'OfferFrom' ({unmapped}) nie zostały zmapowane i są NaN.")
        logging.info(f"Ręcznie zakodowano 'OfferFrom'. Przykładowe wartości: {df['OfferFrom'].unique()[:5]}")
        df = df.drop(columns=['OfferFrom_original_str'], errors='ignore')
    elif 'OfferFrom' in final_feature_list: 
        df['OfferFrom'] = np.nan

    type_of_market_map = {'Wtórny': 0.0, 'Pierwotny': 1.0} 
    if 'TypeOfMarket' in df.columns:
        df['TypeOfMarket_original_str'] = df['TypeOfMarket']
        df['TypeOfMarket'] = df['TypeOfMarket'].map(type_of_market_map).astype(float)
        if df['TypeOfMarket'].isnull().any() and df['TypeOfMarket_original_str'].notna().any():
            unmapped = df.loc[df['TypeOfMarket'].isnull() & df['TypeOfMarket_original_str'].notna(), 'TypeOfMarket_original_str'].unique()
            logging.warning(f"Niektóre wartości w 'TypeOfMarket' ({unmapped}) nie zostały zmapowane i są NaN.")
        logging.info(f"Ręcznie zakodowano 'TypeOfMarket'. Przykładowe wartości: {df['TypeOfMarket'].unique()[:5]}")
        df = df.drop(columns=['TypeOfMarket_original_str'], errors='ignore')
    elif 'TypeOfMarket' in final_feature_list:
        df['TypeOfMarket'] = np.nan
    
    if 'BuiltYear' in df.columns:
        df['BuiltYear_dt_temp'] = pd.to_numeric(df['BuiltYear'], errors='coerce')
        df['BuiltYear_dt_temp'] = pd.to_datetime(df['BuiltYear_dt_temp'], format='%Y', errors='coerce')
        df['BuiltYear_year'] = df['BuiltYear_dt_temp'].dt.year.astype(float) 
        df['BuiltYear_month'] = df['BuiltYear_dt_temp'].dt.month.astype(float)
        df['BuiltYear_day'] = df['BuiltYear_dt_temp'].dt.day.astype(float)
        df = df.drop(columns=['BuiltYear', 'BuiltYear_dt_temp'], errors='ignore')
        logging.info("Ręcznie utworzono BuiltYear_year/month/day (jako float) i usunięto oryginalne BuiltYear.")
    else:
        logging.warning("Brak 'BuiltYear'. Dodaję puste komponenty daty (float), jeśli oczekiwane.")
        for comp in ['BuiltYear_year', 'BuiltYear_month', 'BuiltYear_day']:
            if comp in final_feature_list and comp not in df.columns:
                 df[comp] = pd.Series([np.nan] * len(df), dtype=float)

    if 'Location' in df.columns and vectorizer:
        df['Location_Clean'] = df['Location'].fillna('').astype(str)
        tfidf_features = vectorizer.transform(df['Location_Clean'])
        try: tfidf_names = vectorizer.get_feature_names_out()
        except AttributeError: tfidf_names = vectorizer.get_feature_names_()
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=['loc_tfidf_' + n for n in tfidf_names], index=df.index)
        df = df.drop(columns=['Location', 'Location_Clean'], errors='ignore')
        df = pd.concat([df, tfidf_df], axis=1)
    elif vectorizer: 
        logging.warning("Brak Location. Dodaję puste kolumny TF-IDF, jeśli są w FINAL_ESTIMATOR_FEATURES.")
        all_tfidf_names = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else vectorizer.get_feature_names_()
        for name in all_tfidf_names:
            col_name_tfidf = 'loc_tfidf_' + name
            if col_name_tfidf in FINAL_ESTIMATOR_FEATURES and col_name_tfidf not in df.columns: df[col_name_tfidf] = 0.0
            
    logging.info(f"Po inżynierii cech (BuiltYear_*, TF-IDF, Ordinal): {df.shape}")
    return df

def align_to_final_estimator_features(df_engineered: pd.DataFrame, final_features_list: list) -> pd.DataFrame:
    df_aligned = pd.DataFrame(index=df_engineered.index)
    if not final_features_list:
        logging.error("`final_features_list` jest pusta!"); return pd.DataFrame()

    logging.info(f"Rozpoczynam align_to_final_estimator_features. Oczekiwane cechy: {len(final_features_list)}")
    logging.debug(f"Kolumny w df_engineered na wejściu do align: {df_engineered.columns.tolist()}")

    for final_col_name in final_features_list:
        is_ohe_target_col = any(final_col_name.startswith(cat_col + "_") for cat_col in ORIGINAL_COLS_THAT_GET_ONE_HOT_ENCODED)
        
        if not is_ohe_target_col: 
            if final_col_name in df_engineered.columns:
                df_aligned[final_col_name] = df_engineered[final_col_name]
            else: 
                logging.warning(f"Oczekiwana nie-OHE kolumna '{final_col_name}' nie istnieje w df_engineered. Wypełniam 0/np.nan.")
                if final_col_name.startswith('loc_tfidf_'): df_aligned[final_col_name] = 0.0
                elif final_col_name.startswith('BuiltYear_'): df_aligned[final_col_name] = np.nan
                else: df_aligned[final_col_name] = np.nan 
            continue

        was_one_hot_created_in_align = False
        for original_cat_col in ORIGINAL_COLS_THAT_GET_ONE_HOT_ENCODED:
            if final_col_name.startswith(original_cat_col + "_"):
                category_value_in_ohe = final_col_name[len(original_cat_col)+1:]
                
                if original_cat_col in df_engineered.columns:
                    source_series = df_engineered[original_cat_col]
                    if category_value_in_ohe == 'nan':
                        df_aligned[final_col_name] = source_series.isnull().astype(np.int32)
                    else:
                        df_aligned[final_col_name] = (source_series.astype(str) == category_value_in_ohe).astype(np.int32)
                else: 
                    df_aligned[final_col_name] = 0 
                was_one_hot_created_in_align = True
                break 
        
        if not was_one_hot_created_in_align and final_col_name not in df_aligned.columns:
             logging.warning(f"Nierozpoznana oczekiwana kolumna '{final_col_name}'. Wypełniam 0.")
             df_aligned[final_col_name] = 0
    
    missing_cols_after_align = set(final_features_list) - set(df_aligned.columns)
    if missing_cols_after_align:
        logging.error(f"Po wyrównaniu nadal brakuje kolumn: {missing_cols_after_align}! Dodaję je jako 0/np.nan.")
        for mc in missing_cols_after_align:
            if mc.startswith('loc_tfidf_') or any(mc.startswith(cat_col + "_") for cat_col in ORIGINAL_COLS_THAT_GET_ONE_HOT_ENCODED):
                df_aligned[mc] = 0.0
            else: df_aligned[mc] = np.nan
            
    try:
        df_final_output = df_aligned[final_features_list]
    except KeyError as e_final_align_key:
        m_keys = set(final_features_list) - set(df_aligned.columns)
        logging.error(f"KeyError przy ostatecznym df_aligned[final_features_list]! Brakuje: {m_keys}. Błąd: {e_final_align_key}")
        return pd.DataFrame()

    logging.info(f"DataFrame wyrównany do cech finalnego estymatora. Kształt: {df_final_output.shape}")
    return df_final_output

def process_single_json_file(json_file_path: str):
    if not (loaded_final_estimator and loaded_location_vectorizer and FINAL_ESTIMATOR_FEATURES):
        logging.error("Krytyczne: Estymator, Vectorizer lub FINAL_ESTIMATOR_FEATURES niezaładowane."); return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f: raw_json_content = json.load(f)
    except Exception as e: logging.error(f"Błąd wczytywania JSON {json_file_path}: {e}", exc_info=True); return
    
    data_records_list = [] 
    original_json_structure_key = None; is_single_item_under_key = False; is_single_top_level_item = False
    if isinstance(raw_json_content, dict) and len(raw_json_content) == 1:
        original_json_structure_key = list(raw_json_content.keys())[0]
        content_under_key = raw_json_content[original_json_structure_key]
        if isinstance(content_under_key, list): data_records_list = content_under_key
        elif isinstance(content_under_key, dict): data_records_list = [content_under_key]; is_single_item_under_key = True
        else: logging.error(f"Nieoczekiwany typ ({type(content_under_key)}) pod '{original_json_structure_key}'."); return
    elif isinstance(raw_json_content, dict): data_records_list = [raw_json_content]; is_single_top_level_item = True
    elif isinstance(raw_json_content, list): data_records_list = raw_json_content
    else: logging.error(f"Nieobsługiwany główny typ danych w JSON: {type(raw_json_content)}."); return
    if not data_records_list: logging.warning(f"JSON {json_file_path} nie zawiera rekordów."); return

    df_initial = None
    try: 
        schema_keys = list(EXPECTED_COLUMNS_SCHEMA.keys())
        normalized_records = [{normalize_json_column_name(k, schema_keys): v for k, v in record.items()} for record in data_records_list]
        df_initial = pd.DataFrame(normalized_records)
        logging.info(f"DataFrame z JSON: {df_initial.shape}.")
        for col_name, expected_pd_type in EXPECTED_COLUMNS_SCHEMA.items():
            if col_name not in df_initial.columns:
                is_ext_dtype_init = str(expected_pd_type).endswith("Dtype")
                val_to_fill = pd.NA if is_ext_dtype_init or expected_pd_type == float else None
                if is_ext_dtype_init: df_initial[col_name] = pd.Series([val_to_fill] * len(df_initial), dtype=expected_pd_type)
                elif expected_pd_type == float: df_initial[col_name] = pd.Series([np.nan] * len(df_initial), dtype=float)
                else: df_initial[col_name] = pd.Series([val_to_fill] * len(df_initial), dtype=object)
            else:
                current_dtype = df_initial[col_name].dtype; is_expected_ext_type = not isinstance(expected_pd_type, type)
                if pd.api.types.is_string_dtype(current_dtype) or current_dtype == 'object':
                    try:
                        if expected_pd_type == float: df_initial[col_name] = pd.to_numeric(df_initial[col_name].replace(['', None, 'None', 'null', 'nan', 'NaN'], np.nan), errors='coerce')
                        elif is_expected_ext_type and str(expected_pd_type) == 'Int64': df_initial[col_name] = pd.to_numeric(df_initial[col_name].replace(['', None, 'None', 'null', 'nan', 'NaN'], np.nan), errors='coerce').astype(pd.Int64Dtype())
                        elif is_expected_ext_type and str(expected_pd_type) == 'boolean':
                            bool_map = {'true': True, '1': True, 'false': False, '0': False, '': pd.NA, None: pd.NA}
                            df_initial[col_name] = df_initial[col_name].fillna(pd.NA).astype(str).str.lower().map(bool_map).astype(pd.BooleanDtype())
                    except Exception as e_conv_init: logging.warning(f"Wstępna konwersja dla '{col_name}' nieudana: {e_conv_init}")
    except Exception as e: logging.error(f"Błąd tworzenia DataFrame: {e}", exc_info=True); return
    if df_initial is None or df_initial.empty: logging.error("df_initial pusty."); return
        
    df_features_engineered = create_features_for_final_estimator(df_initial.copy(), loaded_location_vectorizer, FINAL_ESTIMATOR_FEATURES)

    if df_features_engineered.empty:
        logging.warning("Dane puste po `create_features_for_final_estimator`."); return

    data_to_predict_with = align_to_final_estimator_features(df_features_engineered, FINAL_ESTIMATOR_FEATURES) 
    
    if data_to_predict_with.empty:
        logging.error("Dane puste po wyrównaniu do cech finalnego estymatora."); return

    if 'Price' in data_to_predict_with.columns:
        logging.info("Usuwam 'Price' z danych przekazywanych do final_estimator.predict().")
        data_to_predict_with = data_to_predict_with.drop(columns=['Price'], errors='ignore')

    logging.info(f"OSTATECZNE Kolumny DO `final_estimator.predict()` ({data_to_predict_with.shape[1]}): {data_to_predict_with.columns.tolist()}")
    # Poprawione logowanie kluczowych kolumn
    key_cols_final_log = [col for col in ['BuiltYear_year', 'Area', 'Type_Mieszkania', 'OfferFrom', 'TypeOfMarket'] if col in data_to_predict_with.columns]
    if key_cols_final_log: logging.info(f"OSTATECZNE Typy kluczowych kolumn DO `final_estimator.predict()`:\n{data_to_predict_with[key_cols_final_log].dtypes.to_string()}")
    else: logging.info("Brak kluczowych kolumn do wylogowania typów.")

    final_json_to_save = {}
    try:
        logging.info(f"Próba wywołania .predict() na: {type(loaded_final_estimator)}")
        cols_for_prediction_final = [col for col in FINAL_ESTIMATOR_FEATURES if col != 'Price']
        
        missing_before_predict = set(cols_for_prediction_final) - set(data_to_predict_with.columns)
        if missing_before_predict:
            logging.error(f"KRYTYCZNE: Brakuje kolumn {missing_before_predict} w data_to_predict_with tuż przed .predict()!")
            for mc_bp in missing_before_predict:
                if mc_bp.startswith('loc_tfidf_') or any(mc_bp.startswith(cat_col + "_") for cat_col in ORIGINAL_COLS_THAT_GET_ONE_HOT_ENCODED): data_to_predict_with[mc_bp] = 0.0
                else: data_to_predict_with[mc_bp] = np.nan
        
        data_payload_for_estimator = data_to_predict_with[cols_for_prediction_final]

        predicted_values = loaded_final_estimator.predict(data_payload_for_estimator)
        predictions_result_df = pd.DataFrame({'prediction_label': predicted_values}, index=data_payload_for_estimator.index)
        logging.info("Predykcja bezpośrednio na finalnym estymatorze zakończona.")
        
        df_output_base = df_features_engineered.copy()
        if len(predictions_result_df) == len(df_output_base):
            predictions_result_df.index = df_output_base.index
            df_output_base['prediction_label'] = predictions_result_df['prediction_label']
        else: 
            logging.error(f"Niezgodność liczby wierszy! Pred: {len(predictions_result_df)}, Baza: {len(df_output_base)}. Próba resetu indeksów.")
            if len(predictions_result_df) == len(df_output_base.reset_index(drop=True)):
                 df_output_base = df_output_base.reset_index(drop=True)
                 predictions_result_df = predictions_result_df.reset_index(drop=True)
                 df_output_base['prediction_label'] = predictions_result_df['prediction_label']
                 logging.info("Przypisano predykcje po zresetowaniu indeksów.")
            else: df_output_base['prediction_label'] = "Pred Length Mismatch"

        output_records_final = [] # ... (Reszta logiki łączenia i zapisu JSON - jak wcześniej) ...
        for original_idx, original_json_record in enumerate(data_records_list):
            final_record_to_output = original_json_record.copy(); predicted_price_formatted = "N/A (no match or not processed)"
            current_record_sale_id_key_norm = normalize_json_column_name('SaleId', list(original_json_record.keys()))
            current_record_sale_id_val = original_json_record.get(current_record_sale_id_key_norm)
            if current_record_sale_id_val is not None and 'SaleId' in df_output_base.columns:
                match_prediction_series = df_output_base[df_output_base['SaleId'] == current_record_sale_id_val]
                if not match_prediction_series.empty:
                    raw_pred = match_prediction_series.iloc[0]['prediction_label']
                    try: predicted_price_formatted = f"{float(raw_pred):,.0f}"
                    except (ValueError, TypeError): predicted_price_formatted = str(raw_pred) if raw_pred is not None else "Pred Error"
                else: logging.warning(f"SaleId: {current_record_sale_id_val} z JSON nie znaleziono w przetworzonych danych.")
            else: logging.warning(f"Brak SaleId w rekordzie {original_idx} (JSON) lub w przetworzonych danych.")
            price_key_in_output = None
            for k_output in final_record_to_output.keys():
                if k_output.lower() == 'price': price_key_in_output = k_output; break
            items_list = list(final_record_to_output.items())
            if price_key_in_output:
                idx_price_output = -1
                for i, (k,v) in enumerate(items_list):
                    if k.lower() == 'price': idx_price_output = i; break
                if idx_price_output != -1: items_list.insert(idx_price_output + 1, ('PredictedPriceByModel', predicted_price_formatted))
                else: items_list.append(('PredictedPriceByModel', predicted_price_formatted))
            else: items_list.append(('PredictedPriceByModel', predicted_price_formatted))
            output_records_final.append(dict(items_list))
        
        # === DODANE LOGOWANIE PRZED ZAPISEM DO PLIKU ===
        if output_records_final:
            # Logika wyciągania ceny zależy od struktury output_records_final
            # Zakładamy, że output_records_final[0] to słownik z predykcją dla pierwszego rekordu
            if isinstance(output_records_final[0], dict) and 'PredictedPriceByModel' in output_records_final[0]:
                price_to_log = output_records_final[0]['PredictedPriceByModel']
                sid_to_log = output_records_final[0].get('SaleId', 'N/A')
                logging.info(f"--- KOŃCOWA PRZEWIDZIANA CENA (SaleId: {sid_to_log}): {price_to_log} ---")
            else:
                logging.info("--- Przewidziana cena (nie znaleziono klucza 'PredictedPriceByModel' w pierwszym rekordzie) ---")
        # ===============================================

        if is_single_item_under_key and original_json_structure_key: final_json_to_save = {original_json_structure_key: output_records_final[0]}
        elif is_single_top_level_item: final_json_to_save = output_records_final[0]
        elif original_json_structure_key: final_json_to_save = {original_json_structure_key: output_records_final}
        else: final_json_to_save = output_records_final
    except Exception as e:
        logging.error(f"Błąd predykcji/łączenia dla {json_file_path}: {e}", exc_info=True)
        error_message = f"Prediction/joining error: {str(e)}"
        error_payload = {"error": error_message}
        if is_single_item_under_key and original_json_structure_key: final_json_to_save = {original_json_structure_key: error_payload}
        elif is_single_top_level_item: final_json_to_save = error_payload
        elif original_json_structure_key: final_json_to_save = {original_json_structure_key: [error_payload]}
        else: final_json_to_save = [error_payload]

    base_name, ext_name = os.path.splitext(os.path.basename(json_file_path))
    output_file_name = f"{base_name}_out{ext_name}"
    output_file_path = os.path.join(JSON_OUT_DIR, output_file_name)
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(final_json_to_save, f_out, ensure_ascii=False, indent=4)
        logging.info(f"Wynik zapisano do: {output_file_path}")
    except Exception as e:
        logging.error(f"Błąd zapisu wyniku JSON {output_file_path}: {e}", exc_info=True)

# --- Watchdog Handler i __main__ ---
class JsonFileChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            logging.info(f"Wykryto nowy plik JSON: {event.src_path}")
            time.sleep(1.5) 
            if not os.path.exists(event.src_path) or os.path.getsize(event.src_path) == 0:
                logging.warning(f"Plik {event.src_path} nie istnieje/pusty. Pomijam.")
                return
            process_single_json_file(event.src_path)

def create_dirs_if_not_exist():
    os.makedirs(JSON_IN_DIR, exist_ok=True); logging.info(f"Katalog IN: {os.path.abspath(JSON_IN_DIR)}")
    os.makedirs(JSON_OUT_DIR, exist_ok=True); logging.info(f"Katalog OUT: {os.path.abspath(JSON_OUT_DIR)}")

if __name__ == "__main__":
    logging.info("Inicjalizacja skryptu predykcyjnego.")
    create_dirs_if_not_exist()
    if not load_final_estimator_and_vectorizer(): 
        logging.critical("Nie załadowano artefaktów lub FINAL_ESTIMATOR_FEATURES. Koniec."); sys.exit(1)
    
    event_handler = JsonFileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, JSON_IN_DIR, recursive=False)
    logging.info(f"Nasłuchiwanie na .json w: {os.path.abspath(JSON_IN_DIR)}")
    observer.start()
    try:
        while True: time.sleep(5)
    except KeyboardInterrupt: logging.info("Zatrzymano przez użytkownika.")
    except Exception as e_main: logging.error(f"Błąd w głównej pętli: {e_main}", exc_info=True)
    finally:
        observer.stop()
        observer.join()
        logging.info("Skrypt zakończył działanie.")