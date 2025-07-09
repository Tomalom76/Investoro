import time
import os
import json
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import sys
from pycaret.regression import load_model as pycaret_load_model
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
# === POPRAWIONA NAZWA ZMIENNEJ GLOBALNEJ ===
POTENTIALLY_TRANSFORMED_CATEGORICAL = [
    'Type', 'BuildingType', 'BuildingCondition', 'OwnerType', 
    'VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber',
    'OfferFrom', 'TypeOfMarket' # Te są mapowane ordinalnie, ale oryginalnie są kategoryczne
]
# =========================================

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
                 logging.info(f"Wczytano {len(FINAL_ESTIMATOR_FEATURES)} cech z {debug_final_file}")
            else: logging.warning(f"Plik {debug_final_file} jest pusty.")
        except Exception as e_read_debug: logging.error(f"Błąd odczytu {debug_final_file}: {e_read_debug}")

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
                        if FINAL_ESTIMATOR_FEATURES: # Zapisz tylko jeśli udało się uzyskać
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

def create_and_align_features(df_input: pd.DataFrame, vectorizer: joblib.memory.MemorizedResult, 
                              final_features_list: list) -> pd.DataFrame:
    df = df_input.copy()
    df_final_payload = pd.DataFrame(index=df.index) 
    logging.info(f"Dane do create_and_align_features: {df.shape}")

    required_for_nan = ['Area', 'Location']; df.dropna(subset=required_for_nan, inplace=True)
    if df.empty: logging.warning("DataFrame pusty po dropna."); return pd.DataFrame()
    if "Area" in df.columns and not df["Area"].dropna().empty:
        Q1, Q3 = df["Area"].quantile(0.25), df["Area"].quantile(0.75); IQR = Q3 - Q1
        df = df[~((df["Area"] < (Q1 - 1.5 * IQR)) | (df["Area"] > (Q3 + 1.5 * IQR)))]
    if df.empty: logging.warning("DataFrame pusty po usunięciu outlierów Area."); return pd.DataFrame()

    direct_numeric_or_ordinal_in_final = []
    for col_ff in final_features_list:
        # === UŻYJ POPRAWIONEJ NAZWY ZMIENNEJ GLOBALNEJ ===
        is_ohe = any(col_ff.startswith(cat_col + "_") for cat_col in POTENTIALLY_TRANSFORMED_CATEGORICAL)
        # ==============================================
        is_tfidf = col_ff.startswith('loc_tfidf_')
        is_date_comp = col_ff.startswith('BuiltYear_') 
        if not is_ohe and not is_tfidf and not is_date_comp:
            direct_numeric_or_ordinal_in_final.append(col_ff)
    logging.debug(f"Kolumny traktowane jako direct numeric/ordinal: {direct_numeric_or_ordinal_in_final}")

    offer_from_map = {'Osoba prywatna': 0.0, 'Biuro nieruchomości': 1.0, 'Deweloper': 2.0} 
    type_of_market_map = {'Wtórny': 0.0, 'Pierwotny': 1.0} 

    for col in direct_numeric_or_ordinal_in_final:
        if col in df.columns:
            if col == 'OfferFrom':
                df[col] = df[col].map(offer_from_map).astype(float).fillna(-1.0)
            elif col == 'TypeOfMarket':
                df[col] = df[col].map(type_of_market_map).astype(float).fillna(-1.0)
            else: 
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col].dtype):
                    df[col] = pd.to_numeric(df[col], errors='coerce') # to_numeric samo obsłuży NaN
                df[col] = df[col].astype(float) 
                if df[col].isnull().any():
                    logging.warning(f"Kolumna '{col}' zawiera NaN po konwersji na float. Imputuję -1.")
                    df[col].fillna(-1.0, inplace=True)
            df_final_payload[col] = df[col]
        elif col in final_features_list:
             logging.warning(f"Oczekiwana kolumna '{col}' nie istnieje w df. Wypełniam -1.0.")
             df_final_payload[col] = -1.0

    date_components = ['BuiltYear_year', 'BuiltYear_month', 'BuiltYear_day']
    if any(comp in final_features_list for comp in date_components):
        if 'BuiltYear' in df.columns:
            builtyear_dt = pd.to_datetime(pd.to_numeric(df['BuiltYear'], errors='coerce'), format='%Y', errors='coerce')
            year_series = builtyear_dt.dt.year.astype(float)
            month_series = builtyear_dt.dt.month.astype(float)
            day_series = builtyear_dt.dt.day.astype(float)
            if 'BuiltYear_year' in final_features_list: 
                df_final_payload['BuiltYear_year'] = year_series.fillna(year_series.median())
            if 'BuiltYear_month' in final_features_list:
                df_final_payload['BuiltYear_month'] = month_series.fillna(month_series.median())
            if 'BuiltYear_day' in final_features_list:
                df_final_payload['BuiltYear_day'] = day_series.fillna(day_series.median())
            logging.info("Utworzono komponenty daty (float).")
        else: 
            for comp in date_components:
                if comp in final_features_list: df_final_payload[comp] = np.nan

    tfidf_cols_from_final_list = [col for col in final_features_list if col.startswith('loc_tfidf_')]
    if tfidf_cols_from_final_list:
        if vectorizer and 'Location' in df.columns :
            tfidf_features = vectorizer.transform(df['Location'].fillna('').astype(str))
            tfidf_names_raw = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else vectorizer.get_feature_names_()
            tfidf_df_generated = pd.DataFrame(tfidf_features.toarray(), columns=['loc_tfidf_' + n for n in tfidf_names_raw], index=df.index)
            for col_tfidf in tfidf_cols_from_final_list:
                if col_tfidf in tfidf_df_generated.columns:
                    df_final_payload[col_tfidf] = tfidf_df_generated[col_tfidf]
                else: df_final_payload[col_tfidf] = 0.0 
            logging.info(f"Dodano cechy TF-IDF.")
        elif vectorizer: 
            logging.warning("Brak Location, dodaję puste (0.0) kolumny TF-IDF oczekiwane przez model.")
            for col_tfidf in tfidf_cols_from_final_list:
                df_final_payload[col_tfidf] = 0.0

    # === UŻYJ POPRAWIONEJ NAZWY ZMIENNEJ GLOBALNEJ ===
    for original_cat_col in POTENTIALLY_TRANSFORMED_CATEGORICAL:
    # ==============================================
        if original_cat_col not in final_features_list: 
            if original_cat_col in df.columns:
                source_series_for_ohe = df[original_cat_col].astype(str).fillna("nan_ohe_placeholder")
                ohe_variants_for_this_col = [f for f in final_features_list if f.startswith(original_cat_col + "_")]
                for ohe_col_name in ohe_variants_for_this_col:
                    category_value_in_ohe = ohe_col_name[len(original_cat_col)+1:]
                    if category_value_in_ohe == 'nan': 
                        df_final_payload[ohe_col_name] = (source_series_for_ohe == "nan_ohe_placeholder").astype(np.int32)
                    else:
                        df_final_payload[ohe_col_name] = (source_series_for_ohe == category_value_in_ohe).astype(np.int32)
            else: 
                ohe_variants_for_this_col = [f for f in final_features_list if f.startswith(original_cat_col + "_")]
                for ohe_col_name in ohe_variants_for_this_col:
                    df_final_payload[ohe_col_name] = 0 
    logging.info("Wykonano One-Hot Encoding.")

    for col_ff in final_features_list:
        if col_ff not in df_final_payload.columns:
            logging.warning(f"Ostatecznie dodaję brakującą kolumnę '{col_ff}' jako 0/NaN.")
            # === UŻYJ POPRAWIONEJ NAZWY ZMIENNEJ GLOBALNEJ ===
            if col_ff.startswith('loc_tfidf_') or any(col_ff.startswith(cat_col + "_") for cat_col in POTENTIALLY_TRANSFORMED_CATEGORICAL):
            # ==============================================
                df_final_payload[col_ff] = 0.0
            else: 
                df_final_payload[col_ff] = np.nan
    
    try:
        df_aligned = df_final_payload[final_features_list]
    except KeyError as e_final_align:
        missing = set(final_features_list) - set(df_final_payload.columns)
        logging.error(f"OSTATECZNY KEYERROR PRZY SELEKCJI KOLUMN! Brakuje: {missing}. Błąd: {e_final_align}")
        return pd.DataFrame()

    logging.info(f"DataFrame GOTOWY dla finalnego estymatora. Kształt: {df_aligned.shape}")
    return df_aligned


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
                try:
                    if is_ext_dtype_init: df_initial[col_name] = pd.Series([val_to_fill] * len(df_initial), dtype=expected_pd_type)
                    elif expected_pd_type == float: df_initial[col_name] = pd.Series([np.nan] * len(df_initial), dtype=float)
                    else: df_initial[col_name] = pd.Series([val_to_fill] * len(df_initial), dtype=object)
                except Exception as e_astype_empty: df_initial[col_name] = pd.Series([val_to_fill] * len(df_initial))
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
        
    data_payload_for_estimator = create_and_align_features(df_initial.copy(), loaded_location_vectorizer, FINAL_ESTIMATOR_FEATURES)
    
    if data_payload_for_estimator.empty:
        logging.error("Dane do predykcji są puste po wszystkich transformacjach."); return

    if 'Price' in data_payload_for_estimator.columns: # Upewnij się, że Price nie idzie do predict
        logging.info("Usuwam 'Price' z ostatecznego payloadu dla predict().")
        data_payload_for_estimator = data_payload_for_estimator.drop(columns=['Price'], errors='ignore')

    # Upewnij się, że payload ma tylko kolumny z FINAL_ESTIMATOR_FEATURES (bez Price)
    final_cols_for_predict_strict = [col for col in FINAL_ESTIMATOR_FEATURES if col != 'Price']
    
    # Sprawdzenie przed reindex
    missing_strict = set(final_cols_for_predict_strict) - set(data_payload_for_estimator.columns)
    if missing_strict:
        logging.error(f"KRYTYCZNE: Brakuje {missing_strict} w payloadzie PRZED finalnym reindex!")
        for m_s in missing_strict: # Dodaj jako NaN/0
             # === UŻYJ POPRAWIONEJ NAZWY ZMIENNEJ GLOBALNEJ ===
            if m_s.startswith('loc_tfidf_') or any(m_s.startswith(cat_col + "_") for cat_col in POTENTIALLY_TRANSFORMED_CATEGORICAL):
            # ==============================================
                data_payload_for_estimator[m_s] = 0.0
            else: data_payload_for_estimator[m_s] = np.nan

    try:
        data_payload_for_estimator = data_payload_for_estimator[final_cols_for_predict_strict]
    except KeyError as e_payload_final:
        logging.error(f"KeyError przy ustawianiu ostatecznych kolumn dla payloadu: {e_payload_final}")
        # ... obsługa błędu ...
        return

    logging.info(f"OSTATECZNE Kolumny DO `final_estimator.predict()` ({data_payload_for_estimator.shape[1]}): {data_payload_for_estimator.columns.tolist()}")
    key_cols_final_log = [col for col in ['BuiltYear_year', 'Area', 'Type_Mieszkania', 'OfferFrom', 'TypeOfMarket', 'OwnerType_Pełna własność', 'CountyNumber'] if col in data_payload_for_estimator.columns]
    if key_cols_final_log: logging.info(f"OSTATECZNE Typy kluczowych kolumn DO `final_estimator.predict()`:\n{data_payload_for_estimator[key_cols_final_log].dtypes.to_string()}")
    else: logging.info("Brak kluczowych kolumn do wylogowania typów.")

    final_json_to_save = {}
    try:
        logging.info(f"Próba wywołania .predict() na: {type(loaded_final_estimator)}")
        
        logging.info("--- TYPY DANYCH W data_payload_for_estimator PRZED predict() ---")
        all_numeric_and_correct_shape = True
        if data_payload_for_estimator.shape[1] != len(final_cols_for_predict_strict) :
            logging.error(f"Niezgodność liczby kolumn! Oczekiwano: {len(final_cols_for_predict_strict)}, jest: {data_payload_for_estimator.shape[1]}")
            all_numeric_and_correct_shape = False

        for col in data_payload_for_estimator.columns:
            col_dtype = data_payload_for_estimator[col].dtype
            if not pd.api.types.is_numeric_dtype(col_dtype):
                logging.error(f"KOLUMNA '{col}' NIE JEST NUMERYCZNA PRZED PREDICT! Typ: {col_dtype}. Wartości: {data_payload_for_estimator[col].unique()[:5]}")
                all_numeric_and_correct_shape = False
        
        if not all_numeric_and_correct_shape:
            raise TypeError("Nie wszystkie kolumny w data_payload_for_estimator są numeryczne lub zła liczba kolumn!")
        
        predicted_values = loaded_final_estimator.predict(data_payload_for_estimator)
        predictions_result_df = pd.DataFrame({'prediction_label': predicted_values}, index=data_payload_for_estimator.index)
        logging.info("Predykcja bezpośrednio na finalnym estymatorze zakończona.")
        
        # Używamy df_initial.loc[data_payload_for_estimator.index] jako bazy do łączenia,
        # ponieważ data_payload_for_estimator ma indeksy odpowiadające wierszom, które przeszły przez preprocessing.
        df_output_base = df_initial.loc[data_payload_for_estimator.index].copy() 
        
        if len(predictions_result_df) == len(df_output_base):
            df_output_base['prediction_label'] = predictions_result_df['prediction_label'].values
        else:
            logging.error(f"Niezgodność liczby wierszy przy łączeniu! Pred: {len(predictions_result_df)}, Baza: {len(df_output_base)}")
            df_output_base['prediction_label'] = "Pred Length Mismatch"

        output_records_final = [] 
        for original_idx, original_json_record in enumerate(data_records_list):
            final_record_to_output = original_json_record.copy(); predicted_price_formatted = "N/A (rekord mógł być odrzucony)"
            current_record_sale_id_key_norm = normalize_json_column_name('SaleId', list(original_json_record.keys()))
            current_record_sale_id_val = original_json_record.get(current_record_sale_id_key_norm)

            if current_record_sale_id_val is not None and 'SaleId' in df_output_base.columns:
                matched_rows = df_output_base[df_output_base['SaleId'] == current_record_sale_id_val]
                if not matched_rows.empty:
                    if 'prediction_label' in matched_rows.columns and pd.notna(matched_rows.iloc[0]['prediction_label']):
                        raw_pred = matched_rows.iloc[0]['prediction_label']
                        try: predicted_price_formatted = f"{float(raw_pred):,.0f}"
                        except (ValueError, TypeError): predicted_price_formatted = str(raw_pred) if raw_pred is not None else "Pred Error"
                    else: logging.warning(f"Brak predykcji dla SaleId: {current_record_sale_id_val} (pusta lub NaN).")
                else: logging.warning(f"SaleId: {current_record_sale_id_val} z JSON nie znaleziono w df_output_base.")
            else: 
                # Fallback na indeks, jeśli df_initial.index był prosty (0..N-1)
                # i df_output_base zachował te indeksy po .loc[]
                if original_idx in df_output_base.index and \
                   'prediction_label' in df_output_base.columns and \
                   pd.notna(df_output_base.loc[original_idx, 'prediction_label']):
                    raw_pred = df_output_base.loc[original_idx, 'prediction_label']
                    try: predicted_price_formatted = f"{float(raw_pred):,.0f}"
                    except (ValueError, TypeError): predicted_price_formatted = str(raw_pred) if raw_pred is not None else "Pred Error"
                    logging.info(f"Dopasowano predykcję dla rekordu (indeks JSON: {original_idx}, indeks DataFrame: {original_idx}) po indeksie.")
                else:
                    logging.warning(f"Nie można dopasować predykcji dla rekordu (indeks JSON: {original_idx}) ani po SaleId, ani po indeksie.")

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
        
        if output_records_final:
            if isinstance(output_records_final[0], dict) and 'PredictedPriceByModel' in output_records_final[0]:
                price_to_log = output_records_final[0]['PredictedPriceByModel']
                sid_to_log = output_records_final[0].get('SaleId', 'N/A')
                logging.info(f"--- KOŃCOWA PRZEWIDZIANA CENA (SaleId: {sid_to_log}): {price_to_log} ---")
        
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