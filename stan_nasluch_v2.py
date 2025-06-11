import pandas as pd
from pycaret.classification import setup, create_model, finalize_model, get_config
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
import numpy as np
import os
import time
import shutil
import joblib
import logging
import sys
import json
import traceback

# --- Konfiguracja ---
DATA_INPUT_DIR_TRAINING = "DATA_STAN"
ARTIFACTS_OUTPUT_DIR = "DATA_STAN_OUT"
PROCESSED_TRAINING_DATA_DIR_NAME = "PROCESSED_DATA_STAN_TRAINING"
INPUT_CSV_FILENAME = "data.csv"

# --- Nazwy plików wyjściowych dla artefaktów ---
VECTORIZER_FILENAME_OUTPUT = "count_vectorizer.joblib"
ENCODER_FILENAME_OUTPUT = "one_hot_encoder.pkl"
IMPUTER_VALUES_FILENAME_OUTPUT = "imputer_values.json"
FINAL_MODEL_COLUMNS_FILENAME_OUTPUT = "final_model_columns.json"
CLEAN_MODEL_FILENAME_OUTPUT = "clean_lgbm_model.pkl"

# Konfiguracja loggingu
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def preprocess_data_for_training(df_input_original):
    """
    Pełny proces preprocessingu danych, odtwarzający logikę z notatnika.
    Zwraca finalny DataFrame gotowy do `setup()` oraz wytrenowany `vectorizer`.
    """
    logging.info("Rozpoczynam preprocessing danych wejściowych dla treningu...")
    df = df_input_original.copy()
    
    # Konwersja daty
    df['BuiltYear'] = pd.to_datetime(df['BuiltYear'], format='%Y', errors='coerce')
    
    # Usuwanie wierszy z brakami w kluczowych kolumnach
    initial_rows = len(df)
    df = df.dropna(subset=['Description', 'Location', 'BuildingCondition'])
    logging.info(f"Po dropna na kluczowych kolumnach: {len(df)} wierszy (usunięto {initial_rows - len(df)}).")
    if df.empty:
        logging.error("Brak danych po podstawowym czyszczeniu (Description, Location, BuildingCondition).")
        return None, None
    
    # Przetwarzanie tekstu
    df['Description'] = df['Description'].str.slice(0, 3000)
    vectorizer = CountVectorizer(max_features=1500)
    X_bow = vectorizer.fit_transform(df["Description"])
    df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index)
    df_prepared = pd.concat([df.drop(columns=['Description']), df_bow], axis=1)

    # Usuwanie outlierów
    for col in ['Price', 'PricePerSquareMeter', 'Area']:
        if col in df_prepared.columns and not df_prepared[col].isnull().all() and len(df_prepared) > 1:
            Q1 = df_prepared[col].quantile(0.25); Q3 = df_prepared[col].quantile(0.75); IQR = Q3 - Q1
            lower_b, upper_b = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_prepared = df_prepared[df_prepared[col].between(lower_b, upper_b)]
            
    # Modyfikacje biznesowe
    df_prepared.loc[:,'BuiltYear'] = pd.to_datetime(df_prepared['BuiltYear'], format='%Y', errors='coerce')
    df_prepared.loc[df_prepared['TypeOfMarket'] == 'pierwotny', 'BuildingCondition'] = 'DEVELOPER_STATE'
    if not df_prepared.empty:
        df_prepared.loc[df_prepared['BuiltYear'].dt.year >= 2024, 'BuildingCondition'] = 'DEVELOPER_STATE'
    condition = (df_prepared['Link'].str.contains('otodom', na=False)) & (df_prepared['BuildingCondition'] == 'AFTER_RENOVATION')
    df_prepared = df_prepared[~condition]

    # Balansowanie klas (undersampling)
    class_counts = df_prepared['BuildingCondition'].value_counts()
    if class_counts.empty or len(class_counts) < 2 or class_counts.min() == 0:
        logging.error("Niewystarczające dane do balansowania."); return None, None
    min_count = class_counts.min()
    dfs_balanced = [resample(df_prepared[df_prepared['BuildingCondition'] == c], replace=False, n_samples=min_count, random_state=42) for c in class_counts.index]
    df_balanced = pd.concat(dfs_balanced).reset_index(drop=True)

    # Konwersja typów kolumn ID na string
    string_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']
    for col in string_cols:
        if col in df_balanced.columns:
            df_balanced[col] = df_balanced[col].fillna(-1).astype(int).astype(str)
            
    return df_balanced, vectorizer

def main():
    logging.info("Uruchamiam skrypt treningowy stan_nasluch.py...")
    os.makedirs(DATA_INPUT_DIR_TRAINING, exist_ok=True)
    os.makedirs(ARTIFACTS_OUTPUT_DIR, exist_ok=True)
    processed_path = os.path.join(DATA_INPUT_DIR_TRAINING, PROCESSED_TRAINING_DATA_DIR_NAME)
    os.makedirs(processed_path, exist_ok=True)

    while True:
        try:
            input_file_path = os.path.join(DATA_INPUT_DIR_TRAINING, INPUT_CSV_FILENAME)
            
            if os.path.exists(input_file_path):
                logging.info(f"Znaleziono {INPUT_CSV_FILENAME}, rozpoczynam przetwarzanie.")
                time.sleep(2)
                
                df_input = pd.read_csv(input_file_path, sep=',')
                df_train_data, vectorizer = preprocess_data_for_training(df_input)

                if df_train_data is None:
                    shutil.move(input_file_path, os.path.join(processed_path, f"{INPUT_CSV_FILENAME}_error_preprocess.csv"))
                    continue

                # Krok 1: Inicjalizacja PyCaret
                logging.info("Konfiguruję eksperyment PyCaret (setup)...")
                categorical_features = [c for c in ['BuildingType', 'TypeOfMarket', 'OwnerType', 'Type', 'OfferFrom','VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber','RegionNumber', 'Location', 'StreetNumber'] if c in df_train_data.columns]
                numeric_features = [c for c in ['Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'CommunityScore'] if c in df_train_data.columns]
                date_features = [c for c in ['BuiltYear'] if c in df_train_data.columns]
                ignore_features = [c for c in ['SaleId', 'OriginalId', 'PortalId', 'Title', 'OfferPrice', 'RealPriceAfterRenovation', 'OriginalPrice','PricePerSquareMeter', 'DateAddedToDatabase', 'DateAdded','DateLastModification', 'DateLastRaises', 'NewestDate','AvailableFrom', 'Link', 'Phone', 'MainImage', 'OtherImages','NumberOfDuplicates', 'NumberOfRaises', 'NumberOfModifications','IsDuplicatePriceLower', 'IsDuplicatePrivateOwner', 'Score', 'ScorePrecision','NumberOfCommunityComments', 'NumberOfCommunityOpinions', 'Archive','SubRegionNumber', 'EncryptedId','id'] if c in df_train_data.columns]
                
                exp = setup(data=df_train_data, target='BuildingCondition', verbose=False, session_id=123,
                            categorical_features=categorical_features, numeric_features=numeric_features,
                            date_features=date_features, ignore_features=ignore_features)

                # Krok 2: Trening i finalizacja modelu
                logging.info("Tworzę i finalizuję model LightGBM...")
                lgbm_model = create_model('lightgbm', verbose=False)
                final_model_pipeline = finalize_model(lgbm_model)
                
                # Krok 3: Zapisywanie wszystkich artefaktów
                logging.info("Zapisywanie wszystkich artefaktów...")
                
                # 1. Zapisz CountVectorizer
                joblib.dump(vectorizer, os.path.join(ARTIFACTS_OUTPUT_DIR, VECTORIZER_FILENAME_OUTPUT))
                logging.info(f"Zapisano: {VECTORIZER_FILENAME_OUTPUT}")

                pipeline = get_config('pipeline')
                
                # 2. Zapisz OneHotEncoder
                try:
                    one_hot_encoder = pipeline.named_steps['onehot_encoding']
                    joblib.dump(one_hot_encoder, os.path.join(ARTIFACTS_OUTPUT_DIR, ENCODER_FILENAME_OUTPUT))
                    logging.info(f"Zapisano: {ENCODER_FILENAME_OUTPUT}")
                except Exception as e: logging.error(f"Błąd zapisu OneHotEncoder: {e}")

                # 3. Zapisz wartości z Imputera
                try:
                    imputer_numeric = pipeline.named_steps['numerical_imputer']
                    imputer_values = {'statistics': dict(zip(imputer_numeric.feature_names_in_, imputer_numeric.transformer.statistics_))}
                    with open(os.path.join(ARTIFACTS_OUTPUT_DIR, IMPUTER_VALUES_FILENAME_OUTPUT), 'w') as f: json.dump(imputer_values, f)
                    logging.info(f"Zapisano: {IMPUTER_VALUES_FILENAME_OUTPUT}")
                except Exception as e: logging.error(f"Błąd zapisu wartości z Imputera: {e}")
                
                # 4. Zapisz czysty model i jego finalne kolumny
                try:
                    clean_model = final_model_pipeline.steps[-1][1]
                    joblib.dump(clean_model, os.path.join(ARTIFACTS_OUTPUT_DIR, CLEAN_MODEL_FILENAME_OUTPUT))
                    logging.info(f"Zapisano: {CLEAN_MODEL_FILENAME_OUTPUT}")

                    final_columns = clean_model.feature_name_
                    with open(os.path.join(ARTIFACTS_OUTPUT_DIR, FINAL_MODEL_COLUMNS_FILENAME_OUTPUT), 'w') as f: json.dump(final_columns, f)
                    logging.info(f"Zapisano: {FINAL_MODEL_COLUMNS_FILENAME_OUTPUT}")
                except Exception as e: logging.error(f"Błąd zapisu modelu/kolumn: {e}")

                # Przeniesienie pliku
                shutil.move(input_file_path, os.path.join(processed_path, f"{INPUT_CSV_FILENAME}_processed_{time.strftime('%Y%m%d-%H%M%S')}.csv"))
                logging.info(f"Przeniesiono plik. Wracam do nasłuchiwania.")
            
            time.sleep(10)
        except Exception as e:
            logging.error(f"Wystąpił błąd w głównej pętli: {e}")
            traceback.print_exc()
            if 'input_file_path' in locals() and os.path.exists(input_file_path):
                shutil.move(input_file_path, os.path.join(processed_path, f"{INPUT_CSV_FILENAME}_CRITICAL_ERROR.csv"))
            time.sleep(20)

if __name__ == "__main__":
    main()