# WERSJA FINALNA: stan_nasluch_v3.py (Tworzy CZYSTY Pipeline)
import pandas as pd
from pycaret.classification import setup, create_model, save_model
from sklearn.utils import resample
import os
import time
import shutil
import logging
import sys

# --- Konfiguracja ---
DATA_INPUT_DIR_TRAINING = "DATA_STAN"
ARTIFACTS_OUTPUT_DIR = "DATA_STAN_OUT"
PROCESSED_TRAINING_DATA_DIR_NAME = "PROCESSED_DATA_STAN_TRAINING"
INPUT_CSV_FILENAME = "data.csv"
MODEL_FILENAME_OUTPUT = "model_LGBM_stan_final"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def clean_and_balance_data(df_input_original):
    logging.info("Rozpoczynam czyszczenie i balansowanie danych (BEZ ręcznej wektoryzacji)...")
    df = df_input_original.copy()
    
    initial_rows = len(df)
    df = df.dropna(subset=['Description', 'Location', 'BuildingCondition'])
    logging.info(f"Po dropna: {len(df)} wierszy (usunięto {initial_rows - len(df)}).")
    if df.empty: return None

    # NIE ROBIMY CountVectorizer. Zostawiamy Description w spokoju.
    df['Description'] = df['Description'].fillna('').str.slice(0, 3000)

    for col in ['Price', 'PricePerSquareMeter', 'Area']:
        if col in df.columns and not df[col].isnull().all() and len(df) > 1:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_b, upper_b = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df = df[df[col].between(lower_b, upper_b)]
            
    df.loc[df['TypeOfMarket'] == 'pierwotny', 'BuildingCondition'] = 'DEVELOPER_STATE'
    if 'BuiltYear' in df.columns:
        # Konwersja na datę jest OK, ale robimy ją przed setup()
        df['BuiltYear'] = pd.to_datetime(df['BuiltYear'], format='%Y', errors='coerce')
        if not df.empty:
            df.loc[df['BuiltYear'].dt.year >= 2024, 'BuildingCondition'] = 'DEVELOPER_STATE'
            
    condition = (df['Link'].str.contains('otodom', na=False)) & (df['BuildingCondition'] == 'AFTER_RENOVATION')
    df = df[~condition]

    class_counts = df['BuildingCondition'].value_counts()
    if class_counts.empty or len(class_counts) < 2 or class_counts.min() == 0:
        logging.error("Niewystarczające dane do balansowania."); return None
    min_count = class_counts.min()
    dfs_balanced = [resample(df[df['BuildingCondition'] == c], replace=False, n_samples=min_count, random_state=42) for c in class_counts.index]
    df_balanced = pd.concat(dfs_balanced).reset_index(drop=True)

    return df_balanced

def main():
    logging.info("Uruchamiam skrypt treningowy (wersja 'Czysty Pipeline')...")
    os.makedirs(ARTIFACTS_OUTPUT_DIR, exist_ok=True)
    processed_path = os.path.join(DATA_INPUT_DIR_TRAINING, PROCESSED_TRAINING_DATA_DIR_NAME)
    os.makedirs(processed_path, exist_ok=True)

    input_file_path = os.path.join(DATA_INPUT_DIR_TRAINING, INPUT_CSV_FILENAME)
    if not os.path.exists(input_file_path):
        logging.error(f"Nie znaleziono pliku {INPUT_CSV_FILENAME}. Zamykanie.")
        return

    df_input = pd.read_csv(input_file_path, sep=',')
    df_train_data = clean_and_balance_data(df_input)

    if df_train_data is None: return

    logging.info("Konfiguruję eksperyment PyCaret (z automatyczną obsługą tekstu)...")
    
    # Przekazujemy do PyCaret prawie surowe dane i pozwalamy mu zająć się tekstem
    exp = setup(data=df_train_data, target='BuildingCondition', session_id=123,
          text_features=['Description'], # <-- KLUCZOWA ZMIANA!
          ignore_features=['SaleId', 'OriginalId', 'PortalId', 'Title', 'Link', 'Phone', 'MainImage', 'OtherImages'],
          verbose=True) # Włączamy logi, żeby widzieć postęp

    logging.info("Tworzę model LightGBM...")
    lgbm_model = create_model('lightgbm', verbose=True)
    
    logging.info(f"Zapisywanie JEDNEGO, kompletnego pipeline'u do {ARTIFACTS_OUTPUT_DIR}...")
    save_model(lgbm_model, os.path.join(ARTIFACTS_OUTPUT_DIR, MODEL_FILENAME_OUTPUT))
    
    logging.info(f"Zapisano: {MODEL_FILENAME_OUTPUT}.pkl")
    shutil.move(input_file_path, os.path.join(processed_path, f"{INPUT_CSV_FILENAME}_processed_{time.strftime('%Y%m%d-%H%M%S')}.csv"))
    logging.info("Skrypt treningowy zakończył pracę.")

if __name__ == "__main__":
    main()