# WERSJA FINALNA "KLON ORKIESTRATORA": json_state_v3.py
import pandas as pd
import numpy as np
import os
import time
import shutil
import json
import joblib
import traceback
import logging
import sys
from pycaret.classification import load_model, predict_model

# --- Konfiguracja ---
SOURCE_DIR = "DATA_STAN_OUT"
MODEL_FILENAME = "model_LGBM_stan_final"
VECTORIZER_FILENAME = "count_vectorizer.joblib"

DATA_INPUT_DIR = "2JSON_STATE"
DATA_OUTPUT_DIR = "2JSON_STATE_OUT"
PROCESSED_DIR = os.path.join(DATA_INPUT_DIR, "PROCESSED_JSONS")
INPUT_JSON_FILENAME = "mieszkania.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# --- WIERNA REPLIKA FUNKCJI PREPROCESSingu Z TWOJEGO `start_all.py` ---
def preprocess_for_prediction(df_input_original, fitted_vectorizer):
    logging.info("Rozpoczynam preprocessing danych (logika z orkiestratora)...")
    df = df_input_original.copy()

    # Ta lista kolumn i definicje są skopiowane 1:1 z Twojego działającego skryptu
    expected_columns_before_bow = [
        'SaleId', 'OriginalId', 'PortalId', 'Title', 'Area', 'Price', 'OfferPrice', 
        'RealPriceAfterRenovation', 'OriginalPrice', 'PricePerSquareMeter', 'Description',
        'NumberOfRooms', 'BuiltYear', 'Type', 'BuildingType', 'OfferFrom', 'Floor', 
        'Floors', 'TypeOfMarket', 'OwnerType', 'DateAddedToDatabase', 'DateAdded',
        'DateLastModification', 'DateLastRaises', 'NewestDate', 'AvailableFrom', 
        'Link', 'Phone', 'MainImage', 'OtherImages', 'NumberOfDuplicates', 
        'NumberOfRaises', 'NumberOfModifications', 'IsDuplicatePriceLower', 
        'IsDuplicatePrivateOwner', 'Score', 'ScorePrecision', 'CommunityScore',
        'NumberOfCommunityComments', 'NumberOfCommunityOpinions', 'Archive', 
        'Location', 'VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 
        'KindNumber', 'RegionNumber', 'SubRegionNumber', 'StreetNumber', 'EncryptedId'
    ]
    numeric_cols_setup = ['Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'CommunityScore']

    df['Description'] = df['Description'].fillna('').astype(str).str.slice(0, 3000)

    # Uzupełnianie brakujących kolumn
    for col in expected_columns_before_bow:
        if col not in df.columns:
            if col in numeric_cols_setup: df[col] = np.nan
            elif col == 'BuiltYear': df[col] = pd.NaT
            else: df[col] = 'NA_placeholder'
    
    # Konwersja typów, dokładnie jak w orkiestratorze
    df['BuiltYear'] = pd.to_datetime(df['BuiltYear'], format='%Y', errors='coerce')
    string_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int).astype(str)
    
    # Wektoryzacja opisu, dokładnie jak w orkiestratorze
    X_bow_pred = fitted_vectorizer.transform(df["Description"])
    df_bow_pred = pd.DataFrame(X_bow_pred.toarray(), columns=fitted_vectorizer.get_feature_names_out(), index=df.index)
    
    df_processed = df.drop(columns=['Description'])
    df_processed = pd.concat([df_processed, df_bow_pred], axis=1)

    logging.info(f"Zakończono preprocessing. Kształt danych: {df_processed.shape}")
    return df_processed

def main():
    print("Uruchomiono skrypt predykcyjny (wersja 'Klon Orkiestratora')...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    try:
        model_path = os.path.join(SOURCE_DIR, MODEL_FILENAME)
        vectorizer_path = os.path.join(SOURCE_DIR, VECTORIZER_FILENAME)
        model = load_model(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Artefakty (model i vectorizer) wczytane pomyślnie.")
    except Exception as e:
        print(f"KRYTYCZNY BŁĄD: Nie można wczytać modeli. Błąd: {e}")
        return

    while True:
        input_path = os.path.join(DATA_INPUT_DIR, INPUT_JSON_FILENAME)
        if os.path.exists(input_path):
            print(f"\nZnaleziono plik: {input_path}")
            try:
                time.sleep(0.5)
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = list(json.load(f).values())[0]
                df_original = pd.DataFrame(data)

                # --- PREPROCESSING I PREDYKCJA ---
                df_processed = preprocess_for_prediction(df_original.copy(), vectorizer)
                
                if 'BuildingCondition' in df_processed.columns:
                    df_processed = df_processed.drop(columns=['BuildingCondition'])

                print("Wykonywanie predykcji za pomocą predict_model...")
                predictions_df = predict_model(estimator=model, data=df_processed)
                
                df_original['Predict_State'] = predictions_df['prediction_label'].values

                # --- ZAPIS ---
                print("Zapisywanie wyniku...")
                output_path = os.path.join(DATA_OUTPUT_DIR, "mieszkania_state_out.json")
                if 'BuiltYear' in df_original.columns and pd.api.types.is_datetime64_any_dtype(df_original['BuiltYear']):
                     df_original['BuiltYear'] = df_original['BuiltYear'].dt.year
                
                output_data = {'data': df_original.to_dict('records')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
                print(f"SUKCES! Wynik zapisany w: {output_path}")

                processed_path = os.path.join(PROCESSED_DIR, f"mieszkania_processed_{int(time.time())}.json")
                shutil.move(input_path, processed_path)
                print(f"Plik wejściowy przeniesiony do: {processed_path}")

            except Exception as e:
                print(f"BŁĄD PRZETWARZANIA: {e}")
                traceback.print_exc()
                error_path = os.path.join(PROCESSED_DIR, f"mieszkania_error_{int(time.time())}.json")
                if os.path.exists(input_path):
                    shutil.move(input_path, error_path)
        time.sleep(2)

if __name__ == "__main__":
    main()