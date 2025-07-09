import pandas as pd
import numpy as np
import os
import time
import shutil
import json
import joblib
import traceback

# --- Konfiguracja ---
SOURCE_DIR = "DATA_STATE_OUT"
DATA_INPUT_DIR = "2JSON_STATE"
DATA_OUTPUT_DIR = "2JSON_STATE_OUT"
PROCESSED_DIR = os.path.join(DATA_INPUT_DIR, "PROCESSED_JSONS")
INPUT_JSON_FILENAME = "mieszkania.json"

def main():
    print("Uruchomiono skrypt predykcyjny (wersja v4 - Elastyczny odczyt JSON)...")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    try:
        print("Wczytywanie artefaktów...")
        vectorizer = joblib.load(os.path.join(SOURCE_DIR, "count_vectorizer.joblib"))
        encoder = joblib.load(os.path.join(SOURCE_DIR, "one_hot_encoder.pkl"))
        model = joblib.load(os.path.join(SOURCE_DIR, "clean_lgbm_model.pkl"))
        with open(os.path.join(SOURCE_DIR, "imputer_values.json"), 'r') as f:
            imputer_values = json.load(f)
        with open(os.path.join(SOURCE_DIR, "final_model_columns.json"), 'r') as f:
            final_model_columns = json.load(f)
        print("Artefakty wczytane.")
    except Exception as e:
        print(f"KRYTYCZNY BŁĄD: Nie można wczytać modeli. Zamykanie. Błąd: {e}")
        return

    while True:
        input_path = os.path.join(DATA_INPUT_DIR, INPUT_JSON_FILENAME)
        
        if os.path.exists(input_path):
            print(f"\nZnaleziono plik: {input_path}")
            
            try:
                time.sleep(0.5) 
                
                with open(input_path, 'r', encoding='utf-8') as f:
                    json_object = json.load(f)

                # Sprawdzamy, czy to słownik z jednym kluczem
                if not isinstance(json_object, dict) or len(json_object.keys()) != 1:
                    raise ValueError("Format JSON jest nieprawidłowy. Oczekiwano słownika z jednym kluczem głównym.")

                # Pobieramy listę spod tego JEDYNEGO klucza, bez względu na jego nazwę
                data = list(json_object.values())[0]

                if not isinstance(data, list):
                    raise ValueError("Wartość pod kluczem głównym nie jest listą.")

                df = pd.DataFrame(data)
                df_original = df.copy()

                # --- PREPROCESSING ---
                print("1. Preprocessing danych...")
                numeric_cols = list(imputer_values['statistics'].keys())
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(imputer_values['statistics'].get(col, 0))
                
                if 'BuiltYear' in df.columns:
                    df['BuiltYear'] = pd.to_datetime(df['BuiltYear'], errors='coerce')
                    df['BuiltYear_year'] = df['BuiltYear'].dt.year.fillna(2015.0)
                    df = df.drop(columns=['BuiltYear'])

                categorical_cols = encoder.transformer.feature_names_in_
                for col in categorical_cols:
                    if col not in df.columns: df[col] = 'missing'
                df_cat = df[categorical_cols].astype(str).fillna('missing')
                encoded_data = encoder.transform(df_cat)
                df_encoded = pd.DataFrame(encoded_data, columns=encoder.transformer.get_feature_names_out(categorical_cols), index=df.index)

                if 'Description' not in df.columns: df['Description'] = ''
                df['Description'] = df['Description'].fillna('').str.slice(0, 3000)
                X_bow = vectorizer.transform(df['Description'])
                df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index)

                df_numeric = df.select_dtypes(include=np.number)
                df_final = pd.concat([df_numeric, df_encoded, df_bow], axis=1)
                df_final = df_final.reindex(columns=final_model_columns, fill_value=0)

                # --- PREDYKCJA I TŁUMACZENIE ---
                print("2. Wykonywanie predykcji i tłumaczenie na etykiety...")
                predictions_numeric = model.predict(df_final)

                label_map = {
                    0: 'AFTER_RENOVATION',
                    1: 'DEVELOPER_STATE',
                    2: 'FOR_RENOVATION',
                    3: 'GOOD'
                }
                
                predictions_labels = [label_map.get(p, 'UNKNOWN') for p in predictions_numeric]
                df_original['Predict_State'] = predictions_labels
                
                # --- ZAPIS ---
                print("3. Zapisywanie wyniku...")
                output_path = os.path.join(DATA_OUTPUT_DIR, "mieszkania_state_out.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({'data': df_original.to_dict('records')}, f, ensure_ascii=False, indent=4)
                
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