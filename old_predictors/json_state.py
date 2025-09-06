import pandas as pd
# import mlflow # Odkomentuj, jeśli będziesz używać MLflow w tym skrypcie
from pycaret.classification import load_model, predict_model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import time
import shutil
import json
import joblib

# --- Konfiguracja ---
DATA_INPUT_DIR = "2JSON_STATE"
DATA_OUTPUT_DIR = "2JSON_STATE_OUT"
PROCESSED_INPUT_DIR_NAME = "PROCESSED_JSONS"

MODEL_AND_VECTORIZER_SOURCE_DIR = "DATA_STAN_OUT"

# MLFLOW_TRACKING_URI = "http://localhost:5000" # Odkomentuj, jeśli używasz

MODEL_FILENAME = "model_LGBM_stan_final.pkl"
VECTORIZER_FILENAME = "count_vectorizer.joblib"

INPUT_JSON_FILENAME = "mieszkania.json"

# --- Funkcje pomocnicze ---

def preprocess_data_for_prediction(df_input_json, fitted_vectorizer, expected_columns_before_bow_from_training):
    """
    Przygotowuje dane z JSON do predykcji.
    expected_columns_before_bow_from_training: Lista nazw kolumn, które były w danych treningowych
                                              PRZED dodaniem cech Bag-of-Words i PRZED usunięciem 'Description'.
    """
    print("Rozpoczynam preprocessing danych z JSON do predykcji...")
    df = df_input_json.copy()

    # 1. Upewnij się, że wszystkie oczekiwane kolumny (poza BoW) istnieją.
    #    Jeśli brakuje, dodaj je z NaN lub odpowiednim placeholderem.
    #    'Description' jest kluczowa, jeśli jej nie ma, tworzymy pustą.
    if 'Description' not in df.columns:
        print("OSTRZEŻENIE: Brak kolumny 'Description' w JSON. Zostanie utworzona jako pusta.")
        df['Description'] = ''
    else:
        df['Description'] = df['Description'].fillna('')

    for col in expected_columns_before_bow_from_training:
        if col not in df.columns and col != 'Description': # Description już obsłużone
            # Rozróżnianie typu na podstawie nazwy to uproszczenie, lepsza byłaby informacja z treningu
            if col in ['Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'CommunityScore']:
                df[col] = np.nan
                print(f"Dodaję brakującą kolumnę numeryczną '{col}' z NaN.")
            elif col == 'BuiltYear':
                df[col] = pd.NaT
                print(f"Dodaję brakującą kolumnę daty '{col}' z NaT.")
            else: # Zakładamy, że reszta to kategoryczne/stringi
                df[col] = 'NA_placeholder'
                print(f"Dodaję brakującą kolumnę kategoryczną/tekstową '{col}' z 'NA_placeholder'.")
        elif col in df.columns and col != 'Description': # Jeśli kolumna istnieje, upewnij się o typie/NaN
            if col == 'BuiltYear':
                df[col] = pd.to_datetime(df[col], format='%Y', errors='coerce')
            elif col in ['Location'] or 'Number' in col or col in ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']: # Kolumny które mają być stringami
                 df[col] = df[col].fillna('NA_placeholder').astype(str)
            elif col in ['BuildingType', 'TypeOfMarket', 'OwnerType', 'Type', 'OfferFrom']: # Inne kategoryczne
                 df[col] = df[col].fillna('NA_placeholder').astype(str)
            # Numeryczne zostaną obsłużone przez PyCaret (imputacja)

    df.loc[:, 'Description'] = df['Description'].astype(str).str.slice(0, 3000)

    # 2. Transformacja 'Description' przez CountVectorizer
    print("Transformuję Description używając wczytanego CountVectorizer...")
    if fitted_vectorizer is None:
        print("BŁĄD: Obiekt Vectorizer nie został wczytany/przekazany.")
        return None
        
    X_bow_pred = fitted_vectorizer.transform(df["Description"])
    bow_feature_names = fitted_vectorizer.get_feature_names_out()
    df_bow_pred = pd.DataFrame(X_bow_pred.toarray(), columns=bow_feature_names, index=df.index)
    
    df_processed = df.drop(columns=['Description'], errors='ignore') 
    df_processed = pd.concat([df_processed, df_bow_pred], axis=1)

    # Upewnienie się, że wszystkie kolumny BoW istnieją i są numeryczne
    for bow_col in bow_feature_names:
        if bow_col not in df_processed.columns:
            df_processed[bow_col] = 0 
        else: # Upewnij się, że są numeryczne
            df_processed[bow_col] = pd.to_numeric(df_processed[bow_col], errors='coerce').fillna(0).astype(int)


    # 3. Konwersja typów kolumn "administracyjnych" na string (jak w setup)
    #    (Częściowo zrobione wyżej, ale tu upewniamy się)
    string_conversion_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']
    for col in string_conversion_cols:
        if col in df_processed.columns:
            df_processed.loc[:, col] = df_processed[col].astype(str) # Powinny już być stringami, ale dla pewności

    # Usuń kolumny, które nie są ani w expected_training_cols_info (oryginalne) ani w bow_feature_names
    # To jest trudniejsze, bo expected_training_cols_info nie zawiera info o typach po transformacji PyCaret
    # Zamiast tego, upewnijmy się, że przekazujemy tylko te kolumny, które były w `df_train_data`
    # użytego do `setup` (plus nowe cechy BoW).
    
    # Stwórz listę wszystkich kolumn, które powinny być w danych końcowych
    final_expected_cols = [c for c in expected_columns_before_bow_from_training if c != 'Description'] + list(bow_feature_names)
    
    # Dodaj brakujące (jeśli jakieś jeszcze są po powyższych krokach - nie powinno być)
    for col in final_expected_cols:
        if col not in df_processed.columns:
            # To nie powinno się zdarzyć, jeśli expected_columns_before_bow_from_training jest dobre
            print(f"KRYTYCZNE OSTRZEŻENIE: Kolumna '{col}' nadal brakuje po preprocessingu. Dodaję z NaN/placeholderem.")
            if col in numeric_cols_setup: df_processed[col] = np.nan # Zakładając, że numeric_cols_setup jest dostępne
            else: df_processed[col] = 'NA_placeholder'

    # Usuń nadmiarowe kolumny, które nie są w final_expected_cols
    cols_to_remove = [col for col in df_processed.columns if col not in final_expected_cols]
    if cols_to_remove:
        print(f"Usuwam nadmiarowe kolumny: {cols_to_remove}")
        df_processed = df_processed.drop(columns=cols_to_remove)
        
    # Ustawienie kolejności kolumn (opcjonalnie, ale PyCaret bywa na to wrażliwy)
    # df_processed = df_processed[final_expected_cols] # To może być zbyt restrykcyjne jeśli kolejność się lekko zmieniła

    print(f"Zakończono preprocessing dla danych z JSON. Kształt finalny: {df_processed.shape}")
    return df_processed

# --- Główna logika skryptu ---
def main():
    print(f"Skrypt predykcyjny uruchomiony. Nasłuchuję pliku {INPUT_JSON_FILENAME} w folderze {DATA_INPUT_DIR}...")
    
    try:
        os.makedirs(DATA_INPUT_DIR, exist_ok=True)
        os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
        os.makedirs(MODEL_AND_VECTORIZER_SOURCE_DIR, exist_ok=True)
        processed_input_files_path = os.path.join(DATA_INPUT_DIR, PROCESSED_INPUT_DIR_NAME)
        os.makedirs(processed_input_files_path, exist_ok=True)
        print(f"Foldery zweryfikowane/utworzone.")
    except OSError as e:
        print(f"BŁĄD KRYTYCZNY: Nie można utworzyć wymaganych folderów: {e}")
        return

    model_path = os.path.join(MODEL_AND_VECTORIZER_SOURCE_DIR, MODEL_FILENAME)
    vectorizer_path = os.path.join(MODEL_AND_VECTORIZER_SOURCE_DIR, VECTORIZER_FILENAME) 

    final_model = None
    vectorizer = None
    
    # --- Wczytywanie artefaktów ---
    try:
        final_model = load_model(model_path[:-4]) 
        print(f"Model wczytany z: {model_path}")
        # Próba uzyskania listy kolumn użytych w setup() skryptu treningowego
        # To jest BARDZO WAŻNE dla spójności danych wejściowych
        # PyCaret 3.x: final_model.pipeline_[:-1] to transformery, final_model.pipeline_[-1] to estymator
        # Możemy spróbować uzyskać cechy z pierwszego kroku transformacji (jeśli to możliwe)
        # lub z `final_model.feature_names_in_` jeśli model był prosty (bez pipeline'u PyCaret, co tu nie zachodzi)
        # Najlepiej byłoby, gdyby skrypt treningowy ZAPISAŁ listę kolumn z `df_balanced_for_setup.columns`
        # PRZED dodaniem cech BoW i PRZED przekazaniem do setup().
        # NA RAZIE UŻYJEMY ZGADYWANIA - TO MUSI BYĆ POPRAWIONE DLA NIEZAWODNOŚCI
        
        # ----- POCZĄTEK SEKCJI DO POTENCJALNEJ POPRAWY (lista kolumn) -----
        # Te listy powinny dokładnie odpowiadać temu, co było w `categorical_features_to_use`, 
        # `numeric_features_to_use`, `date_features_to_use` w setup() skryptu treningowego,
        # PLUS inne kolumny, które były w `df_train_data` ale zostały zignorowane przez `ignore_features`.
        # Najlepiej byłoby wczytać tę listę z pliku zapisanego przez skrypt treningowy.
        
        # Kolumny, które były PRZED dodaniem BoW i przed setup().
        # To jest przybliżenie, bazujące na Twoim notebooku.
        expected_cols_before_bow = [
            'SaleId', 'OriginalId', 'PortalId', 'Title', 'Area', 'Price', 'OfferPrice', 
            'RealPriceAfterRenovation', 'OriginalPrice', 'PricePerSquareMeter', 
            'NumberOfRooms', 'BuiltYear', 'Type', 'BuildingType', 'OfferFrom', 'Floor', 
            'Floors', 'TypeOfMarket', 'OwnerType', 'DateAddedToDatabase', 'DateAdded',
            'DateLastModification', 'DateLastRaises', 'NewestDate', 'AvailableFrom', 
            'Link', 'Phone', 'MainImage', 'OtherImages', 'NumberOfDuplicates', 
            'NumberOfRaises', 'NumberOfModifications', 'IsDuplicatePriceLower', 
            'IsDuplicatePrivateOwner', 'Score', 'ScorePrecision', 'CommunityScore',
            'NumberOfCommunityComments', 'NumberOfCommunityOpinions', 'Archive', 
            'Location', 'VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 
            'KindNumber', 'RegionNumber', 'SubRegionNumber', 'StreetNumber', 'EncryptedId'
            # Pamiętaj, że 'BuildingCondition' (target) nie powinno tu być, bo jest dodawane przez PyCaret
            # lub usuwane przed predykcją
        ]
        print(f"Używam predefiniowanej listy {len(expected_cols_before_bow)} oczekiwanych kolumn (przed BoW).")
        # ----- KONIEC SEKCJI DO POTENCJALNEJ POPRAWY -----

    except Exception as e:
        print(f"BŁĄD KRYTYCZNY podczas wczytywania modelu: {e}"); return
    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer wczytany z: {vectorizer_path}")
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY podczas wczytywania vectorizera: {e}"); return

    # ----- Pętla nasłuchiwania -----
    while True:
        try:
            input_json_full_path = os.path.join(DATA_INPUT_DIR, INPUT_JSON_FILENAME)
            if os.path.exists(input_json_full_path):
                print(f"\nZnaleziono plik {INPUT_JSON_FILENAME} w {DATA_INPUT_DIR}")
                time.sleep(1)
                df_from_json = None
                try:
                    with open(input_json_full_path, 'r', encoding='utf-8') as f:
                        json_data_content = f.read()
                    if not json_data_content.strip():
                        print(f"Plik {INPUT_JSON_FILENAME} jest pusty. Pomijam.")
                        df_from_json = pd.DataFrame()
                    else:
                        raw_json_data = json.loads(json_data_content)
                        if isinstance(raw_json_data, dict): # Pojedynczy obiekt
                            df_from_json = pd.DataFrame([raw_json_data])
                        elif isinstance(raw_json_data, list): # Lista obiektów
                            if not raw_json_data: # Pusta lista
                                df_from_json = pd.DataFrame()
                            else:
                                df_from_json = pd.DataFrame(raw_json_data)
                        else:
                            raise ValueError(f"Nieznany główny typ danych w JSON: {type(raw_json_data)}")
                except Exception as e:
                    print(f"BŁĄD podczas wczytywania/parsowania JSON '{input_json_full_path}': {e}")
                    # ... (przenoszenie pliku z błędem) ...
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(input_json_full_path, os.path.join(processed_input_files_path, f"{INPUT_JSON_FILENAME}_parsing_error_{timestamp}.json"))
                    print(f"Przeniesiono {INPUT_JSON_FILENAME} do {processed_input_files_path} jako plik z błędem parsowania.")
                    time.sleep(5)
                    continue

                if df_from_json is None or df_from_json.empty:
                    # ... (obsługa pustego pliku) ...
                    print(f"Plik {INPUT_JSON_FILENAME} (po próbie wczytania) jest pusty. Pomijam.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(input_json_full_path, os.path.join(processed_input_files_path, f"{INPUT_JSON_FILENAME}_empty_or_read_fail_{timestamp}.json"))
                    print(f"Przeniesiono {INPUT_JSON_FILENAME} do {processed_input_files_path}")
                    time.sleep(5)
                    continue
                
                print(f"Wczytano dane z {INPUT_JSON_FILENAME}, kształt: {df_from_json.shape}")
                df_original_json_for_output = df_from_json.copy() 

                df_processed_json = preprocess_data_for_prediction(df_from_json, vectorizer, expected_cols_before_bow)

                if df_processed_json is None:
                    # ... (obsługa błędu preprocessingu) ...
                    print("Preprocessing danych z JSON nie powiódł się. Pomijam ten plik.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(input_json_full_path, os.path.join(processed_input_files_path, f"{INPUT_JSON_FILENAME}_preprocess_error_{timestamp}.json"))
                    print(f"Przeniesiono {INPUT_JSON_FILENAME} do {processed_input_files_path} jako plik z błędem preprocessingu.")
                    time.sleep(5)
                    continue
                
                data_for_prediction_final_step = df_processed_json.copy()
                target_column_name = 'BuildingCondition' 
                if target_column_name in data_for_prediction_final_step.columns:
                    data_for_prediction_final_step = data_for_prediction_final_step.drop(columns=[target_column_name])

                # --- DODATKOWY DEBUG PRZED predict_model ---
                print("\n--- DEBUG tuż PRZED predict_model ---")
                print(f"Kształt danych do predykcji: {data_for_prediction_final_step.shape}")
                # Sprawdźmy kilka pierwszych i ostatnich nazw kolumn
                print(f"Pierwsze 5 kolumn: {list(data_for_prediction_final_step.columns[:5])}")
                print(f"Ostatnie 5 kolumn: {list(data_for_prediction_final_step.columns[-5:])}")
                # Sprawdź typy dla kilku przykładowych kolumn, które mogłyby być problematyczne
                cols_to_check_type = ['BuildingType', 'TypeOfMarket', 'VoivodeshipNumber', 'Area'] + list(data_for_prediction_final_step.columns[50:52]) # Przykładowe BoW
                for col_check in cols_to_check_type:
                    if col_check in data_for_prediction_final_step.columns:
                        print(f"  Typ kolumny '{col_check}': {data_for_prediction_final_step[col_check].dtype}, Przykład: {data_for_prediction_final_step[col_check].iloc[0] if len(data_for_prediction_final_step)>0 else 'brak danych'}")
                print("--- KONIEC DEBUGU PRZED predict_model ---")
                # --- KONIEC DODATKOWEGO DEBUGU ---

                print("Wykonuję predykcje na danych z JSON...")
                predictions_object = predict_model(
                    estimator=final_model, 
                    data=data_for_prediction_final_step
                )

                df_results = df_original_json_for_output.copy() 
                df_results['Predict_State'] = predictions_object['prediction_label'].values
                
                timestamp_out = time.strftime("%Y%m%d-%H%M%S")
                output_json_filename = f"mieszkania_state_out.json"
                output_json_path = os.path.join(DATA_OUTPUT_DIR, output_json_filename) 
                
                df_results.to_json(output_json_path, orient='records', indent=4, force_ascii=False, default_handler=str)
                print(f"Wyniki predykcji zapisane w: {output_json_path}")

                shutil.move(input_json_full_path, os.path.join(processed_input_files_path, f"{INPUT_JSON_FILENAME}_processed_{timestamp_out}.json"))
                print(f"Przeniesiono {INPUT_JSON_FILENAME} do {processed_input_files_path}")
                
                print(f"Zakończono przetwarzanie {INPUT_JSON_FILENAME}. Oczekuję na nowy plik...")
            else:
                pass
            time.sleep(5)
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd w głównej pętli monitorowania: {e}")
            import traceback
            traceback.print_exc()
            if 'input_json_full_path' in locals() and os.path.exists(input_json_full_path):
                try:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    failed_file_name = os.path.basename(input_json_full_path)
                    shutil.move(input_json_full_path, os.path.join(processed_input_files_path, f"{os.path.splitext(failed_file_name)[0]}_CRITICAL_ERROR_{timestamp}{os.path.splitext(failed_file_name)[1]}"))
                    print(f"Przeniesiono {input_json_full_path} (powodujący błąd krytyczny) do {processed_input_files_path}")
                except Exception as move_e:
                    print(f"Nie udało się przenieść pliku powodującego błąd krytyczny: {move_e}")
            time.sleep(20)

if __name__ == "__main__":
    main()