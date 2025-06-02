import os
import shutil
import time
import subprocess
import pandas as pd
import logging
import sys 
import joblib 
from pycaret.classification import load_model, predict_model 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 

# --- Konfiguracja Globalna dla start_all.py ---
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) 

DIR_ALL_INPUT = os.path.join(BASE_PROJECT_DIR, "ALL")
DIR_ALL_OUTPUT = os.path.join(BASE_PROJECT_DIR, "ALL_OUT")
DIR_ALL_PROCESSED = os.path.join(DIR_ALL_INPUT, "PROCESSED_ALL")

SCRIPT_STAN_NASLUCH = "stan_nasluch.py" 
SCRIPT_MODEL_WATCHER = "model_watcher_v1_3.py" 

DIR_STAN_INPUT = os.path.join(BASE_PROJECT_DIR, "DATA_STAN")
DIR_STAN_OUTPUT_ARTIFACTS = os.path.join(BASE_PROJECT_DIR, "DATA_STAN_OUT") 
PROCESSED_STAN_TRAINING_DIR = os.path.join(DIR_STAN_INPUT, "PROCESSED_DATA_STAN_TRAINING")

MODEL_FILENAME_FROM_STAN = "model_LGBM_stan_final.pkl" 
VECTORIZER_FILENAME_FROM_STAN = "count_vectorizer.joblib" 

DIR_WATCHER_INPUT = os.path.join(BASE_PROJECT_DIR, "DATA")
DIR_WATCHER_OUTPUT = os.path.join(BASE_PROJECT_DIR, "DATA_OUT")
FINAL_OUTPUT_FILENAME = "data_all_out.csv" 

INPUT_DATA_CSV = "data.csv" 
TEMP_PREDICTED_STATE_CSV_BASENAME = "data_with_predicted_state"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# --- Funkcje pomocnicze ---
def create_directories():
    logging.info("Tworzenie wymaganych katalogów...")
    os.makedirs(DIR_ALL_INPUT, exist_ok=True)
    os.makedirs(DIR_ALL_OUTPUT, exist_ok=True)
    os.makedirs(DIR_ALL_PROCESSED, exist_ok=True)
    os.makedirs(DIR_STAN_INPUT, exist_ok=True)
    os.makedirs(DIR_STAN_OUTPUT_ARTIFACTS, exist_ok=True)
    os.makedirs(PROCESSED_STAN_TRAINING_DIR, exist_ok=True)
    os.makedirs(DIR_WATCHER_INPUT, exist_ok=True)
    os.makedirs(DIR_WATCHER_OUTPUT, exist_ok=True)
    logging.info("Katalogi gotowe.")

def run_script(script_name, cwd=None, args_list=None, wait_for_completion=True):
    try:
        script_path = os.path.join(BASE_PROJECT_DIR, script_name)
        if not os.path.exists(script_path):
            logging.error(f"Skrypt {script_name} nie został znaleziony w {BASE_PROJECT_DIR}")
            return None if wait_for_completion else False
        
        command = [sys.executable, script_path]
        if args_list:
            command.extend(args_list)
            
        logging.info(f"Uruchamianie polecenia: {' '.join(command)}...")
        process = subprocess.Popen(command, cwd=cwd if cwd else BASE_PROJECT_DIR)
        
        if wait_for_completion:
            process.wait() 
            if process.returncode == 0:
                logging.info(f"Skrypt {script_name} zakończony pomyślnie.")
                return True
            else:
                logging.error(f"Skrypt {script_name} zakończony z kodem błędu: {process.returncode}")
                return False
        else:
            logging.info(f"Skrypt {script_name} uruchomiony w tle (PID: {process.pid}).")
            return process 
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania skryptu {script_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None if wait_for_completion else False

def preprocess_data_for_prediction_in_orchestrator(df_input_original, fitted_vectorizer, expected_columns_before_bow):
    logging.info("Rozpoczynam preprocessing danych (dla funkcji add_predictions...)")
    df = df_input_original.copy()

    categorical_cols_setup = ['BuildingType', 'TypeOfMarket', 'OwnerType', 'Type', 'OfferFrom',
                              'VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber',
                              'RegionNumber', 'Location', 'StreetNumber']
    numeric_cols_setup = ['Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'CommunityScore']
    date_col_setup = 'BuiltYear'

    if 'Description' not in df.columns:
        df['Description'] = ''
    else:
        df['Description'] = df['Description'].fillna('')

    for col in expected_columns_before_bow:
        if col not in df.columns and col != 'Description': 
            if col == date_col_setup: df[col] = pd.NaT
            elif col in numeric_cols_setup: df[col] = np.nan
            elif col in categorical_cols_setup or col == 'Location': df[col] = 'NA_placeholder'
            else: df[col] = 'NA_placeholder'
        elif col in df.columns and col != 'Description':
            if col == 'BuiltYear':
                df[col] = pd.to_datetime(df[col], format='%Y', errors='coerce')
            elif col == 'Location' or ('Number' in col and col != 'NumberOfRooms') or col in ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']:
                 df[col] = df[col].fillna('NA_placeholder').astype(str)
            elif col in ['BuildingType', 'TypeOfMarket', 'OwnerType', 'Type', 'OfferFrom']:
                 df[col] = df[col].fillna('NA_placeholder').astype(str)

    df.loc[:, 'Description'] = df['Description'].astype(str).str.slice(0, 3000)

    if fitted_vectorizer is None:
        logging.error("BŁĄD: Obiekt Vectorizer nie został wczytany/przekazany do preprocessingu w orkiestratorze.")
        return None
        
    X_bow_pred = fitted_vectorizer.transform(df["Description"])
    bow_feature_names = fitted_vectorizer.get_feature_names_out()
    df_bow_pred = pd.DataFrame(X_bow_pred.toarray(), columns=bow_feature_names, index=df.index)
    
    df_processed = df.drop(columns=['Description'], errors='ignore') 
    df_processed = pd.concat([df_processed, df_bow_pred], axis=1)

    for bow_col in bow_feature_names:
        if bow_col not in df_processed.columns:
            df_processed[bow_col] = 0 
        else: 
            df_processed[bow_col] = pd.to_numeric(df_processed[bow_col], errors='coerce').fillna(0).astype(int)

    string_conversion_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']
    for col in string_conversion_cols:
        if col in df_processed.columns:
            if not pd.api.types.is_string_dtype(df_processed[col]):
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed.loc[:, col] = df_processed[col].fillna(-1).astype(int).astype(str)
                else: 
                    df_processed.loc[:, col] = df_processed[col].fillna('NA_placeholder').astype(str)
            
    for col in numeric_cols_setup:
        if col in df_processed.columns:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed.loc[:, col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    for col in categorical_cols_setup:
        if col in df_processed.columns and col not in string_conversion_cols: 
            if not pd.api.types.is_string_dtype(df_processed[col]) and not pd.api.types.is_object_dtype(df_processed[col]):
                df_processed.loc[:, col] = df_processed[col].fillna('NA_placeholder').astype(str)
    
    final_expected_cols = [c for c in expected_columns_before_bow if c != 'Description'] + list(bow_feature_names)
    
    for col in final_expected_cols:
        if col not in df_processed.columns:
            if col in numeric_cols_setup: df_processed[col] = np.nan
            else: df_processed[col] = 'NA_placeholder'

    cols_to_remove = [col for col in df_processed.columns if col not in final_expected_cols]
    if cols_to_remove:
        logging.info(f"Usuwam nadmiarowe kolumny (w add_predictions): {cols_to_remove}")
        df_processed = df_processed.drop(columns=cols_to_remove)
        
    logging.info(f"Zakończono preprocessing dla danych (w add_predictions). Kształt finalny: {df_processed.shape}")
    return df_processed

def create_data_csv_with_predicted_state(original_csv_path, model_artifact_path, vectorizer_artifact_path, 
                                         output_path_data_plus_target):
    logging.info(f"Tworzenie pliku z podmienioną kolumną BuildingCondition dla: {original_csv_path}")
    try:
        df_original_to_modify = pd.read_csv(original_csv_path, sep=',')
        vectorizer_loaded = joblib.load(vectorizer_artifact_path)
        model_loaded = load_model(model_artifact_path[:-4])
        
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
        ]
        expected_cols_before_bow = list(set(expected_cols_before_bow))

        df_processed_for_state_pred = preprocess_data_for_prediction_in_orchestrator(
            df_original_to_modify.copy(), vectorizer_loaded, expected_cols_before_bow # Przekaż kopię
        )

        if df_processed_for_state_pred is None: 
            logging.error("Preprocessing (w orkiestratorze) dla predykcji stanu nie powiódł się.")
            return False

        data_for_state_prediction_step = df_processed_for_state_pred.copy()
        if 'BuildingCondition' in data_for_state_prediction_step.columns:
            data_for_state_prediction_step = data_for_state_prediction_step.drop(columns=['BuildingCondition'])

        logging.info("Wykonywanie predykcji stanu (w orkiestratorze)...")
        predictions_state_object = predict_model(estimator=model_loaded, data=data_for_state_prediction_step)

        df_output_data_plus = df_original_to_modify.copy()
        if df_output_data_plus.shape[0] == predictions_state_object.shape[0]:
            df_output_data_plus['BuildingCondition'] = predictions_state_object['prediction_label'].values
        else:
            logging.warning(f"Niezgodność liczby wierszy przy tworzeniu {output_path_data_plus_target}. Oryg.: {df_output_data_plus.shape[0]}, Pred.: {predictions_state_object.shape[0]}")
            return False

        df_output_data_plus.to_csv(output_path_data_plus_target, index=False, sep=',')
        logging.info(f"Plik z podmienionym BuildingCondition zapisany jako: {output_path_data_plus_target}")
        return True
    except Exception as e:
        logging.error(f"Błąd w funkcji create_data_csv_with_predicted_state: {e}")
        import traceback; logging.error(traceback.format_exc())
        return False

# --- Główna funkcja start_all.py ---
def main_orchestrator():
    create_directories()
    logging.info(f"Orkiestrator uruchomiony. Nasłuchuję folderu: {DIR_ALL_INPUT} na plik {INPUT_DATA_CSV}")
    
    processed_one_file_successfully = False
    while not processed_one_file_successfully:
        try:
            main_input_file_path = os.path.join(DIR_ALL_INPUT, INPUT_DATA_CSV)

            if os.path.exists(main_input_file_path):
                logging.info(f"Znaleziono plik {INPUT_DATA_CSV} w {DIR_ALL_INPUT}.")
                time.sleep(2) 

                stan_input_path = os.path.join(DIR_STAN_INPUT, INPUT_DATA_CSV)
                if os.path.exists(stan_input_path): os.remove(stan_input_path)
                shutil.copy(main_input_file_path, stan_input_path)
                logging.info(f"Skopiowano {INPUT_DATA_CSV} do {DIR_STAN_INPUT} dla skryptu {SCRIPT_STAN_NASLUCH}.")
                
                if not run_script(SCRIPT_STAN_NASLUCH, wait_for_completion=True):
                    logging.error(f"Skrypt {SCRIPT_STAN_NASLUCH} nie zakończył się pomyślnie. Przerywam ten cykl.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(main_input_file_path, os.path.join(DIR_ALL_PROCESSED, f"{INPUT_DATA_CSV}_error_stan_nasluch_{timestamp}.csv"))
                    time.sleep(10); continue 
                
                model_stan_path = os.path.join(DIR_STAN_OUTPUT_ARTIFACTS, MODEL_FILENAME_FROM_STAN)
                vectorizer_stan_path = os.path.join(DIR_STAN_OUTPUT_ARTIFACTS, VECTORIZER_FILENAME_FROM_STAN)

                if not (os.path.exists(model_stan_path) and os.path.exists(vectorizer_stan_path)):
                    logging.error(f"Nie znaleziono modelu stanu lub vectorizera stanu w {DIR_STAN_OUTPUT_ARTIFACTS} po uruchomieniu {SCRIPT_STAN_NASLUCH}.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(main_input_file_path, os.path.join(DIR_ALL_PROCESSED, f"{INPUT_DATA_CSV}_error_stan_artifacts_{timestamp}.csv"))
                    time.sleep(10); continue
                
                temp_file_for_watcher_name = f"{TEMP_PREDICTED_STATE_CSV_BASENAME}_{time.strftime('%Y%m%d%H%M%S')}.csv"
                path_to_temp_data_for_watcher = os.path.join(DIR_ALL_PROCESSED, temp_file_for_watcher_name)

                if not create_data_csv_with_predicted_state(
                    original_csv_path=main_input_file_path, 
                    model_artifact_path=model_stan_path,
                    vectorizer_artifact_path=vectorizer_stan_path,
                    output_path_data_plus_target=path_to_temp_data_for_watcher
                ):
                    logging.error("Nie udało się stworzyć pliku CSV z podmienioną kolumną BuildingCondition.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(main_input_file_path, os.path.join(DIR_ALL_PROCESSED, f"{INPUT_DATA_CSV}_error_stan_predict_{timestamp}.csv"))
                    time.sleep(10); continue

                logging.info(f"Tymczasowy plik z przewidzianym stanem zapisany w: {path_to_temp_data_for_watcher}")

                logging.info(f"Uruchamianie {SCRIPT_MODEL_WATCHER} w tle (jeśli tak jest zaprojektowany)...")
                watcher_process = run_script(SCRIPT_MODEL_WATCHER, wait_for_completion=False)
                
                if not watcher_process: 
                    logging.error(f"Nie udało się uruchomić {SCRIPT_MODEL_WATCHER} w tle. Przerywam cykl.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(main_input_file_path, os.path.join(DIR_ALL_PROCESSED, f"{INPUT_DATA_CSV}_error_watcher_start_{timestamp}.csv"))
                    time.sleep(10); continue

                wait_time_for_watcher = 10 
                logging.info(f"Czekam {wait_time_for_watcher} sekund, aby {SCRIPT_MODEL_WATCHER} się zainicjował...")
                time.sleep(wait_time_for_watcher)

                path_for_model_watcher_final_input = os.path.join(DIR_WATCHER_INPUT, INPUT_DATA_CSV)
                if os.path.exists(path_for_model_watcher_final_input): os.remove(path_for_model_watcher_final_input)
                shutil.copy(path_to_temp_data_for_watcher, path_for_model_watcher_final_input)
                logging.info(f"Skopiowano {path_to_temp_data_for_watcher} do {path_for_model_watcher_final_input} dla {SCRIPT_MODEL_WATCHER}.")
                
                logging.info(f"Czekam na zakończenie {SCRIPT_MODEL_WATCHER} (PID: {watcher_process.pid})...")
                watcher_process.wait() 
                if watcher_process.returncode == 0:
                    logging.info(f"Skrypt {SCRIPT_MODEL_WATCHER} zakończył przetwarzanie pomyślnie.")
                else:
                    logging.error(f"Skrypt {SCRIPT_MODEL_WATCHER} zakończył przetwarzanie z kodem błędu: {watcher_process.returncode}")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(main_input_file_path, os.path.join(DIR_ALL_PROCESSED, f"{INPUT_DATA_CSV}_error_model_watcher_run_{timestamp}.csv"))
                    time.sleep(10); continue
                
                # --- Krok 6: Skopiowanie pliku wynikowego do ALL_OUT ---
                logging.info(f"--- ROZPOCZĘCIE KROKU 6: Kopiowanie wyniku z model_watcher ---")
                logging.info(f"Przeszukiwany folder wyjściowy model_watcher: '{DIR_WATCHER_OUTPUT}'")
                
                if not os.path.exists(DIR_WATCHER_OUTPUT):
                    logging.error(f"KRYTYCZNY BŁĄD KROKU 6: Folder wyjściowy dla {SCRIPT_MODEL_WATCHER} ('{DIR_WATCHER_OUTPUT}') nie istnieje! Nie można skopiować wyniku.")
                else:
                    logging.info(f"Folder '{DIR_WATCHER_OUTPUT}' istnieje.")
                    all_items_in_watcher_output = os.listdir(DIR_WATCHER_OUTPUT)
                    logging.info(f"Wszystkie elementy w '{DIR_WATCHER_OUTPUT}': {all_items_in_watcher_output}")

                    potential_output_files_paths = []
                    for item_name in all_items_in_watcher_output:
                        item_path = os.path.join(DIR_WATCHER_OUTPUT, item_name)
                        if os.path.isfile(item_path) and item_name.lower().endswith(".csv"):
                            potential_output_files_paths.append(item_path)
                    logging.info(f"Potencjalne pliki CSV wynikowe (pełne ścieżki) w '{DIR_WATCHER_OUTPUT}': {potential_output_files_paths}")

                    if not potential_output_files_paths:
                        logging.warning(f"Nie znaleziono żadnych plików CSV w folderze wynikowym '{DIR_WATCHER_OUTPUT}' skryptu {SCRIPT_MODEL_WATCHER}.")
                    else:
                        # Filtruj dalej te, które zawierają "data_out" w nazwie pliku
                        candidate_file_paths = [
                            f_path for f_path in potential_output_files_paths
                            if "data_out" in os.path.basename(f_path).lower() 
                        ]
                        # Jeśli powyższy filtr nic nie znajdzie, weź wszystkie CSV jako kandydatów
                        if not candidate_file_paths:
                            logging.warning(f"Nie znaleziono plików z 'data_out' w nazwie. Próbuję z wszystkimi plikami CSV w '{DIR_WATCHER_OUTPUT}'.")
                            candidate_file_paths = potential_output_files_paths # Wróć do wszystkich CSV

                        logging.info(f"Pliki CSV kandydaci po filtracji w '{DIR_WATCHER_OUTPUT}': {candidate_file_paths}")

                        if not candidate_file_paths:
                            logging.warning(f"Ostatecznie nie znaleziono żadnych plików CSV kandydatów w folderze '{DIR_WATCHER_OUTPUT}'.")
                        else:
                            try:
                                files_with_mtime = []
                                for f_path in candidate_file_paths:
                                    files_with_mtime.append((f_path, os.path.getmtime(f_path)))
                                logging.info(f"Kandydaci z czasami modyfikacji: {files_with_mtime}")

                                if files_with_mtime:
                                    files_with_mtime.sort(key=lambda item: item[1], reverse=True)
                                    latest_watcher_output_file_path = files_with_mtime[0][0] 
                                    latest_watcher_output_filename = os.path.basename(latest_watcher_output_file_path)
                                    
                                    logging.info(f"Najnowszy pasujący plik CSV zidentyfikowany: '{latest_watcher_output_filename}' (pełna ścieżka: '{latest_watcher_output_file_path}')")
                                    final_output_path_all = os.path.join(DIR_ALL_OUTPUT, FINAL_OUTPUT_FILENAME)
                                    logging.info(f"Próba kopiowania z '{latest_watcher_output_file_path}' do '{final_output_path_all}'")
                                    
                                    if not os.path.exists(latest_watcher_output_file_path):
                                        logging.error(f"BŁĄD KRYTYCZNY KROKU 6: Zidentyfikowany plik źródłowy '{latest_watcher_output_file_path}' nie istnieje tuż przed próbą kopiowania!")
                                    else:
                                        shutil.copy(latest_watcher_output_file_path, final_output_path_all)
                                        logging.info(f"Skopiowano finalny wynik '{latest_watcher_output_filename}' z '{latest_watcher_output_file_path}' do '{final_output_path_all}'.")

                                        if os.path.exists(final_output_path_all):
                                            logging.info(f"WERYFIKACJA: Plik '{final_output_path_all}' istnieje po skopiowaniu.")
                                        else:
                                            logging.error(f"BŁĄD KROKU 6: Plik '{final_output_path_all}' NIE istnieje po próbie skopiowania!")
                                else:
                                     logging.warning("Brak pasujących plików CSV do posortowania (po utworzeniu listy files_with_mtime).")
                            except IndexError:
                                logging.error(f"Błąd IndexError podczas próby dostępu do elementu [0] w 'candidate_files_with_mtime'. Prawdopodobnie lista była pusta.")
                            except Exception as e_copy_result:
                                logging.error(f"Błąd podczas identyfikacji lub kopiowania pliku wynikowego z '{DIR_WATCHER_OUTPUT}': {e_copy_result}")
                                import traceback; logging.error(traceback.format_exc())
                logging.info(f"--- ZAKOŃCZENIE KROKU 6 ---")
                
                timestamp_all_done = time.strftime("%Y%m%d-%H%M%S")
                if 'main_input_file_path' in locals() and os.path.exists(main_input_file_path):
                    shutil.move(main_input_file_path, os.path.join(DIR_ALL_PROCESSED, f"{INPUT_DATA_CSV}_fully_processed_{timestamp_all_done}.csv"))
                    logging.info(f"Przeniesiono oryginalny {INPUT_DATA_CSV} z {DIR_ALL_INPUT} do {DIR_ALL_PROCESSED}.")
                
                processed_one_file_successfully = True 
                logging.info(f"Pełny cykl przetwarzania dla {INPUT_DATA_CSV} zakończony.")
                break 
            
            else: 
                pass
            time.sleep(10)

        except Exception as e:
            logging.error(f"Wystąpił krytyczny błąd w głównej pętli orkiestratora: {e}")
            import traceback
            logging.error(traceback.format_exc())
            if 'main_input_file_path' in locals() and os.path.exists(main_input_file_path):
                try:
                    timestamp_err = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(main_input_file_path, os.path.join(DIR_ALL_PROCESSED, f"{INPUT_DATA_CSV}_orchestrator_error_{timestamp_err}.csv"))
                except Exception as move_e_crit:
                    logging.error(f"Nie udało się przenieść pliku po błędzie krytycznym orkiestratora: {move_e_crit}")
            
            processed_one_file_successfully = True 
            logging.info("Przerywam działanie orkiestratora z powodu błędu krytycznego.")
            break 

    if not processed_one_file_successfully:
        logging.info(f"Nie znaleziono pliku {INPUT_DATA_CSV} w {DIR_ALL_INPUT} podczas uruchomienia lub wystąpił błąd uniemożliwiający przetworzenie.")
    
    logging.info("Skrypt start_all.py zakończył działanie.")

if __name__ == "__main__":
    main_orchestrator()