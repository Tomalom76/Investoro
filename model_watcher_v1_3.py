import time
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import sys
import mlflow
import joblib

from pycaret.regression import (
    setup, pull, compare_models, plot_model,
    load_model, tune_model, finalize_model,
    save_model, predict_model, get_config
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
DATA_OUT_DIR = os.path.join(BASE_DIR, "DATA_OUT")
NOTEBOOK_FILENAME = "EDA_Projekt1_v6_3FULL.ipynb"
NOTEBOOK_PATH = os.path.join(BASE_DIR, NOTEBOOK_FILENAME)
TRIGGER_FILE = "data.csv"
OUTPUT_FILENAME = "data_out.csv"

MLFLOW_EXPERIMENT_NAME = 'Investoro_Ceny'
MLFLOW_TAGS = {'data': 'Investoro_ceny', 'library': 'pycaret'}
DEFAULT_MLFLOW_EXPERIMENT_NAME = "Predykcja_Cen_Mieszkan_Automatyczna_Watcher_v20" # Zwiększamy wersję

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def create_dirs_if_not_exist():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    logging.info(f"Katalogi DATA ('{DATA_DIR}') i DATA_OUT ('{DATA_OUT_DIR}') gotowe.")

def extract_and_execute_notebook_logic(csv_input_path, final_output_path):
    logging.info(f"Przetwarzanie notebooka: {NOTEBOOK_PATH}")
    logging.info(f"Plik wejściowy CSV: {csv_input_path}")
    logging.info(f"Docelowy plik wyjściowy: {final_output_path}")

    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)
    except FileNotFoundError:
        logging.error(f"Nie znaleziono pliku notebooka: {NOTEBOOK_PATH}")
        return
    except json.JSONDecodeError:
        logging.error(f"Błąd parsowania pliku notebooka: {NOTEBOOK_PATH}.")
        return

    python_code_segments = []
    phrases_to_skip_cell_if_not_ml_or_setup = [
        "df_original.head(", "df_original.sample(", "df_cleaned.head(",
        "df_original.info()", "plt.figure(", "sns.heatmap(", "df_cleaned.describe().T",
        "df_cleaned.nunique()", "df_cleaned.isnull().sum()", "compare_metrics_df",
        "test_set_metrics_after_tuning", "holdout_final_metrics",
        "reg_exp.X_train_transformed.info()", "reg_exp.X_train_transformed.head()",
        "reg_exp.dataset.head()", "reg_exp.dataset_transformed.head()",
        "df_transformed_pycaret.plot.scatter", "df_transformed.plot.scatter"
    ]
    bare_variable_displays_to_skip_if_alone = [
        "df_original", "df_cleaned", "train_df_processed", "holdout_df_processed",
        "correlation_matrix", "unique_btype", "best_model_from_compare", 
        "tuned_best_model", "best_final_model", "final_holdout_predictions_df", 
        "df_last_to_save_compare", "holdout_df_with_predictions",
        "original_predictions", "df_original_with_all_predictions"
    ]
    critical_phrases_for_cell_commenting = [
        "df_last_to_save_compare.to_csv('0_new_prices_compare.csv'",
        "holdout_df_with_predictions.to_csv('full_holdout_with_predictions.csv'",
        "df_original_with_all_predictions.to_csv('sale_2024_0_predict.csv'" 
    ]
    lines_to_remove_from_cells = []
    stop_processing_notebook_after_phrase = "save_model(best_final_model, '0_full-basic-model')"
    stop_adding_cells_from_notebook = False
    mlflow_params_adjusted_in_setup = False

    for cell_index, cell in enumerate(notebook_content['cells']):
        if stop_adding_cells_from_notebook:
            logging.info(f"Pominięto komórkę {cell_index+1} i kolejne (po znalezieniu frazy stopu).")
            break
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            if not source_lines: continue
            current_cell_content_as_string = "".join(source_lines)
            if any(critical_phrase in current_cell_content_as_string for critical_phrase in critical_phrases_for_cell_commenting):
                if not (stop_processing_notebook_after_phrase in current_cell_content_as_string and \
                        any(cp == stop_processing_notebook_after_phrase for cp in critical_phrases_for_cell_commenting)):
                    logging.info(f"AUTO-MOD: Komórka {cell_index+1} zakomentowana (krytyczna fraza).")
                    python_code_segments.append(f"\n# --- KOMÓRKA {cell_index+1} ZAKOMENTOWANA ---\n" + "".join([f"# {l.strip()}\n" for l in source_lines]) + f"# --- KONIEC KOMÓRKI {cell_index+1} ---\n")
                    continue
            modified_source_lines_for_this_cell = []
            for line in source_lines:
                if not any(lineToRemove in line for lineToRemove in lines_to_remove_from_cells):
                    if line.strip().startswith("display("):
                        logging.info(f"AUTO-MOD: Usunięto linię display: {line.strip()} z komórki {cell_index+1}")
                    else:
                        modified_source_lines_for_this_cell.append(line)
                else:
                    logging.info(f"AUTO-MOD: Usunięto linię '{line.strip()}' z komórki {cell_index+1}")
            if not modified_source_lines_for_this_cell: continue
            cell_content_str = "".join(modified_source_lines_for_this_cell)
            is_eda_or_display_only = False
            if any(skip_phrase in cell_content_str for skip_phrase in phrases_to_skip_cell_if_not_ml_or_setup):
                if not any(op in cell_content_str for op in ["setup(", "compare_models(", "tune_model(", "finalize_model(", "save_model(", "load_model(", "predict_model(", "pull()", "mlflow."]):
                    is_eda_or_display_only = True
            non_empty_lines_for_bare_display = [l.strip() for l in modified_source_lines_for_this_cell if l.strip() and not l.strip().startswith('#')]
            if len(non_empty_lines_for_bare_display) == 1:
                last_line_stripped_for_bare_display = non_empty_lines_for_bare_display[0]
                if last_line_stripped_for_bare_display in bare_variable_displays_to_skip_if_alone and \
                   '=' not in last_line_stripped_for_bare_display and 'return ' not in last_line_stripped_for_bare_display and \
                   '(' not in last_line_stripped_for_bare_display and '.' not in last_line_stripped_for_bare_display:
                    is_eda_or_display_only = True
            if is_eda_or_display_only:
                logging.info(f"Pominięto komórkę EDA/plot/print/display (ok. {cell_index+1}): {cell_content_str.strip()[:80]}...")
                continue
            if "reg_exp = setup(" in cell_content_str and not mlflow_params_adjusted_in_setup:
                if "log_experiment=True" not in cell_content_str and "log_experiment = True" not in cell_content_str :
                    cell_content_str = cell_content_str.replace("target='Price',", "target='Price', log_experiment=True,", 1) if "target='Price'," in cell_content_str else cell_content_str.replace(")\n", ", log_experiment=True)\n", 1)
                    logging.info(f"AUTO-MOD: Dodano log_experiment=True do setup()")
                if "experiment_name=" not in cell_content_str and "MLFLOW_EXPERIMENT_NAME" not in cell_content_str:
                    cell_content_str = cell_content_str.replace("log_experiment=True,", f"log_experiment=True, experiment_name=MLFLOW_EXPERIMENT_NAME,", 1) if "log_experiment=True," in cell_content_str else cell_content_str.replace(")\n", f", experiment_name=MLFLOW_EXPERIMENT_NAME)\n", 1)
                    logging.info(f"AUTO-MOD: Dodano experiment_name=MLFLOW_EXPERIMENT_NAME do setup()")
                mlflow_params_adjusted_in_setup = True
            
            csv_read_pattern_original = "df_original  = pd.read_csv('data.csv', sep=',')" 
            if csv_read_pattern_original in cell_content_str:
                cell_content_str = cell_content_str.replace(
                    csv_read_pattern_original,
                    f"df_original  = pd.read_csv(r'{csv_input_path}', sep=',') # ZMODYFIKOWANE PRZEZ SKRYPT"
                )
                logging.info(f"AUTO-MOD: Zmodyfikowano wczytywanie df_original w komórce {cell_index+1}")

            string_do_znalezienia_save_model = "save_model(best_final_model, '0_full-basic-model')"
            if string_do_znalezienia_save_model in cell_content_str:
                model_save_path_script = os.path.join(DATA_OUT_DIR, '0_best_LGBM_model_via_script_watch').replace('\\', '\\\\')
                cell_content_str = cell_content_str.replace(
                    string_do_znalezienia_save_model,
                    f"save_model(best_final_model, r'{model_save_path_script}') # ZMODYFIKOWANE PRZEZ SKRYPT"
                )
                logging.info(f"AUTO-MOD: Zmodyfikowano linię save_model w komórce {cell_index+1}")
            
            python_code_segments.append(cell_content_str)
            
            current_stop_phrase = f"save_model(best_final_model, r'{model_save_path_script}')" if string_do_znalezienia_save_model in current_cell_content_as_string else stop_processing_notebook_after_phrase
            if current_stop_phrase in cell_content_str:
                logging.info(f"Znaleziono frazę stopu ('{current_stop_phrase}') w komórce {cell_index+1}.")
                stop_adding_cells_from_notebook = True

    final_processing_code = f"""
# ----- POCZĄTEK FINALNEGO PRZETWARZANIA I ZAPISU (dla notebooka v6.3) -----

# === DODANY BLOK: Zapis TfidfVectorizer ===
# Ten kod wykona się wewnątrz exec(), po wykonaniu kodu z notebooka
if 'location_vectorizer' in locals() and 'DATA_OUT_DIR' in locals() and 'os' in locals() and 'joblib' in locals() and 'mlflow' in locals():
    try:
        # Używamy stałej nazwy pliku, tej samej, której oczekuje skrypt predykcyjny
        vectorizer_save_path = os.path.join(DATA_OUT_DIR, 'location_vectorizer.joblib') 
        joblib.dump(location_vectorizer, vectorizer_save_path)
        logging.info(f"ZAPISANO (z watcher.py): TfidfVectorizer do: {{vectorizer_save_path}}")
        if mlflow.active_run():
            mlflow.log_artifact(vectorizer_save_path, "location_vectorizer_artifact") # Nazwa artefaktu w MLflow
            logging.info(f"ZAPISANO (z watcher.py): TfidfVectorizer jako artefakt MLflow.")
    except Exception as e_vec_save:
        logging.error(f"BŁĄD (z watcher.py): Nie udało się zapisać TfidfVectorizer: {{e_vec_save}}")
else:
    missing_vars_for_vec_save = [var for var in ['location_vectorizer', 'DATA_OUT_DIR', 'os', 'joblib', 'mlflow'] if var not in locals()]
    logging.warning(f"Nie znaleziono wymaganych zmiennych/modułów do zapisu TfidfVectorizer w watcher.py. Brakujące: {{missing_vars_for_vec_save}}")
# === KONIEC BLOKU: Zapis TfidfVectorizer ===

if 'reg_exp' in locals() and \\
   'best_final_model' in locals() and \\
   'df_original' in locals() and isinstance(df_original, pd.DataFrame) and \\
   'location_vectorizer' in locals() and \\
   'convert_column_types' in locals() :

    logging.info("Przygotowywanie df_original do finalnej predykcji...")
    
    df_to_predict_on = df_original.copy()

    df_to_predict_on.dropna(subset=['Area', 'Price', 'Location'], inplace=True) # DODANO 'Price' TUTAJ
    logging.info(f"Rozmiar df_to_predict_on po usunięciu NaN z Area, Price, Location: {{df_to_predict_on.shape}}")

    if not df_to_predict_on.empty:
        Q1_area_orig = df_to_predict_on["Area"].quantile(0.25)
        Q3_area_orig = df_to_predict_on["Area"].quantile(0.75)
        IQR_area_orig = Q3_area_orig - Q1_area_orig
        lower_bound_area_orig = Q1_area_orig - 1.5 * IQR_area_orig
        upper_bound_area_orig = Q3_area_orig + 1.5 * IQR_area_orig
        df_to_predict_on = df_to_predict_on[
            ~((df_to_predict_on["Area"] < lower_bound_area_orig) | (df_to_predict_on["Area"] > upper_bound_area_orig))
        ]
        logging.info(f"Rozmiar df_to_predict_on po usunięciu outlierów z Area: {{df_to_predict_on.shape}}")
    
    if not df_to_predict_on.empty:
        df_to_predict_on = convert_column_types(df_to_predict_on)
        logging.info("Konwersja typów danych w df_to_predict_on zakończona.")

    df_original_processed_for_script = pd.DataFrame() 
    if not df_to_predict_on.empty:
        df_to_predict_on['Location_Clean'] = df_to_predict_on['Location'].fillna('').astype(str)
        original_location_tfidf_features = location_vectorizer.transform(df_to_predict_on['Location_Clean'])
        
        try:
            feature_names_loc_script = location_vectorizer.get_feature_names_out()
        except AttributeError:
            feature_names_loc_script = location_vectorizer.get_feature_names_()

        original_location_tfidf_df_script = pd.DataFrame(
            original_location_tfidf_features.toarray(),
            columns=['loc_tfidf_' + name for name in feature_names_loc_script],
            index=df_to_predict_on.index
        )
        df_original_processed_for_script = pd.concat(
            [df_to_predict_on.drop(columns=['Location', 'Location_Clean'], errors='ignore'), original_location_tfidf_df_script],
            axis=1
        )
        logging.info(f"Utworzono {{original_location_tfidf_df_script.shape[1]}} cech TF-IDF dla df_original_processed_for_script.")
        logging.info(f"Rozmiar df_original_processed_for_script: {{df_original_processed_for_script.shape}}")
    
    if not df_original_processed_for_script.empty:
        logging.info("Wykonywanie predykcji na całym przetworzonym df_original...")
        data_for_final_pred_script = df_original_processed_for_script.copy()
        
        actual_prices_for_metric_calc = None
        if 'Price' in df_original_processed_for_script.columns:
            actual_prices_for_metric_calc = df_original_processed_for_script['Price'].copy() 
        
        if 'Price' in data_for_final_pred_script.columns:
            data_for_final_pred_script = data_for_final_pred_script.drop(columns=['Price'])
            logging.info("Usunięto kolumnę 'Price' z danych przekazywanych do finalnego predict_model.")
        
        predictions_on_original_df = predict_model(best_final_model, data=data_for_final_pred_script)
        
        if 'prediction_label' in predictions_on_original_df.columns:
            if actual_prices_for_metric_calc is not None:
                metrics_df = pd.DataFrame({{
                    'actual': actual_prices_for_metric_calc,
                    'predicted': predictions_on_original_df.loc[actual_prices_for_metric_calc.index, 'prediction_label']
                }})
                metrics_df.dropna(inplace=True)

                if not metrics_df.empty:
                    r2_orig = r2_score(metrics_df['actual'], metrics_df['predicted'])
                    mae_orig = mean_absolute_error(metrics_df['actual'], metrics_df['predicted'])
                    logging.info(f"Metryki na przetworzonym df_original (po usunięciu NaN w Price i predykcjach): R2 = {{r2_orig:.4f}}, MAE = {{mae_orig:.2f}}")
                    if mlflow.active_run():
                        mlflow.log_metric("df_original_final_R2", r2_orig)
                        mlflow.log_metric("df_original_final_MAE", mae_orig)
                else:
                    logging.warning("Nie można obliczyć metryk - brak danych po usunięciu NaN z cen rzeczywistych lub predykcji.")
            
            output_df = df_original.copy() 
            
            predictions_to_add_script = pd.DataFrame(index=df_original_processed_for_script.index)
            if 'SaleId' in df_original_processed_for_script.columns:
                 predictions_to_add_script['SaleId'] = df_original_processed_for_script['SaleId']
            else: 
                 if df_original_processed_for_script.index.name == 'SaleId':
                     predictions_to_add_script['SaleId'] = df_original_processed_for_script.index.to_series()
                 else:
                     logging.warning("Nie można jednoznacznie zidentyfikować 'SaleId' w df_original_processed_for_script. Predykcje mogą nie zostać poprawnie połączone.")

            predictions_to_add_script['PredictedPrice_LGBM'] = predictions_on_original_df.loc[predictions_to_add_script.index, 'prediction_label']

            if 'SaleId' in predictions_to_add_script.columns and 'SaleId' in output_df.columns:
                try:
                    output_df['SaleId'] = output_df['SaleId'].astype(str)
                    predictions_to_add_script['SaleId'] = predictions_to_add_script['SaleId'].astype(str)
                    predictions_to_add_script.drop_duplicates(subset=['SaleId'], keep='first', inplace=True)
                    output_df = pd.merge(output_df, predictions_to_add_script, on='SaleId', how='left')
                except Exception as e_merge:
                    logging.error(f"Błąd podczas łączenia predykcji z oryginalnym DataFrame: {{e_merge}}")
            else:
                 logging.error("Nie można połączyć predykcji - brak kolumny 'SaleId' w jednym z DataFrame'ów.")

            if 'PredictedPrice_LGBM' in output_df.columns:
                output_df['PredictedPrice_LGBM_num'] = pd.to_numeric(output_df['PredictedPrice_LGBM'], errors='coerce').round(0)
                output_df['PredictedPrice_LGBM_formatted'] = output_df['PredictedPrice_LGBM_num'].apply(
                    lambda x: f'{{x:,.0f}}' if pd.notna(x) else '' 
                )
                output_df['PredictedPrice_LGBM'] = output_df['PredictedPrice_LGBM_formatted']
                output_df.drop(columns=['PredictedPrice_LGBM_num', 'PredictedPrice_LGBM_formatted'], inplace=True, errors='ignore')

                cols_script = list(output_df.columns)
                if 'Price' in cols_script and 'PredictedPrice_LGBM' in cols_script:
                    try:
                        price_idx_script = cols_script.index('Price')
                        if 'PredictedPrice_LGBM' in cols_script: 
                            cols_script.remove('PredictedPrice_LGBM')
                        cols_script.insert(price_idx_script + 1, 'PredictedPrice_LGBM')
                        output_df = output_df[cols_script]
                        logging.info("Przesunięto kolumnę 'PredictedPrice_LGBM' obok 'Price'.")
                    except ValueError as ve_reorder: 
                        logging.error(f"Błąd podczas przesuwania kolumny (ValueError): {{ve_reorder}}")
                    except Exception as e_reorder: 
                         logging.error(f"Inny błąd podczas przesuwania kolumny: {{e_reorder}}")
                elif 'PredictedPrice_LGBM' in cols_script:
                    logging.info("Kolumna 'Price' nie istnieje, 'PredictedPrice_LGBM' pozostaje na końcu.")
                else:
                    logging.warning("Nie znaleziono kolumny 'PredictedPrice_LGBM' do przesunięcia.")
            else:
                 logging.warning("Kolumna 'PredictedPrice_LGBM' nie została dodana do output_df po merge.")

            try:
                output_path_final_script = r'{final_output_path}'
                output_df.to_csv(output_path_final_script, index=False, sep=',')
                logging.info("Finalny DataFrame (oryginalne dane + predykcje) zapisany do: %s", output_path_final_script)
                if mlflow.active_run():
                   mlflow.log_artifact(output_path_final_script, "output_data_on_full_original")
                   logging.info(f"Zalogowano {{output_path_final_script}} jako artefakt MLflow.")
            except Exception as e_save_final:
                logging.error("Błąd podczas zapisywania finalnego DataFrame: %s", e_save_final)
                import traceback; logging.error(traceback.format_exc())
        else:
            logging.error("Kolumna 'prediction_label' nie została znaleziona w wynikach predict_model na df_original_processed_for_script.")
    else:
        logging.warning("DataFrame df_original_processed_for_script jest pusty po przetworzeniu, nie wykonano finalnej predykcji.")
else:
    logging.warning("Nie znaleziono wymaganych obiektów ('reg_exp', 'best_final_model', 'df_original', 'location_vectorizer', 'convert_column_types') do przeprowadzenia finalnego przetwarzania.")

# ----- KONIEC FINALNEGO PRZETWARZANIA I ZAPISU (dla notebooka v6.3) -----
"""
    mlflow_init_code_lines = [
        "import mlflow", "import time", "import os", "import logging",
        f"notebook_mlflow_uri_check = locals().get('MLFLOW_TRACKING_URI', 'http://localhost:5000')",
        f"current_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()",
        f"script_local_mlruns = os.path.join(current_script_dir, 'mlruns_SCRIPT_EXEC')",
        f"if not os.path.exists(script_local_mlruns): os.makedirs(script_local_mlruns, exist_ok=True)",
        f"script_tracking_uri = f\"file:///{{os.path.abspath(script_local_mlruns).replace(os.sep, '/')}}\"",
        f"use_notebook_uri = False",
        f"if notebook_mlflow_uri_check and not notebook_mlflow_uri_check.startswith('file:///C:/Users/Ai/Desktop/Tomek/Investoro/mlruns_DIRECT_LOCAL_TEST'):", # Ścieżka do dostosowania
        f"    use_notebook_uri = True",
        f"effective_tracking_uri = notebook_mlflow_uri_check if use_notebook_uri else script_tracking_uri",
        f"mlflow.set_tracking_uri(effective_tracking_uri)",
        f"notebook_mlflow_exp_name = None",
        f"if 'MLFLOW_EXPERIMENT_NAME' in locals(): notebook_mlflow_exp_name = MLFLOW_EXPERIMENT_NAME",
        f"current_experiment_name = notebook_mlflow_exp_name if notebook_mlflow_exp_name else '{DEFAULT_MLFLOW_EXPERIMENT_NAME}'",
        f"experiment = mlflow.get_experiment_by_name(current_experiment_name)",
        f"if experiment is None: experiment_id = mlflow.create_experiment(current_experiment_name)",
        f"else: experiment_id = experiment.experiment_id",
        f"mlflow.set_experiment(experiment_name=current_experiment_name)",
        f"logging.info(f'MLflow: Ustawiono URI na: {{mlflow.get_tracking_uri()}}')",
        f"logging.info(f'MLflow: Ustawiono eksperyment na: {{current_experiment_name}}')"
    ]
    code_from_notebook_processed = "\n".join(python_code_segments)
    convert_column_types_func_def = """
def convert_column_types(df_to_convert):
    df_copy = df_to_convert.copy()
    str_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']
    for col in str_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str)
    if 'BuiltYear' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['BuiltYear']):
         df_copy['BuiltYear'] = pd.to_datetime(df_copy['BuiltYear'], format='%Y', errors='coerce')
    return df_copy
"""
    combined_code_for_run = convert_column_types_func_def + "\n" + code_from_notebook_processed + "\n" + final_processing_code
    script_body_with_run = "with mlflow.start_run(run_name=f'Automated_Run_NB63_{{time.strftime(\"%Y%m%d-%H%M%S\")}}') as run:\n"
    script_body_with_run += "    # Logowanie parametrów wejściowych\n"
    script_body_with_run += f"    mlflow.log_param('input_csv_path', r'{csv_input_path}')\n"
    script_body_with_run += f"    mlflow.log_param('notebook_path', r'{NOTEBOOK_PATH}')\n"
    script_body_with_run += "    mlflow_tags_from_notebook_exec = {}\n" 
    script_body_with_run += "    if 'MLFLOW_TAGS' in locals() and isinstance(MLFLOW_TAGS, dict):\n"
    script_body_with_run += "        mlflow_tags_from_notebook_exec = MLFLOW_TAGS\n" 
    script_body_with_run += "    mlflow.set_tags(mlflow_tags_from_notebook_exec)\n" 
    for line in combined_code_for_run.splitlines():
        script_body_with_run += f"    {line}\n"
    global_script_imports = [
        "import pandas as pd", "import numpy as np", "import matplotlib.pyplot as plt",
        "import seaborn as sns", "import mlflow", "import time", "import os",
        "import logging", "import sys",
        "from pycaret.regression import (setup, pull, compare_models, plot_model, "
        "load_model, tune_model, finalize_model, save_model, predict_model, get_config)",
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "from sklearn.metrics import r2_score, mean_absolute_error"
    ]
    full_script = "\n".join(global_script_imports) + "\n\n" + "\n".join(mlflow_init_code_lines) + "\n\n" + script_body_with_run
    debug_script_path = os.path.join(BASE_DIR, "debug_generated_script_v6_3.py")
    try:
        with open(debug_script_path, "w", encoding="utf-8") as f_debug:
            f_debug.write(full_script)
        logging.info(f"Pełny wygenerowany skrypt zapisany do: {debug_script_path}")
    except Exception as e_debug_save:
        logging.error(f"Nie udało się zapisać debug_generated_script.py: {e_debug_save}")
    execution_globals = {
        "pd": pd, "np": np, "plt": plt, "sns": sns, "mlflow": mlflow, "time": time,
        "os": os, "logging": logging, "sys": sys, "setup": setup, "pull": pull,
        "compare_models": compare_models, "plot_model": plot_model, "load_model": load_model,
        "tune_model": tune_model, "finalize_model": finalize_model, "save_model": save_model,
        "predict_model": predict_model, "get_config": get_config,
        "TfidfVectorizer": TfidfVectorizer, "r2_score": r2_score,
        "mean_absolute_error": mean_absolute_error, "__name__": "__main__",
        "DATA_DIR": DATA_DIR, "DATA_OUT_DIR": DATA_OUT_DIR,
        "csv_input_path": csv_input_path, "final_output_path": final_output_path,
        "NOTEBOOK_PATH": NOTEBOOK_PATH,
        "joblib": joblib, 
        "DEFAULT_MLFLOW_EXPERIMENT_NAME": DEFAULT_MLFLOW_EXPERIMENT_NAME,
        "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME, 
        "MLFLOW_TAGS": MLFLOW_TAGS,
    }
    logging.info("--- Rozpoczęcie wykonywania logiki z notebooka (z MLflow) ---")
    try:
        exec(full_script, execution_globals)
        logging.info("--- Zakończono wykonywanie logiki z notebooka (z MLflow) ---")
    except Exception as e:
        logging.error(f"Błąd podczas wykonywania kodu z notebooka: {e}")
        import traceback
        logging.error(traceback.format_exc())

class ModelTrainingHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and os.path.basename(event.src_path) == TRIGGER_FILE:
            logging.info(f"Wykryto nowy plik: {event.src_path}")
            time.sleep(5)
            csv_file_to_process = event.src_path
            if not os.path.exists(csv_file_to_process) or os.path.getsize(csv_file_to_process) == 0:
                logging.warning(f"Plik {csv_file_to_process} nie istnieje lub jest pusty. Pomijanie.")
                return
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_output_filename, ext_output_filename = os.path.splitext(OUTPUT_FILENAME)
            unique_output_filename = f"{base_output_filename}_{timestamp}{ext_output_filename}"
            output_file_path = os.path.join(DATA_OUT_DIR, unique_output_filename)
            extract_and_execute_notebook_logic(csv_file_to_process, output_file_path)

if __name__ == "__main__":
    create_dirs_if_not_exist()
    event_handler = ModelTrainingHandler()
    observer = Observer()
    observer.schedule(event_handler, DATA_DIR, recursive=False)
    logging.info(f"Nasłuchiwanie na zmiany w katalogu: {DATA_DIR}, oczekiwany plik: {TRIGGER_FILE}")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Zatrzymano nasłuchiwanie.")
    observer.join()