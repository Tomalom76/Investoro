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
DATA_DIR = os.path.join(BASE_DIR, "DATA")
DATA_OUT_DIR = os.path.join(BASE_DIR, "DATA_OUT")
NOTEBOOK_FILENAME = "EDA_Projekt1_LGBM_v5.ipynb"  # Upewnij się, że to jest nazwa ZMODYFIKOWANEGO notebooka
NOTEBOOK_PATH = os.path.join(BASE_DIR, NOTEBOOK_FILENAME)
TRIGGER_FILE = "data.csv"
OUTPUT_FILENAME = "data_out.csv"

DEFAULT_MLFLOW_EXPERIMENT_NAME = "Predykcja_Cen_Mieszkan_Automatyczna_Watcher_v15"

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
        "df.head(", "df.sample(", "plt.figure(", "sns.heatmap(", "df.describe().T",
        "print(df_price1['Price']", "df_price1.nunique()", "unique_btype", "df.info",
        # ".plot_model(" # Zostawiamy plot_model, jeśli log_plots=True w setup
    ]
    bare_variable_displays_to_skip_if_alone = [
        "df", "df_beznull_price", "df_price1", "df_price2", 
        "unbalanced_metrics_df", "metrics_df_from_exp", # Z nowej nazwy
        "best_final_model", "df_last", "saleflats_df", "new_prices_df", 
        "merged_df2", "predicted_column"
    ]
    
    # Te frazy oznaczają bloki kodu, które chcemy pominąć, bo obsługujemy to inaczej
    critical_phrases_for_cell_commenting = [
        "df_last.to_csv('0_new_prices.csv')",
        "new_prices_df = pd.read_csv('0_new_prices.csv')",
        "saleflats_df = pd.read_csv('sale_2024_14.csv')", 
        "merged_df = pd.merge(saleflats_df, new_prices_df", 
        "merged_df.to_csv('0_new_prices_full.csv')",
        "merged_df = pd.read_csv('0_new_prices_full.csv')",
        "merged_df.to_csv('uzupelnione_mieszkania_ceny.csv')"
    ]
    
    lines_to_remove_from_cells = [ # Zawsze usuwane, bo to tylko display z PyCaret
        "exp.dataset_transformed.sample(10)" # Teraz tylko jedno `exp`
    ]

    # Zakładamy, że po sfinalizowaniu i zapisie modelu, reszta to Twoja logika CSV.
    stop_processing_notebook_after_phrase = "exp.save_model(best_final_model" 
    # Możesz też użyć: "predictions = exp.predict_model(best_final_model)"
    # jeśli chcesz, aby ta linia jeszcze się wykonała z notebooka.
    # Ale bezpieczniej jest uciąć po save_model, a predykcje zrobić w naszym bloku.

    stop_adding_cells_from_notebook = False
    mlflow_params_adjusted_in_setup = False

    for cell_index, cell in enumerate(notebook_content['cells']):
        if stop_adding_cells_from_notebook:
            logging.info(f"Pominięto komórkę {cell_index+1} i kolejne (po znalezieniu frazy stopu).")
            break

        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            if not source_lines: continue
            
            current_cell_content = "".join(source_lines)
            # Sprawdź, czy cała komórka powinna być zakomentowana
            if any(critical_phrase in current_cell_content for critical_phrase in critical_phrases_for_cell_commenting):
                # Upewnij się, że nie komentujesz komórki stopu, jeśli jest krytyczna
                effective_stop_phrase = stop_processing_notebook_after_phrase # Już używamy 'exp'
                if not (effective_stop_phrase in current_cell_content and any(cp == effective_stop_phrase for cp in critical_phrases_for_cell_commenting)):
                    logging.info(f"AUTO-MOD: Komórka {cell_index+1} zakomentowana (krytyczna fraza).")
                    python_code_segments.append(f"\n# --- KOMÓRKA {cell_index+1} ZAKOMENTOWANA ---\n" + "".join([f"# {l.strip()}\n" for l in source_lines]) + f"# --- KONIEC KOMÓRKI {cell_index+1} ---\n")
                    continue

            modified_source_lines_for_this_cell = []
            for line in source_lines:
                if not any(lineToRemove in line for lineToRemove in lines_to_remove_from_cells):
                    modified_source_lines_for_this_cell.append(line)
                else: logging.info(f"AUTO-MOD: Usunięto linię '{line.strip()}' z komórki {cell_index+1}")
            
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
            
            # Modyfikacja `setup()` jeśli to konieczne (zakładamy, że notebook MA już `log_experiment=True` i `experiment_name`)
            if "exp = setup(" in cell_content_str and not mlflow_params_adjusted_in_setup:
                # Jeśli notebook sam nie ustawia `log_experiment` lub `experiment_name` dla `exp`, dodajemy je.
                # W Twoim nowym notebooku `unbalanced_exp` (teraz `exp`) już to ma.
                # Ta logika może być teraz uproszczona, jeśli zakładamy, że notebook jest poprawnie skonfigurowany.
                if "log_experiment=True" not in cell_content_str and "log_experiment = True" not in cell_content_str :
                    if "session_id=" in cell_content_str:
                        cell_content_str = cell_content_str.replace("session_id=", "log_experiment=True, session_id=")
                    else: 
                        cell_content_str = cell_content_str.rstrip().rstrip(',')
                        if cell_content_str.endswith(')'): cell_content_str = cell_content_str[:-1] + ", log_experiment=True)"
                        else: cell_content_str += ", log_experiment=True"
                    logging.info(f"AUTO-MOD: Dodano log_experiment=True do setup() dla 'exp'")
                
                if "experiment_name=" not in cell_content_str and "MLFLOW_EXPERIMENT_NAME" not in cell_content_str:
                    exp_name_to_add_in_setup = "MLFLOW_EXPERIMENT_NAME" 
                    if "session_id=" in cell_content_str:
                        cell_content_str = cell_content_str.replace("session_id=", f"experiment_name={exp_name_to_add_in_setup}, session_id=")
                    else:
                        cell_content_str = cell_content_str.rstrip().rstrip(',')
                        if cell_content_str.endswith(')'): cell_content_str = cell_content_str[:-1] + f", experiment_name={exp_name_to_add_in_setup})"
                        else: cell_content_str += f", experiment_name={exp_name_to_add_in_setup}"
                    logging.info(f"AUTO-MOD: Dodano experiment_name={exp_name_to_add_in_setup} do setup() dla 'exp'")
                mlflow_params_adjusted_in_setup = True


            cell_content_str = cell_content_str.replace(
                "df = pd.read_csv('sale_2024_14.csv', sep=',')", 
                f"df = pd.read_csv(r'{csv_input_path}', sep=',') # ZMODYFIKOWANE"
            )
            model_save_path = os.path.join(DATA_OUT_DIR, '0_best_price_modelLGBM_auto').replace('\\', '\\\\')
            cell_content_str = cell_content_str.replace(
                "exp.save_model(best_final_model, '0-basic-model')", # Zgodnie z nowym notebookiem
                f"exp.save_model(best_final_model, r'{model_save_path}')"
            )
            cell_content_str = cell_content_str.replace( 
                "model = load_model('0-basic-model')", 
                f"model = load_model(r'{model_save_path}') # ZMODYFIKOWANE"
            )
            
            if "df_price2 = df_price1.dropna(subset=['Price'])" in cell_content_str:
                cell_content_str += """
# ----- WSTRZYKNIĘTY KOD PO UTWORZENIU df_price2 -----
if 'df_price2' in locals() and isinstance(df_price2, pd.DataFrame):
    if 'Description' in df_price2.columns:
        df_price2 = df_price2.drop(columns=['Description'], errors='ignore')
        logging.info("AUTO-MOD: Usunięto kolumnę 'Description' z df_price2.")
# ----- KONIEC WSTRZYKNIĘTEGO KODU -----"""
            
            python_code_segments.append(cell_content_str)

            if stop_processing_notebook_after_phrase in cell_content_str:
                logging.info(f"Znaleziono frazę stopu ('{stop_processing_notebook_after_phrase}') w komórce {cell_index+1}.")
                stop_adding_cells_from_notebook = True
    
    # Definicja final_processing_code (WERSJA 14.2)
    final_processing_code = f"""
# ----- POCZĄTEK FINALNEGO PRZETWARZANIA I ZAPISU (WERSJA 15) -----
# Zakładamy, że `exp` jest główną zmienną eksperymentu PyCaret
if 'df' in locals() and isinstance(df, pd.DataFrame) and \
   'exp' in locals() and \
   'best_final_model' in locals():
    
    pycaret_exp_obj_for_final_pred = locals().get('exp')
    loaded_model_for_prediction = locals().get('best_final_model')

    logging.info("Przygotowywanie danych do finalnej predykcji (oryginalny DataFrame 'df')...")
    
    df_for_prediction_final = df.copy()
    original_price_column_exists_in_df = 'Price' in df_for_prediction_final.columns
    
    if original_price_column_exists_in_df:
        df_for_prediction_final_no_price = df_for_prediction_final.drop(columns=['Price'])
        logging.info("Usunięto kolumnę 'Price' z danych przekazywanych do finalnego predict_model.")
    else:
        df_for_prediction_final_no_price = df_for_prediction_final

    logging.info("Wykonywanie predykcji na zmodyfikowanym DataFrame za pomocą 'best_final_model'...")
    try:
        predictions_on_full_df = pycaret_exp_obj_for_final_pred.predict_model(loaded_model_for_prediction, data=df_for_prediction_final_no_price)
        
        if 'prediction_label' in predictions_on_full_df.columns:
            if 'SaleId' not in predictions_on_full_df.columns:
                logging.error("Krytyczny błąd: 'SaleId' nie ma w wynikach predict_model. Sprawdź `keep_features` w setup().")
                # Zapisz to, co mamy, dla debugowania
                error_output_path = os.path.join(r"{DATA_OUT_DIR}", "error_predictions_no_saleid.csv")
                if not os.path.exists(r"{DATA_OUT_DIR}"): os.makedirs(r"{DATA_OUT_DIR}", exist_ok=True)
                predictions_on_full_df.to_csv(error_output_path, index=True)
                logging.info(f"Zapisano 'predictions_on_full_df' (bez SaleId) do {{error_output_path}}")

            else:
                final_output_df = df.copy() 
                predictions_to_add = predictions_on_full_df[['SaleId', 'prediction_label']].copy()
                predictions_to_add.rename(columns={{'prediction_label': 'PredictedPriceByModel'}}, inplace=True)
                try:
                    final_output_df['SaleId'] = final_output_df['SaleId'].astype(str)
                    predictions_to_add['SaleId'] = predictions_to_add['SaleId'].astype(str)
                except Exception as e_type_conv_saleid_final: 
                    logging.warning(f"Problem podczas próby konwersji 'SaleId' na string: {{e_type_conv_saleid_final}}")

                final_output_df = pd.merge(final_output_df, predictions_to_add, on='SaleId', how='left')
                
                if 'PredictedPriceByModel' in final_output_df.columns:
                    final_output_df['PredictedPriceByModel_numeric'] = pd.to_numeric(final_output_df['PredictedPriceByModel'], errors='coerce').round(0)
                    final_output_df['PredictedPriceByModel'] = final_output_df['PredictedPriceByModel_numeric'].apply(
                        lambda x: f'{{x:,.0f}}' if pd.notna(x) else '' 
                    )
                    final_output_df.drop(columns=['PredictedPriceByModel_numeric'], inplace=True)
                    logging.info("Sformatowano kolumnę 'PredictedPriceByModel'.")

                cols = final_output_df.columns.tolist()
                if 'Price' in cols and 'PredictedPriceByModel' in cols:
                    try:
                        price_idx = cols.index('Price')
                        predicted_col_data = final_output_df.pop('PredictedPriceByModel')
                        final_output_df.insert(price_idx + 1, 'PredictedPriceByModel', predicted_col_data)
                        if 'SaleId' in final_output_df.columns:
                            id_col_temp = final_output_df.pop('SaleId')
                            final_output_df.insert(0, 'SaleId', id_col_temp)
                        logging.info("Przesunięto kolumnę 'PredictedPriceByModel' obok 'Price'.")
                    except ValueError as ve_reorder: 
                        logging.error(f"Błąd podczas przesuwania kolumny (ValueError): {{ve_reorder}}")
                    except Exception as e_reorder: 
                         logging.error(f"Inny błąd podczas przesuwania kolumny: {{e_reorder}}")
                elif 'PredictedPriceByModel' in cols:
                    logging.info("Kolumna 'Price' nie istnieje, 'PredictedPriceByModel' pozostaje na końcu.")

                try:
                    output_path_script_var = r'{final_output_path}'
                    final_output_df.to_csv(output_path_script_var, index=False, sep=',')
                    logging.info("Finalny DataFrame (oryginalne dane + predykcje) zapisany do: %s", output_path_script_var)
                    if mlflow.active_run():
                       mlflow.log_artifact(output_path_script_var, "output_data_final")
                       logging.info(f"Zalogowano {{output_path_script_var}} jako artefakt MLflow.")
                except Exception as e_save:
                    logging.error("Błąd podczas zapisywania finalnego DataFrame: %s", e_save)
                    import traceback; logging.error(traceback.format_exc())
        else:
            logging.error("Kolumna 'prediction_label' nie została znaleziona w wynikach predict_model na pełnym 'df'.")
    except Exception as e_final_pred_save:
        logging.error("Błąd podczas finalnej predykcji (predict_model): %s", e_final_pred_save)
        import traceback; logging.error(traceback.format_exc())
else:
    logging.warning("Nie znaleziono wymaganych obiektów ('df', 'exp', 'best_final_model') do przeprowadzenia finalnego przetwarzania.")
# ----- KONIEC FINALNEGO PRZETWARZANIA I ZAPISU (WERSJA 15) -----
"""

    mlflow_init_code_lines = [
        "import mlflow", "import time", 
        f"mlflow.set_tracking_uri(\"http://localhost:5000\")", # Zgodnie z notebookiem
        f"notebook_mlflow_exp_name = None",
        f"if 'MLFLOW_EXPERIMENT_NAME' in locals(): notebook_mlflow_exp_name = MLFLOW_EXPERIMENT_NAME",
        f"current_experiment_name = notebook_mlflow_exp_name if notebook_mlflow_exp_name else '{DEFAULT_MLFLOW_EXPERIMENT_NAME}'",
        f"mlflow.set_experiment(current_experiment_name)",
        f"logging.info(f'MLflow: Ustawiono eksperyment na: {{current_experiment_name}}')"
    ]
    
    code_from_notebook_processed = "\n".join(python_code_segments)
    combined_code_for_run = code_from_notebook_processed + "\n" + final_processing_code

    script_body_with_run = "with mlflow.start_run(run_name=f'Automated_Run_{{time.strftime(\"%Y%m%d-%H%M%S\")}}') as run:\n"
    script_body_with_run += "    # Logowanie parametrów wejściowych\n"
    script_body_with_run += f"    mlflow.log_param('input_csv_path', r'{csv_input_path}')\n"
    script_body_with_run += f"    mlflow.log_param('notebook_path', r'{NOTEBOOK_PATH}')\n"
    script_body_with_run += "    if 'MLFLOW_TAGS' in locals() and isinstance(MLFLOW_TAGS, dict):\n"
    script_body_with_run += "        mlflow.set_tags(MLFLOW_TAGS)\n"
    
    for line in combined_code_for_run.splitlines():
        script_body_with_run += f"    {line}\n"
    
    full_script = "\n".join(mlflow_init_code_lines) + "\n\n" + script_body_with_run
    
    debug_script_path = os.path.join(BASE_DIR, "debug_generated_script.py")
    try:
        with open(debug_script_path, "w", encoding="utf-8") as f_debug:
            f_debug.write(full_script)
        logging.info(f"Pełny wygenerowany skrypt zapisany do: {debug_script_path}")
    except Exception as e_debug_save:
        logging.error(f"Nie udało się zapisać debug_generated_script.py: {e_debug_save}")

    execution_globals = {
        "pd": pd, "logging": logging, "__name__": "__main__", 
        "os": os, "sys": sys, "DATA_DIR": DATA_DIR, "DATA_OUT_DIR": DATA_OUT_DIR,
        "csv_input_path": csv_input_path, 
        "final_output_path": final_output_path,
        "mlflow": mlflow, "time": time, "NOTEBOOK_PATH": NOTEBOOK_PATH, 
        "DEFAULT_MLFLOW_EXPERIMENT_NAME": DEFAULT_MLFLOW_EXPERIMENT_NAME,
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
            output_file_path = os.path.join(DATA_OUT_DIR, OUTPUT_FILENAME)
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