import time
import os
import json
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import sys

# --- Konfiguracja ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
DATA_OUT_DIR = os.path.join(BASE_DIR, "DATA_OUT")
NOTEBOOK_FILENAME = "EDA_Projekt1_LGBM_v5.ipynb" 
NOTEBOOK_PATH = os.path.join(BASE_DIR, NOTEBOOK_FILENAME)
TRIGGER_FILE = "data.csv"
OUTPUT_FILENAME = "data_out.csv"

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
    
    phrases_to_skip_cell_if_not_ml = [
        "df.head(", "df.sample(", 
        "plt.figure(", "sns.heatmap(", "df.describe().T",
        "print(df_price1['Price']", "df_price1.nunique()", "unique_btype", "df.info",
        ".plot_model(" 
    ]
    bare_variable_displays_to_skip_if_alone = [
        "df", "df_beznull_price", "df_price1", "df_price2", "unbalanced_metrics_df", 
        "best_final_model", "df_last", "saleflats_df", "new_prices_df", 
        "merged_df2", "predicted_column", "merged_df", "predictions"
    ]
    
    lines_to_remove_from_cells = [
        "exp.dataset_transformed.sample(10)",
        "unbalanced_exp.dataset_transformed.sample(10)"
    ]
    
    # Fraza, po której znalezieniu przestajemy przetwarzać komórki z notebooka.
    # Zakładamy, że po zapisie modelu reszta to już Twoje operacje na plikach.
    # Lub po ostatnim `predict_model`, które definiuje `predictions` używane w Twojej starej logice.
    # Bezpieczniej jest użyć `exp.save_model`.
    stop_processing_notebook_after_phrase = "exp.save_model(best_final_model" 

    exp_setup_processed = False
    first_setup_variable_name = "exp" 
    stop_adding_cells_from_notebook = False

    for cell_index, cell in enumerate(notebook_content['cells']):
        if stop_adding_cells_from_notebook:
            logging.info(f"Pominięto komórkę {cell_index+1} i kolejne (po znalezieniu frazy stopu w poprzedniej).")
            break

        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            if not source_lines:
                continue
            
            modified_source_lines_for_this_cell = []
            for line in source_lines:
                line_was_removed = False
                for lineToRemove in lines_to_remove_from_cells:
                    if lineToRemove in line:
                        logging.info(f"AUTO-MOD: Usunięto linię '{line.strip()}' z komórki {cell_index+1}")
                        line_was_removed = True
                        break
                if not line_was_removed:
                    modified_source_lines_for_this_cell.append(line)
            
            if not modified_source_lines_for_this_cell:
                continue

            cell_content_str = "".join(modified_source_lines_for_this_cell)
            
            is_eda_or_display_only = False
            if any(skip_phrase in cell_content_str for skip_phrase in phrases_to_skip_cell_if_not_ml):
                if not any(op in cell_content_str for op in ["setup(", "compare_models(", "tune_model(", "finalize_model(", "save_model(", "load_model(", "predict_model(", "pull()"]):
                    is_eda_or_display_only = True
            
            non_empty_lines_for_bare_display = [l.strip() for l in modified_source_lines_for_this_cell if l.strip() and not l.strip().startswith('#')]
            if len(non_empty_lines_for_bare_display) == 1:
                last_line_stripped_for_bare_display = non_empty_lines_for_bare_display[0]
                if last_line_stripped_for_bare_display in bare_variable_displays_to_skip_if_alone and \
                   '=' not in last_line_stripped_for_bare_display and \
                   'return ' not in last_line_stripped_for_bare_display and \
                   '(' not in last_line_stripped_for_bare_display and \
                   '.' not in last_line_stripped_for_bare_display:
                    is_eda_or_display_only = True
            
            if is_eda_or_display_only:
                logging.info(f"Pominięto komórkę EDA/plot/print/display (ok. {cell_index+1}): {cell_content_str.strip()[:80]}...")
                continue

            if "setup(" in cell_content_str:
                if not exp_setup_processed:
                    if "unbalanced_exp =" in cell_content_str:
                        first_setup_variable_name = "unbalanced_exp"
                    elif not cell_content_str.strip().startswith(first_setup_variable_name + " ="):
                         cell_content_str = first_setup_variable_name + " = " + cell_content_str
                    exp_setup_processed = True

            cell_content_str = cell_content_str.replace(
                "pd.read_csv('uzupelnione_mieszkania.csv', sep=',')",
                f"pd.read_csv(r'{csv_input_path}', sep=',') # ZMODYFIKOWANE"
            )
            cell_content_str = cell_content_str.replace(
                "pd.read_csv('uzupelnione_mieszkania.csv')", 
                f"pd.read_csv(r'{csv_input_path}') # ZMODYFIKOWANE"
            )
            
            model_save_path = os.path.join(DATA_OUT_DIR, '0_best_price_modelLGBM_auto').replace('\\', '\\\\')
            cell_content_str = cell_content_str.replace(
                f"{first_setup_variable_name}.save_model(best_final_model, \"0_best_price_modelLGBM\"",
                f"{first_setup_variable_name}.save_model(best_final_model, r'{model_save_path}'"
            )
            cell_content_str = cell_content_str.replace( 
                "model = load_model('6_best_price_modelLGBM')", 
                f"model = load_model(r'{model_save_path}') # ZMODYFIKOWANE"
            )
            
            if "df_price2 = df_price1.dropna(subset=['Price'])" in cell_content_str:
                cell_content_str += """

# ----- WSTRZYKNIĘTY KOD PO UTWORZENIU df_price2 -----
if 'df_price2' in locals() and isinstance(df_price2, pd.DataFrame):
    if 'Description' in df_price2.columns:
        df_price2 = df_price2.drop(columns=['Description'], errors='ignore')
        logging.info("AUTO-MOD: Usunięto kolumnę 'Description' z df_price2.")
# ----- KONIEC WSTRZYKNIĘTEGO KODU -----
"""
            
            if first_setup_variable_name != "exp":
                 cell_content_str = cell_content_str.replace("exp.", f"{first_setup_variable_name}.")
            
            python_code_segments.append(cell_content_str)

            # Sprawdź, czy to komórka, po której mamy przestać dodawać kolejne
            if stop_processing_notebook_after_phrase in cell_content_str:
                logging.info(f"Znaleziono frazę stopu ('{stop_processing_notebook_after_phrase}') w komórce {cell_index+1}. Reszta notebooka zostanie pominięta przed dodaniem finalnego kodu zapisu.")
                stop_adding_cells_from_notebook = True
    
    # Dodajemy własny kod do predykcji i zapisu
    final_processing_code = f"""
# ----- POCZĄTEK FINALNEGO PRZETWARZANIA I ZAPISU (WERSJA 10) -----
# Zakładamy, że `df` to oryginalny DataFrame z pliku wejściowego.
# `{first_setup_variable_name}` to obiekt eksperymentu PyCaret.
# `best_final_model` to wytrenowany i sfinalizowany model.

if 'df' in locals() and isinstance(df, pd.DataFrame) and \
   '{first_setup_variable_name}' in locals() and \
   'best_final_model' in locals():
    
    pycaret_exp_obj_for_final_pred = locals().get('{first_setup_variable_name}')
    loaded_model_for_prediction = locals().get('best_final_model')

    logging.info("Przygotowywanie danych do finalnej predykcji (oryginalny DataFrame 'df')...")
    
    df_for_prediction_final = df.copy()
    
    if 'Price' in df_for_prediction_final.columns:
        df_for_prediction_final = df_for_prediction_final.drop(columns=['Price'])
        logging.info("Usunięto kolumnę 'Price' z danych przekazywanych do finalnego predict_model.")

    logging.info("Wykonywanie predykcji na zmodyfikowanym DataFrame 'df_for_prediction_final' za pomocą 'best_final_model'...")
    try:
        # Przekazujemy dane bez kolumny 'Price'
        predictions_on_full_df = pycaret_exp_obj_for_final_pred.predict_model(loaded_model_for_prediction, data=df_for_prediction_final)
        
        if 'prediction_label' in predictions_on_full_df.columns:
            # `predict_model` z argumentem `data` zwraca oryginalne kolumny z `data` (czyli z `df_for_prediction_final`)
            # oraz `prediction_label`. Chcemy teraz połączyć to z ORYGINALNYM `df`, aby zachować kolumnę 'Price'
            # jeśli była, oraz inne kolumny, które mogły zostać usunięte przed przekazaniem do `setup` PyCaret.

            original_df_with_all_cols = df.copy() # Oryginalny df z data.csv

            # Upewnij się, że 'SaleId' jest w `predictions_on_full_df`
            if 'SaleId' not in predictions_on_full_df.columns:
                logging.error("Krytyczny błąd: 'SaleId' nie ma w wynikach predict_model. Nie można połączyć z oryginalnymi danymi. Sprawdź `keep_features` w setup().")
            else:
                # Konwersja typów SaleId dla bezpiecznego merge'a
                try:
                    original_df_with_all_cols['SaleId'] = original_df_with_all_cols['SaleId'].astype(str)
                    predictions_on_full_df['SaleId'] = predictions_on_full_df['SaleId'].astype(str)
                except Exception as e_type_conv_saleid_final:
                    logging.warning(f"Problem podczas próby konwersji 'SaleId' na string w final_save_code: {{e_type_conv_saleid_final}}")
                
                # Scalamy oryginalny pełny df z kolumną predykcji, używając SaleId jako klucza
                # Bierzemy tylko SaleId i prediction_label z wyników predykcji, aby uniknąć duplikatów innych kolumn
                final_output_df = pd.merge(original_df_with_all_cols, 
                                           predictions_on_full_df[['SaleId', 'prediction_label']], 
                                           on='SaleId', 
                                           how='left') # Zachowaj wszystkie wiersze z oryginalnego df

                final_output_df.rename(columns={{'prediction_label': 'PredictedPriceByModel'}}, inplace=True)
            
                try:
                    output_path_script_var = r'{final_output_path}'
                    if 'SaleId' in final_output_df.columns: 
                        id_col = final_output_df.pop('SaleId')
                        final_output_df.insert(0, 'SaleId', id_col)
                    
                    final_output_df.to_csv(output_path_script_var, index=False, sep=',')
                    logging.info("Finalny DataFrame (oryginalne dane + predykcje) zapisany do: %s", output_path_script_var)
                except Exception as e_save:
                    logging.error("Błąd podczas zapisywania finalnego DataFrame: %s", e_save)
                    import traceback
                    logging.error(traceback.format_exc())
        else:
            logging.error("Kolumna 'prediction_label' nie została znaleziona w wynikach predict_model na pełnym 'df'.")

    except Exception as e_final_pred_save:
        logging.error("Błąd podczas finalnej predykcji (predict_model): %s", e_final_pred_save)
        import traceback
        logging.error(traceback.format_exc())
else:
    logging.warning("Nie znaleziono wymaganych obiektów ('df', '{first_setup_variable_name}', 'best_final_model') do przeprowadzenia finalnego przetwarzania.")
# ----- KONIEC FINALNEGO PRZETWARZANIA I ZAPISU (WERSJA 10) -----
"""
    python_code_segments.append(final_processing_code)

    full_script = "\n".join(python_code_segments)
    
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
        "final_output_path": final_output_path
    }

    logging.info("--- Rozpoczęcie wykonywania logiki z notebooka ---")
    try:
        exec(full_script, execution_globals)
        logging.info("--- Zakończono wykonywanie logiki z notebooka ---")
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