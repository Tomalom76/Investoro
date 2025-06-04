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
        "best_final_model", 
        "df_last", 
        "saleflats_df", 
        "new_prices_df", 
        "merged_df2", 
        "predicted_column"
    ]
    
    critical_phrases_for_cell_commenting = [
        "df_last.to_csv('0_new_prices.csv')",
        "new_prices_df = pd.read_csv('0_new_prices.csv')",
        "saleflats_df = pd.read_csv('data.csv')", 
        "merged_df = pd.merge(saleflats_df, new_prices_df", 
        "merged_df.to_csv('0_new_prices_full.csv')",
        "merged_df = pd.read_csv('0_new_prices_full.csv')",
        "merged_df.to_csv('uzupelnione_mieszkania_ceny.csv')"
    ]
    other_lines_to_comment_out = [
        "print(\"new_prices_df.columns:\", new_prices_df.columns.tolist())",
        "new_prices_df = new_prices_df.rename(columns=",
        "print(new_prices_df.columns)",
        "merged_df.drop(columns=['SaleID']",
        "cols = merged_df.columns.tolist()", 
        "price_index = cols.index('Price')",
        "cols.remove('NewPrice')",
        "cols.insert(price_index + 1, 'NewPrice')",
        "merged_df = merged_df[cols]", 
        "merged_df[merged_df.duplicated()]", 
        "prediction_df = merged_df.copy()",
        "prediction_df_clean = prediction_df.drop(columns=['Price']", 
        "predictions['RealPrice'] = prediction_df['Price']", 
        "merged_df['PredictedPrice'] = predictions['prediction_label']", 
        "merged_df.drop(columns=['NewPrice']", 
        "merged_df2=merged_df[['PredictedPrice',]].applymap(", 
        "merged_df['PredictedPrice'] = merged_df2['PredictedPrice']"
    ]
    
    lines_to_remove_from_cells = [
        "exp.dataset_transformed.sample(10)",
        "unbalanced_exp.dataset_transformed.sample(10)"
    ]

    stop_processing_notebook_after_phrase = "predictions = exp.predict_model(best_final_model)" 
    # Lub "exp.save_model(best_final_model" jeśli zapis modelu jest ostatnią potrzebną rzeczą z notebooka.
    # Na podstawie Twojego ostatniego debuga, `predictions = exp.predict_model(best_final_model)` jest przedostatnią
    # linią przed operacjami na plikach.

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
            
            cell_content_for_critical_check = "".join(source_lines)
            cell_should_be_entirely_commented = False
            for critical_phrase in critical_phrases_for_cell_commenting:
                # Dodatkowe sprawdzenie, czy krytyczna fraza nie jest częścią frazy stopu
                # (którą chcemy zachować, jeśli jest np. definicją `predictions`)
                if critical_phrase in cell_content_for_critical_check and \
                   (stop_processing_notebook_after_phrase not in cell_content_for_critical_check or \
                    critical_phrase != stop_processing_notebook_after_phrase):
                    logging.info(f"AUTO-MOD: Cała komórka {cell_index+1} zostanie zakomentowana z powodu krytycznej frazy: '{critical_phrase}'")
                    python_code_segments.append(f"\n# --- POCZĄTEK AUTOMATYCZNIE ZAKOMENTOWANEJ KOMÓRKI {cell_index+1} (krytyczna fraza: {critical_phrase}) ---")
                    for l_orig in source_lines:
                        python_code_segments.append(f"# {l_orig.strip()}")
                    python_code_segments.append(f"# --- KONIEC AUTOMATYCZNIE ZAKOMENTOWANEJ KOMÓRKI {cell_index+1} ---\n")
                    cell_should_be_entirely_commented = True
                    break
            
            if cell_should_be_entirely_commented:
                continue

            modified_source_lines_for_this_cell = []
            for line in source_lines:
                line_was_removed = False
                for lineToRemove in lines_to_remove_from_cells:
                    if lineToRemove in line:
                        logging.info(f"AUTO-MOD: Usunięto linię '{line.strip()}' z komórki {cell_index+1}")
                        line_was_removed = True
                        break
                if line_was_removed:
                    continue

                comment_this_line_flag = False
                for phrase_to_comment in other_lines_to_comment_out:
                    if phrase_to_comment in line:
                        modified_source_lines_for_this_cell.append(f"# {line.strip()} # SKRYPT ZAKOMENTOWAŁ TĘ LINIĘ (indywidualnie)\n")
                        logging.info(f"AUTO-MOD: Zakomentowano linię (indywidualnie) '{line.strip()}' w komórce {cell_index+1}")
                        comment_this_line_flag = True
                        break
                if not comment_this_line_flag:
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
            # Zmiana: użyj first_setup_variable_name dla save_model
            cell_content_str = cell_content_str.replace(
                "exp.save_model(best_final_model, \"0_best_price_modelLGBM\"",  # Oryginalna linia, jeśli 'exp' było używane
                f"{first_setup_variable_name}.save_model(best_final_model, r'{model_save_path}'"
            )
            cell_content_str = cell_content_str.replace( # Jeśli w notebooku było `unbalanced_exp`
                "unbalanced_exp.save_model(best_final_model, \"0_best_price_modelLGBM\"",
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
            
            # Upewnij się, że wszystkie odwołania do `exp.` są zamienione na `first_setup_variable_name.`
            # jeśli `first_setup_variable_name` nie jest już "exp"
            if first_setup_variable_name != "exp":
                 cell_content_str = cell_content_str.replace("exp.", f"{first_setup_variable_name}.")
            # Podobnie dla `unbalanced_exp`, jeśli `first_setup_variable_name` to `exp`
            elif "unbalanced_exp." in cell_content_str:
                 cell_content_str = cell_content_str.replace("unbalanced_exp.", f"{first_setup_variable_name}.")

            python_code_segments.append(cell_content_str)

            # Sprawdź, czy to komórka, po której mamy przestać dodawać kolejne
            # Użyj zmodyfikowanej `cell_content_str` do sprawdzenia frazy stopu
            # (np. jeśli fraza stopu używała `exp.`, a my to zmieniliśmy na `unbalanced_exp.`)
            current_stop_phrase = stop_processing_notebook_after_phrase
            if first_setup_variable_name != "exp": # Jeśli główny eksperyment to nie 'exp'
                current_stop_phrase = current_stop_phrase.replace("exp.", f"{first_setup_variable_name}.")

            if current_stop_phrase in cell_content_str:
                logging.info(f"Znaleziono frazę stopu ('{current_stop_phrase}') w komórce {cell_index+1}. Reszta notebooka zostanie pominięta przed dodaniem finalnego kodu zapisu.")
                stop_adding_cells_from_notebook = True
    
    final_processing_code = f"""
# ----- POCZĄTEK FINALNEGO PRZETWARZANIA I ZAPISU (WERSJA 12) -----
if 'df' in locals() and isinstance(df, pd.DataFrame) and \
   '{first_setup_variable_name}' in locals() and \
   'best_final_model' in locals():
    
    pycaret_exp_obj_for_final_pred = locals().get('{first_setup_variable_name}')
    loaded_model_for_prediction = locals().get('best_final_model')

    logging.info("Przygotowywanie danych do finalnej predykcji (oryginalny DataFrame 'df')...")
    
    df_for_prediction_final = df.copy()
    original_price_column_exists_in_df = 'Price' in df_for_prediction_final.columns # Zapamiętaj oryginalną kolumnę Price z df
    
    if original_price_column_exists_in_df:
        df_for_prediction_final_no_price = df_for_prediction_final.drop(columns=['Price'])
        logging.info("Usunięto kolumnę 'Price' z danych przekazywanych do finalnego predict_model.")
    else:
        df_for_prediction_final_no_price = df_for_prediction_final # Jeśli nie ma 'Price', użyj jak jest

    logging.info("Wykonywanie predykcji na zmodyfikowanym DataFrame za pomocą 'best_final_model'...")
    try:
        predictions_from_pycaret = pycaret_exp_obj_for_final_pred.predict_model(loaded_model_for_prediction, data=df_for_prediction_final_no_price)
        
        if 'prediction_label' in predictions_from_pycaret.columns:
            if 'SaleId' not in predictions_from_pycaret.columns:
                logging.error("Krytyczny błąd: 'SaleId' nie ma w wynikach predict_model. Nie można połączyć z oryginalnymi danymi. Sprawdź `keep_features` w setup().")
            else:
                final_output_df = df.copy() 

                predictions_to_add = predictions_from_pycaret[['SaleId', 'prediction_label']].copy()
                predictions_to_add.rename(columns={{'prediction_label': 'PredictedPriceByModel'}}, inplace=True)

                try:
                    final_output_df['SaleId'] = final_output_df['SaleId'].astype(str)
                    predictions_to_add['SaleId'] = predictions_to_add['SaleId'].astype(str)
                except Exception as e_type_conv_saleid_final:
                    logging.warning(f"Problem podczas próby konwersji 'SaleId' na string w final_save_code: {{e_type_conv_saleid_final}}")

                final_output_df = pd.merge(final_output_df, predictions_to_add, on='SaleId', how='left')
                
                if 'PredictedPriceByModel' in final_output_df.columns:
                    final_output_df['PredictedPriceByModel_numeric'] = pd.to_numeric(final_output_df['PredictedPriceByModel'], errors='coerce').round(0)
                    final_output_df['PredictedPriceByModel'] = final_output_df['PredictedPriceByModel_numeric'].apply(
                        lambda x: f'{{x:,.0f}}' if pd.notna(x) else '' 
                    )
                    final_output_df.drop(columns=['PredictedPriceByModel_numeric'], inplace=True)
                    logging.info("Sformatowano kolumnę 'PredictedPriceByModel'.")

                # Przesuwanie kolumny PredictedPriceByModel
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
                elif 'PredictedPriceByModel' in cols: # Jeśli Price nie ma, ale jest predykcja
                    logging.info("Kolumna 'Price' nie istnieje, 'PredictedPriceByModel' pozostaje na końcu.")


                try:
                    output_path_script_var = r'{final_output_path}'
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
# ----- KONIEC FINALNEGO PRZETWARZANIA I ZAPISU (WERSJA 12) -----
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