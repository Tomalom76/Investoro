import time
import os
import json
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import sys # Do przekierowania stdout

# --- Konfiguracja ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
DATA_OUT_DIR = os.path.join(BASE_DIR, "DATA_OUT")
NOTEBOOK_FILENAME = "EDA_Projekt1_LGBM_v5.ipynb" # Upewnij się, że nazwa jest poprawna
NOTEBOOK_PATH = os.path.join(BASE_DIR, NOTEBOOK_FILENAME)
TRIGGER_FILE = "data.csv"
OUTPUT_FILENAME = "data_out.csv"

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Logowanie do konsoli

def create_dirs_if_not_exist():
    """Tworzy katalogi DATA i DATA_OUT, jeśli nie istnieją."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    logging.info(f"Katalogi DATA ('{DATA_DIR}') i DATA_OUT ('{DATA_OUT_DIR}') gotowe.")

def extract_and_execute_notebook_logic(csv_input_path, final_output_path):
    """
    Wyodrębnia i wykonuje logikę z notebooka Jupyter, dostosowując ścieżki plików.
    """
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
        logging.error(f"Błąd parsowania pliku notebooka: {NOTEBOOK_PATH}. Upewnij się, że to poprawny plik .ipynb.")
        return

    python_code_segments = []
    cells_to_skip_completely_by_content = [
        "df.head(", "df.sample(", ".plot_model(",
        "plt.figure(", "sns.heatmap(", "df.describe().T",
        "print(df_price1['Price']", 
        "exp.dataset_transformed.sample(10)",
        "unbalanced_metrics_df", 
        "best_final_model", # Wyświetlanie obiektu modelu, nie jego wyniku
        # "predictions", # Zostawiamy, bo może być użyte do zapisu
        "merged_df", # Zostawiamy, bo to finalny df
        "merged_df2" 
    ]
    intermediate_saves_to_skip = [
        "df_last.to_csv('0_new_prices.csv')",
        "merged_df.to_csv('0_new_prices_full.csv')",
    ]

    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            if not source_lines:
                continue
            
            cell_content_str = "".join(source_lines)

            if cell_content_str.strip() in ["df", "df_beznull_price", "df_price1", "df_price2"]:
                logging.info(f"Pominięto komórkę wyświetlającą DataFrame: {cell_content_str.strip()[:60]}...")
                continue
            
            if any(skip_phrase in cell_content_str for skip_phrase in cells_to_skip_completely_by_content):
                if "exp.predict_model" not in cell_content_str and "pull()" not in cell_content_str and "best_final_model =" not in cell_content_str:
                    logging.info(f"Pominięto komórkę EDA/plot/print: {cell_content_str.strip()[:60]}...")
                    continue
            
            cell_content_str = cell_content_str.replace(
                "pd.read_csv('uzupelnione_mieszkania.csv', sep=',')",
                f"pd.read_csv(r'{csv_input_path}', sep=',') # ZMODYFIKOWANE PRZEZ SKRYPT"
            )
            cell_content_str = cell_content_str.replace(
                "pd.read_csv('uzupelnione_mieszkania.csv')", 
                f"pd.read_csv(r'{csv_input_path}') # ZMODYFIKOWANE PRZEZ SKRYPT"
            )
            cell_content_str = cell_content_str.replace(
                "exp.save_model(best_final_model, \"0_best_price_modelLGBM\"",
                f"exp.save_model(best_final_model, r'{os.path.join(DATA_OUT_DIR, '0_best_price_modelLGBM_auto')}'"
            )
            cell_content_str = cell_content_str.replace(
                "model = load_model('6_best_price_modelLGBM')", 
                f"model = load_model(r'{os.path.join(DATA_OUT_DIR, '0_best_price_modelLGBM_auto')}') # ZMODYFIKOWANE"
            )
            cell_content_str = cell_content_str.replace( # Na wypadek gdybyś użył tej nazwy pliku w innym miejscu
                "uzupelnione_mieszkania_ceny.csv",
                f"{os.path.join(DATA_OUT_DIR, 'uzupelnione_mieszkania_ceny_auto.csv')}"
            )


            for save_call in intermediate_saves_to_skip:
                if save_call in cell_content_str:
                    cell_content_str = cell_content_str.replace(save_call, f"# {save_call} # SKRYPT POMINĄŁ TEN ZAPIS")
                    logging.info(f"Pominięto zapis pośredni: {save_call}")

            python_code_segments.append(cell_content_str)

            if "df_price2 = df_price1.dropna(subset=['Price'])" in cell_content_str:
                fix_code = """
# ----- POCZĄTEK KODU WSTRZYKNIĘTEGO PRZEZ SKRYPT (Naprawa Description) -----
if 'df_price2' in locals() and isinstance(df_price2, pd.DataFrame):
    if 'Description' in df_price2.columns:
        df_price2 = df_price2.drop(columns=['Description'], errors='ignore')
        logging.info("Kolumna 'Description' została usunięta z df_price2 przed setupem PyCaret.")
# ----- KONIEC KODU WSTRZYKNIĘTEGO PRZEZ SKRYPT -----
"""
                python_code_segments.append(fix_code)
                logging.info("Wstrzyknięto kod naprawczy dla kolumny 'Description' po utworzeniu df_price2.")
    
    final_save_code = f"""
# ----- POCZĄTEK FINALNEGO ZAPISU WSTRZYKNIĘTEGO PRZEZ SKRYPT -----
# Ostatnia komórka Twojego notebooka zapisuje 'merged_df' do 'uzupelnione_mieszkania_ceny.csv'
# Zamiast tego, zapiszemy ją do {OUTPUT_FILENAME} w {DATA_OUT_DIR}
final_df_to_save = None
if 'merged_df' in locals() and isinstance(merged_df, pd.DataFrame):
    final_df_to_save = merged_df
    df_name_for_log = 'merged_df'
elif 'predictions' in locals() and isinstance(predictions, pd.DataFrame) and 'prediction_label' in predictions.columns:
    # Jeśli 'merged_df' nie istnieje, ale 'predictions' tak i ma wyniki
    # Możliwe, że chcesz zapisać `predictions` z dodaną kolumną 'SaleId'
    if 'SaleId' in locals() and isinstance(SaleId, pd.Series): # Zakładając, że 'SaleId' to seria zresetowana
         temp_df = pd.DataFrame({{'SaleId': SaleId, 'prediction_label': predictions['prediction_label']}})
         # Można by spróbować zmergować z oryginalnym df_price2 lub czymś podobnym
         # Ale dla prostoty, zapiszmy to co mamy
         final_df_to_save = temp_df
         df_name_for_log = 'predictions (SaleId + prediction_label)'
    else:
        final_df_to_save = predictions
        df_name_for_log = 'predictions'
else:
    logging.warning("Nie znaleziono odpowiedniej ramki danych ('merged_df' lub 'predictions') do zapisu jako wynik końcowy.")

if final_df_to_save is not None:
    try:
        output_path_script = r'{final_output_path}'
        final_df_to_save.to_csv(output_path_script, index=False, sep=',')
        logging.info("Finalna ramka danych '%s' zapisana do: %s", df_name_for_log, output_path_script)
    except Exception as e_save:
        logging.error("Błąd podczas zapisywania final_df_to_save ('%s'): %s", df_name_for_log, e_save)
# ----- KONIEC FINALNEGO ZAPISU WSTRZYKNIĘTEGO PRZEZ SKRYPT -----
"""
    python_code_segments.append(final_save_code)

    full_script = "\n".join(python_code_segments)
    
    execution_globals = {
        "pd": pd, 
        "logging": logging,
        "__name__": "__main__", 
        "os": os, 
        "sys": sys,
        # Dodajemy zmienne ścieżek, aby były dostępne, jeśli notebook ich używa bezpośrednio (choć nie powinien)
        "DATA_DIR": DATA_DIR,
        "DATA_OUT_DIR": DATA_OUT_DIR,
        "csv_input_path": csv_input_path, # Na wypadek, gdyby był używany bezpośrednio
        "final_output_path": final_output_path # Na wypadek, gdyby był używany bezpośrednio
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
            output_file_path = os.path.join(DATA_OUT_DIR, OUTPUT_FILENAME)
            
            extract_and_execute_notebook_logic(csv_file_to_process, output_file_path)

if __name__ == "__main__":
    create_dirs_if_not_exist()

    event_handler = ModelTrainingHandler()
    observer = Observer()
    observer.schedule(event_handler, DATA_DIR, recursive=False) 
    
    logging.info(f"Nasłuchiwanie na zmiany w katalogu: {DATA_DIR}")
    logging.info(f"Oczekiwany plik: {TRIGGER_FILE}")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Zatrzymano nasłuchiwanie.")
    observer.join()