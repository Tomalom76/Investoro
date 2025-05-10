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
        "print(df_price1['Price']", # Te printy wartości są dla EDA
        "exp.dataset_transformed.sample(10)", # To jest wyświetlanie
        "unbalanced_metrics_df", # Wyświetlanie ramki
        "best_final_model", # Wyświetlanie obiektu modelu
        "predictions", # Wyświetlanie ramki
        "merged_df", # Wyświetlanie ramki
        "merged_df2" # Wyświetlanie ramki
    ]
    # Pomijanie konkretnych plików zapisu, które nie są finalnym outputem
    intermediate_saves_to_skip = [
        "df_last.to_csv('0_new_prices.csv')",
        "merged_df.to_csv('0_new_prices_full.csv')",
        # "merged_df.to_csv('uzupelnione_mieszkania_ceny.csv')" # Ten chcemy zastąpić
    ]

    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            if not source_lines:
                continue
            
            cell_content_str = "".join(source_lines)

            # Pomijanie komórek wyświetlających tylko DataFrame lub goły obiekt
            if cell_content_str.strip() in ["df", "df_beznull_price", "df_price1", "df_price2"]:
                logging.info(f"Pominięto komórkę wyświetlającą: {cell_content_str.strip()[:60]}...")
                continue
            
            # Pomijanie komórek zdefiniowanych przez ich zawartość (np. plotting)
            if any(skip_phrase in cell_content_str for skip_phrase in cells_to_skip_completely_by_content):
                 # Ale nie pomijaj, jeśli to predict_model, bo może być potrzebne
                if "exp.predict_model" not in cell_content_str and "pull()" not in cell_content_str:
                    logging.info(f"Pominięto komórkę EDA/plot: {cell_content_str.strip()[:60]}...")
                    continue
            
            # Modyfikacja ścieżki wczytywania danych
            cell_content_str = cell_content_str.replace(
                "pd.read_csv('uzupelnione_mieszkania.csv', sep=',')",
                f"pd.read_csv(r'{csv_input_path}', sep=',') # ZMODYFIKOWANE PRZEZ SKRYPT"
            )
            cell_content_str = cell_content_str.replace(
                "pd.read_csv('uzupelnione_mieszkania.csv')", # Jeśli bez sep=','
                f"pd.read_csv(r'{csv_input_path}') # ZMODYFIKOWANE PRZEZ SKRYPT"
            )
             # Modyfikacja ścieżki zapisu modelu
            cell_content_str = cell_content_str.replace(
                "exp.save_model(best_final_model, \"0_best_price_modelLGBM\"",
                f"exp.save_model(best_final_model, r'{os.path.join(DATA_OUT_DIR, \"0_best_price_modelLGBM_auto\")}'"
            )
            cell_content_str = cell_content_str.replace(
                "model = load_model('6_best_price_modelLGBM')", # Jeśli ładowany jest model
                f"model = load_model(r'{os.path.join(DATA_OUT_DIR, \"0_best_price_modelLGBM_auto\")}') # ZMODYFIKOWANE"
            )

            # Pomijanie specyficznych zapisów pośrednich
            for save_call in intermediate_saves_to_skip:
                if save_call in cell_content_str:
                    cell_content_str = cell_content_str.replace(save_call, f"# {save_call} # SKRYPT POMINĄŁ TEN ZAPIS")
                    logging.info(f"Pominięto zapis pośredni: {save_call}")

            python_code_segments.append(cell_content_str)

            # Wstrzyknięcie kodu naprawiającego problem z kolumną 'Description'
            # po zdefiniowaniu df_price2
            if "df_price2 = df_price1.dropna(subset=['Price'])" in cell_content_str:
                fix_code = """
# ----- POCZĄTEK KODU WSTRZYKNIĘTEGO PRZEZ SKRYPT -----
if 'df_price2' in locals() and isinstance(df_price2, pd.DataFrame):
    logging.info("Sprawdzanie i potencjalne usuwanie kolumny 'Description' z df_price2...")
    if 'Description' in df_price2.columns:
        df_price2 = df_price2.drop(columns=['Description'], errors='ignore')
        logging.info("Kolumna 'Description' została usunięta z df_price2.")
    else:
        logging.info("Kolumna 'Description' nie istnieje w df_price2.")
else:
    logging.warning("Ramka danych df_price2 nie została znaleziona po jej oczekiwanym utworzeniu.")
# ----- KONIEC KODU WSTRZYKNIĘTEGO PRZEZ SKRYPT -----
"""
                python_code_segments.append(fix_code)
                logging.info("Wstrzyknięto kod naprawczy dla kolumny 'Description' po utworzeniu df_price2.")
    
    # Dodanie zapisu finalnego pliku CSV na końcu skryptu
    final_save_code = f"""
# ----- POCZĄTEK FINALNEGO ZAPISU WSTRZYKNIĘTEGO PRZEZ SKRYPT -----
if 'merged_df' in locals() and isinstance(merged_df, pd.DataFrame):
    try:
        output_path_script = r'{final_output_path}'
        merged_df.to_csv(output_path_script, index=False, sep=',')
        logging.info(f"Finalna ramka danych 'merged_df' zapisana do: {{output_path_script}}")
    except Exception as e_save:
        logging.error(f"Błąd podczas zapisywania merged_df: {{e_save}}")
elif 'predictions' in locals() and isinstance(predictions, pd.DataFrame): # Fallback
    try:
        output_path_script = r'{final_output_path}'
        predictions.to_csv(output_path_script, index=False, sep=',')
        logging.info(f"Ramka danych 'predictions' zapisana do: {{output_path_script}}")
    except Exception as e_save:
        logging.error(f"Błąd podczas zapisywania predictions: {{e_save}}")
else:
    logging.warning("Nie znaleziono ramki danych 'merged_df' ani 'predictions' do zapisu jako wynik końcowy.")
# ----- KONIEC FINALNEGO ZAPISU WSTRZYKNIĘTEGO PRZEZ SKRYPT -----
"""
    python_code_segments.append(final_save_code)

    full_script = "\n".join(python_code_segments)
    
    # Przygotowanie globalnego zakresu dla exec, wstrzyknięcie pandas i logging
    # PyCaret i inne biblioteki z notebooka zostaną zaimportowane przez kod z notebooka
    execution_globals = {
        "pd": pd, 
        "logging": logging,
        "__name__": "__main__", # Symulacja uruchomienia jako główny skrypt
        "os": os, # Może być potrzebne w notebooku
        "sys": sys # Może być potrzebne w notebooku
    }

    logging.info("--- Rozpoczęcie wykonywania logiki z notebooka ---")
    try:
        # Uruchomienie zaimportowanych modułów i kodu z notebooka
        # Pierwsza komórka notebooka zawiera importy, więc powinny być dostępne
        exec(full_script, execution_globals)
        logging.info("--- Zakończono wykonywanie logiki z notebooka ---")
    except Exception as e:
        logging.error(f"Błąd podczas wykonywania kodu z notebooka: {e}")
        import traceback
        logging.error(traceback.format_exc())


class ModelTrainingHandler(FileSystemEventHandler):
    """Obsługuje zdarzenia w systemie plików."""
    def on_created(self, event):
        if not event.is_directory and os.path.basename(event.src_path) == TRIGGER_FILE:
            logging.info(f"Wykryto nowy plik: {event.src_path}")
            # Krótka pauza, aby upewnić się, że plik jest w pełni zapisany
            time.sleep(5) 
            
            csv_file_to_process = event.src_path
            output_file_path = os.path.join(DATA_OUT_DIR, OUTPUT_FILENAME)
            
            extract_and_execute_notebook_logic(csv_file_to_process, output_file_path)

if __name__ == "__main__":
    create_dirs_if_not_exist()

    event_handler = ModelTrainingHandler()
    observer = Observer()
    observer.schedule(event_handler, DATA_DIR, recursive=False) # Nie przeszukuj podkatalogów
    
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