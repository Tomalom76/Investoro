import os
import json
import time
import shutil
import logging
import pandas as pd
from pycaret.regression import load_model, predict_model

# --- Konfiguracja ---
INPUT_DIR = "CENY_IN"
OUTPUT_DIR = "CENY_OUT"
PROCESSED_DIR = "CENY_PROCESSED"
MODEL_PATH = "final_price_model.pkl" # Nazwa pliku z zapisanym modelem
CHECK_INTERVAL_SECONDS = 5 # Co ile sekund sprawdzać katalog wejściowy

# Konfiguracja logowania, aby widzieć co robi skrypt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories():
    """Tworzy wymagane katalogi, jeśli nie istnieją."""
    logging.info("Sprawdzanie i tworzenie katalogów...")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logging.info(f"Katalogi gotowe: '{INPUT_DIR}', '{OUTPUT_DIR}', '{PROCESSED_DIR}'")

def process_json_file(filepath, model):
    """
    Wczytuje plik JSON, wykonuje predykcję i zapisuje wynik.
    """
    filename = os.path.basename(filepath)
    logging.info(f"Znaleziono nowy plik: {filename}")

    try:
        # 1. Wczytaj dane z pliku JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Konwertuj JSON do ramki danych (DataFrame) - model oczekuje ramki
        df_to_predict = pd.DataFrame([data])
        
        # 3. Przygotuj dane (preprocessing) - kluczowy krok!
        # Musimy odtworzyć kroki z notebooka, które PyCaret wykonuje automatycznie.
        # W tym przypadku głównie chodzi o zapewnienie poprawnych typów danych.
        # PyCaret's pipeline zajmie się resztą (imputacją, normalizacją).

        # Konwersja typów numerycznych
        numeric_cols = ['Area', 'NumberOfRooms', 'Floor', 'Floors']
        for col in numeric_cols:
            if col in df_to_predict.columns:
                df_to_predict[col] = pd.to_numeric(df_to_predict[col], errors='coerce')
        
        # Konwersja roku budowy na typ daty (tak jak w notebooku)
        if 'BuiltYear' in df_to_predict.columns:
            # Wypełniamy ewentualne braki (tutaj zakładamy, że dane są poprawne)
            # i konwertujemy na format daty, którego oczekuje pipeline
            df_to_predict['BuiltYear'] = pd.to_datetime(df_to_predict['BuiltYear'], format='%Y', errors='coerce')

        logging.info("Dane przygotowane do predykcji. Rozpoczynam analizę modelem...")

        # 4. Wykonaj predykcję
        predictions = predict_model(model, data=df_to_predict)
        
        # Wyciągnij przewidzianą wartość. W PyCaret domyślnie jest w kolumnie 'prediction_label'
        predicted_price = predictions['prediction_label'].iloc[0]
        
        # Zaokrąglamy dla czytelności
        predicted_price = round(predicted_price, 2)
        
        logging.info(f"Przewidywana cena dla oferty '{data.get('OfferID', 'N/A')}': {predicted_price}")

        # 5. Dodaj wynik do oryginalnych danych
        data['PredictedPrice'] = predicted_price
        
        # 6. Zapisz wynikowy JSON w katalogu wyjściowym
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Wynik zapisano w: {output_path}")

        # 7. Przenieś przetworzony plik do archiwum
        processed_path = os.path.join(PROCESSED_DIR, filename)
        shutil.move(filepath, processed_path)
        logging.info(f"Plik wejściowy przeniesiono do archiwum: {processed_path}")

    except Exception as e:
        logging.error(f"Wystąpił błąd podczas przetwarzania pliku {filename}: {e}")
        # Można też przenieść błędne pliki do osobnego katalogu 'errors'
        error_path = os.path.join(PROCESSED_DIR, f"ERROR_{filename}")
        shutil.move(filepath, error_path)
        logging.warning(f"Plik powodujący błąd został przeniesiony do: {error_path}")


def main():
    """Główna funkcja skryptu - pętla nasłuchująca."""
    setup_directories()

    # Sprawdź, czy model istnieje
    if not os.path.exists(MODEL_PATH):
        logging.error(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku modelu '{MODEL_PATH}'.")
        logging.error("Upewnij się, że plik `final_price_model.pkl` z notebooka znajduje się w tym samym katalogu co skrypt.")
        return

    # Wczytaj model raz, na początku
    logging.info("Wczytywanie wytrenowanego modelu...")
    try:
        model = load_model(MODEL_PATH)
        logging.info("Model wczytany pomyślnie.")
    except Exception as e:
        logging.error(f"Nie udało się wczytać modelu z pliku '{MODEL_PATH}': {e}")
        return

    logging.info("Rozpoczynam nasłuchiwanie na nowe pliki w katalogu CENY_IN...")
    while True:
        try:
            # Szukaj plików .json w katalogu wejściowym
            files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]

            if not files_to_process:
                # Jeśli nic nie ma, poczekaj
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            
            for filename in files_to_process:
                filepath = os.path.join(INPUT_DIR, filename)
                process_json_file(filepath, model)
        
        except KeyboardInterrupt:
            logging.info("Otrzymano sygnał zatrzymania. Kończę pracę.")
            break
        except Exception as e:
            logging.error(f"Wystąpił nieoczekiwany błąd w głównej pętli: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS * 2) # Dłuższa przerwa w razie błędu


if __name__ == "__main__":
    main()