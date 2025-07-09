import os
import json
import time
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from shutil import move

# --- Konfiguracja ---
# Upewnij się, że ta nazwa pasuje do folderu, który stworzył Twój notebook
ARTIFACTS_DIR = 'model_artifacts_final' 
LOC_IN_DIR = 'LOC_IN'
LOC_OUT_DIR = 'LOC_OUT'
PROCESSED_DIR = os.path.join(LOC_IN_DIR, 'processed')
ERROR_DIR = os.path.join(LOC_IN_DIR, 'error')
MAX_LEN = 250

# --- Zmienne globalne ---
model, tokenizer, numeric_pipeline, le_dzielnica, le_ulica = (None,) * 5

def setup():
    """Ładuje wszystkie artefakty do pamięci."""
    global model, tokenizer, numeric_pipeline, le_dzielnica, le_ulica
    print("--- Inicjalizacja skryptu predykcyjnego ---")
    
    for dir_path in [LOC_IN_DIR, LOC_OUT_DIR, PROCESSED_DIR, ERROR_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Ładowanie artefaktów z folderu: {ARTIFACTS_DIR}")
    model = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, 'final_hierarchical_model.keras'))
    with open(os.path.join(ARTIFACTS_DIR, 'tokenizer.pkl'), 'rb') as f: tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'numeric_pipeline.pkl'), 'rb') as f: numeric_pipeline = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'le_dzielnica.pkl'), 'rb') as f: le_dzielnica = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'le_ulica.pkl'), 'rb') as f: le_ulica = pickle.load(f)
        
    print("Artefakty załadowane. Skrypt gotowy do pracy.")

def clean_text(text):
    """Czyści tekst opisu - ta sama funkcja co w notebooku."""
    return re.sub(r'[^a-ząęółśżźćń ]', '', str(text).lower())

def predict_from_json(data):
    """Dokonuje predykcji na podstawie danych z jednego pliku JSON."""
    df = pd.DataFrame([data])
    
    # Przygotowanie wejść dla modelu
    # WAŻNE: Nie wzbogacamy opisu tutaj. Model sam ma znaleźć korelacje.
    df['description_clean'] = df['Description'].apply(clean_text)
    sequences = tokenizer.texts_to_sequences(df['description_clean'])
    X_text = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN)

    df['Price_per_sqm'] = df.get('Price', 0) / df.get('Area', 1)
    df['Price_per_sqm'].replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_features_cols = ['Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'Price_per_sqm']
    X_numeric = numeric_pipeline.transform(df[numeric_features_cols])

    # Predykcja
    model_input = [X_text, X_numeric]
    pred_dzielnica_proba, pred_ulica_proba = model.predict(model_input, verbose=0)
    
    # Dekodowanie wyników
    predicted_dzielnica = le_dzielnica.inverse_transform([np.argmax(pred_dzielnica_proba, axis=1)[0]])[0]
    conf_dzielnica = float(np.max(pred_dzielnica_proba, axis=1)[0])

    predicted_ulica = le_ulica.inverse_transform([np.argmax(pred_ulica_proba, axis=1)[0]])[0]
    conf_ulica = float(np.max(pred_ulica_proba, axis=1)[0])

    return {
        "Dzielnica": (predicted_dzielnica, conf_dzielnica),
        "Ulica": (predicted_ulica, conf_ulica)
    }

def process_file(filepath):
    """Wczytuje plik, dokonuje predykcji i zapisuje wynik."""
    filename = os.path.basename(filepath)
    print(f"\nPrzetwarzanie pliku: {filename}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        
        predictions = predict_from_json(data)
        
        data['Prediction_Result'] = {
            "Predicted_Dzielnica": predictions["Dzielnica"][0],
            "Predicted_Dzielnica_Prob": round(predictions["Dzielnica"][1], 4),
            "Predicted_Ulica": predictions["Ulica"][0],
            "Predicted_Ulica_Prob": round(predictions["Ulica"][1], 4),
        }
        
        output_filepath = os.path.join(LOC_OUT_DIR, filename)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("-" * 40)
        print(f"Wynik dla pliku: {filename}")
        print(f"  > Przewidziana Dzielnica: {predictions['Dzielnica'][0]} (Pewność: {predictions['Dzielnica'][1]:.2%})")
        print(f"  > Przewidziana Ulica: {predictions['Ulica'][0]} (Pewność: {predictions['Ulica'][1]:.2%})")
        print("-" * 40)
        
        move(filepath, os.path.join(PROCESSED_DIR, filename))
    except Exception as e:
        print(f"BŁĄD podczas przetwarzania pliku {filename}: {e}")
        move(filepath, os.path.join(ERROR_DIR, filename))

def main_loop():
    """Główna pętla nasłuchująca."""
    while True:
        try:
            files_to_process = [f for f in os.listdir(LOC_IN_DIR) if f.endswith('.json')]
            if files_to_process:
                for filename in files_to_process:
                    process_file(os.path.join(LOC_IN_DIR, filename))
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nZatrzymywanie skryptu...")
            break
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd w pętli głównej: {e}")
            time.sleep(10) # Dłuższa przerwa w razie błędu

if __name__ == '__main__':
    setup()
    main_loop()