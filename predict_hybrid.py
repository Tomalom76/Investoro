import os
import json
import time
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from shutil import move
from openai import OpenAI

# --- Konfiguracja ---
ARTIFACTS_DIR = 'model_artifacts_final' 
LOC_IN_DIR = 'LOC_IN'
LOC_OUT_DIR = 'LOC_OUT'
PROCESSED_DIR = os.path.join(LOC_IN_DIR, 'processed')
ERROR_DIR = os.path.join(LOC_IN_DIR, 'error')
MAX_LEN = 250

# ==============================================================================
# === KLUCZOWA POPRAWKA - WPISZ SWÓJ KLUCZ API BEZPOŚREDNIO TUTAJ ===
# ==============================================================================
OPENAI_API_KEY = ""  # <--- WKLEJ SWÓJ KLUCZ TUTAJ
# ==============================================================================

# --- Inicjalizacja klienta OpenAI ---
client = None # Ustawiamy None na początku na wszelki wypadek
print("--- Rozpoczęcie inicjalizacji klienta OpenAI ---")
try:
    print("Sprawdzanie klucza API...")
    if not OPENAI_API_KEY or "twoj-dlugi-klucz-api" in OPENAI_API_KEY:
        print("Błąd: Klucz API OpenAI nie został ustawiony lub zawiera placeholder!")
        raise ValueError("Klucz API OpenAI nie został ustawiony w skrypcie.")
    print(f"Klucz API wygląda na ustawiony (fragment: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}).") # Ostrożnie z wyświetlaniem klucza!
    print("Próba utworzenia instancji klienta OpenAI...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("Klient OpenAI utworzony pomyślnie. Testuję połączenie...")
    # Opcjonalnie: dodaj tu proste testowe zapytanie, np. listę modeli
    # client.models.list() 
    print("Klient OpenAI zainicjowany i gotowy do użycia.")
except Exception as e:
    print(f"!!! BŁĄD KRYTYCZNY podczas inicjalizacji klienta OpenAI: {type(e).__name__} - {e}")
    client = None
print(f"--- Koniec inicjalizacji klienta OpenAI. Wartość 'client': {client} ---")


model, tokenizer, numeric_pipeline, le_dzielnica, le_ulica, synonimy_map = (None,) * 6

def setup():
    global model, tokenizer, numeric_pipeline, le_dzielnica, le_ulica
    print("--- Inicjalizacja skryptu ---")
    for dir_path in [LOC_IN_DIR, LOC_OUT_DIR, PROCESSED_DIR, ERROR_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Ładowanie lokalnych artefaktów z folderu: {ARTIFACTS_DIR}")
    model = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, 'final_hierarchical_model.keras'))
    with open(os.path.join(ARTIFACTS_DIR, 'tokenizer.pkl'), 'rb') as f: tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'numeric_pipeline.pkl'), 'rb') as f: numeric_pipeline = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'le_dzielnica.pkl'), 'rb') as f: le_dzielnica = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'le_ulica.pkl'), 'rb') as f: le_ulica = pickle.load(f)
    print("Lokalne artefakty załadowane.")

def predict_with_local_model(data):
    df = pd.DataFrame([data])
    df['description_clean'] = df['Description'].apply(lambda x: re.sub(r'[^a-ząęółśżźćń ]', '', str(x).lower()))
    sequences = tokenizer.texts_to_sequences(df['description_clean'])
    X_text = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    df['Price_per_sqm'] = df.get('Price', 0) / df.get('Area', 1)
    df['Price_per_sqm'].replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_features_cols = ['Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'Price_per_sqm']
    X_numeric = numeric_pipeline.transform(df[numeric_features_cols])
    model_input = [X_text, X_numeric]
    pred_dzielnica_proba, pred_ulica_proba = model.predict(model_input, verbose=0)
    
    predicted_dzielnica = le_dzielnica.inverse_transform([np.argmax(pred_dzielnica_proba, axis=1)[0]])[0]
    conf_dzielnica = float(np.max(pred_dzielnica_proba, axis=1)[0])
    predicted_ulica = le_ulica.inverse_transform([np.argmax(pred_ulica_proba, axis=1)[0]])[0]
    conf_ulica = float(np.max(pred_ulica_proba, axis=1)[0])

    return {"Dzielnica": (predicted_dzielnica, conf_dzielnica), "Ulica": (predicted_ulica, conf_ulica)}

def get_location_from_gpt(description):
    print(" -> Wewnątrz get_location_from_gpt...")
    if not client:
        print(" -> Klient OpenAI nie jest zainicjowany. Zwracam None, None.")
        return None, None

    # Nowa, bardziej strukturalna lista wiadomości
    messages = [
        {"role": "system", "content": """Jesteś ekspertem od rynku nieruchomości w Warszawie. Twoim zadaniem jest precyzyjna identyfikacja najbardziej prawdopodobnej dzielnicy i ulicy na podstawie opisu nieruchomości.
Odpowiedz TYLKO w formacie JSON z kluczami "dzielnica" i "ulica".
Jeśli dzielnicy lub ulicy nie da się jednoznacznie ustalić na podstawie opisu, zwróć dla danego klucza null.
Pamiętaj o specjalnych przypadkach: 'Saska Kępa' to część dzielnicy 'Praga-Południe'. 'Stary Mokotów' to część dzielnicy 'Mokotów'.
Nie dodawaj żadnego innego tekstu ani komentarzy poza samym obiektem JSON."""}, # Bardziej precyzyjna instrukcja systemowa
        {"role": "user", "content": "Przeanalizuj następujący opis nieruchomości z Warszawy i zidentyfikuj dzielnicę oraz ulicę:"}, # Instrukcja dla modelu
        {"role": "user", "content": description} # <--- Tutaj przekazujesz SAM OPIS jako oddzielną wiadomość
    ]

    print(" -> Klient OpenAI jest zainicjowany. Przygotowuję wiadomości do API...")
    # Już nie potrzebujesz budować tego długiego promptu w f-stringu
    # prompt = f""" ... """

    print(" -> Wywołuję API OpenAI (gpt-4o)...")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o", # lub gpt-4o-mini jeśli wolisz, ale gpt-4o może być lepszy
            response_format={"type": "json_object"}, # To jest kluczowe do wymuszenia JSON
            messages=messages # Użyj listy messages z oddzielonym opisem
        )
        print(" -> Sukces: Otrzymano odpowiedź z API.")
        raw_gpt_response_content = completion.choices[0].message.content
        print(f" -> Surowa odpowiedź GPT: {raw_gpt_response_content}")

        # API z response_format="json_object" powinno zawsze zwrócić poprawny JSON
        # Ale nadal warto mieć blok try/except przy parsowaniu na wszelki wypadek
        try:
            response_json = json.loads(raw_gpt_response_content)
            gpt_dzielnica = response_json.get("dzielnica")
            gpt_ulica = response_json.get("ulica")
            print(f" -> Zwracam z GPT: Dzielnica='{gpt_dzielnica}', Ulica='{gpt_ulica}'")
            return gpt_dzielnica, gpt_ulica
        except json.JSONDecodeError:
             print(f"!!! BŁĄD: API nie zwróciło poprawnego JSON: {raw_gpt_response_content}")
             return None, None


    except Exception as e:
        print(f"!!! BŁĄD podczas komunikacji z API OpenAI w get_location_from_gpt: {type(e).__name__} - {e}")
        return None, None

def process_file(filepath):
    filename = os.path.basename(filepath)
    print(f"\nPrzetwarzanie pliku: {filename}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)

        # === KLUCZOWA POPRAWKA TUTAJ ===
        data = None # Inicjalizujemy data jako None

        # Sprawdź, czy wczytany JSON jest słownikiem i ma dokładnie jeden klucz na najwyższym poziomie
        if isinstance(loaded_json, dict) and len(loaded_json) == 1:
            # Pobierz nazwę tego jedynego klucza
            top_level_key = list(loaded_json.keys())[0]
            # Pobierz wartość pod tym kluczem
            top_level_value = loaded_json[top_level_key]

            # Sprawdź, czy wartość jest listą i czy lista nie jest pusta
            if isinstance(top_level_value, list) and len(top_level_value) > 0:
                # Sprawdź, czy pierwszy element listy jest słownikiem (czyli danymi oferty)
                if isinstance(top_level_value[0], dict):
                    data = top_level_value[0] # Jeśli tak, pobierz pierwszy słownik z listy
                    print(f"Odczytano plik {filename}. Zidentyfikowano dane oferty pod kluczem '{top_level_key}'.")
                else:
                    print(f"Błąd formatu pliku {filename}: Wartość pod kluczem '{top_level_key}' nie zawiera słownika oferty jako pierwszego elementu listy.")
            else:
                 print(f"Błąd formatu pliku {filename}: Wartość pod kluczem '{top_level_key}' nie jest listą lub jest pusta.")
        else:
            # Jeśli JSON nie pasuje do formatu ze zmienną nazwą klucza (np. jest to stary format)
            # Sprawdź, czy to może być stary format (bezpośredni słownik oferty)
            if isinstance(loaded_json, dict) and 'Description' in loaded_json: # Proste sprawdzenie, czy wygląda na stary format
                 data = loaded_json
                 print(f"Odczytano plik {filename} w starym formacie.")
            else:
                 print(f"Błąd formatu pliku {filename}: Nie rozpoznano struktury JSON.")


        # Jeśli zmienna data nadal jest None, oznacza to, że format pliku był niepoprawny
        if data is None:
             raise ValueError(f"Nie udało się poprawnie wczytać danych oferty z pliku {filename} z powodu nieznanego formatu JSON.")
        # =============================

        # Teraz zmienna 'data' zawiera już właściwy słownik z danymi oferty ('Description', 'Area', 'Price' etc.)
        # Reszta kodu funkcji process_file, która używa zmiennej 'data', może pozostać bez zmian.

        local_predictions = predict_with_local_model(data)
        print(f"Predykcja lokalnego modelu: Dzielnica='{local_predictions['Dzielnica'][0]}', Ulica='{local_predictions['Ulica'][0]}'")

        # DODANY WCZEŚNIEJ PRINT DEBUGUJĄCY
        # Upewniamy się, że klucz 'Description' istnieje w obiekcie 'data' przed próbą odwołania
        description_for_gpt = data.get('Description', None)
        if description_for_gpt is None:
             print(f"UWAGA: W danych oferty z pliku {filename} brak klucza 'Description' lub jego wartość to null.")
             # Możesz zdecydować, czy w takim przypadku wywoływać GPT czy nie
             # Na razie zostawiamy, ale get_location_from_gpt obsługuje pusty opis
             description_for_gpt = "" # Przekazujemy pusty string zamiast None, jeśli Description brak
        print(f"DEBUG: Wartość data['Description'] przekazywana do GPT: '{description_for_gpt}'")


        # Przekazujemy potencjalnie pusty string do get_location_from_gpt
        gpt_dzielnica, gpt_ulica = get_location_from_gpt(description_for_gpt)
        print(f"Sugestia GPT: Dzielnica='{gpt_dzielnica}', Ulica='{gpt_ulica}'")

        final_dzielnica = local_predictions['Dzielnica'][0]
        final_ulica = local_predictions['Ulica'][0]

        # Logika nadpisywania przez GPT (pozostała bez zmian)
        if gpt_dzielnica and gpt_dzielnica in le_dzielnica.classes_:
            print(f"GPT zweryfikował/poprawił dzielnicę na: '{gpt_dzielnica}'")
            final_dzielnica = gpt_dzielnica

        if gpt_ulica and gpt_ulica in le_ulica.classes_:
            print(f"GPT zweryfikował/poprawił ulicę na: '{gpt_ulica}'")
            final_ulica = gpt_ulica

        # ... reszta funkcji (zapisywanie wyników, przenoszenie pliku) ...
        # Zapisujemy zaktualizowany obiekt 'data'
        data['Prediction_Result'] = {
            "Final_Dzielnica": final_dzielnica,
            "Final_Ulica": final_ulica,
            "Lokalny_Model_Dzielnica": local_predictions['Dzielnica'][0],
            "Lokalny_Model_Dzielnica_Prob": round(local_predictions['Dzielnica'][1], 4),
            "Lokalny_Model_Ulica": local_predictions['Ulica'][0],
            "Lokalny_Model_Ulica_Prob": round(local_predictions['Ulica'][1], 4),
            "GPT_Dzielnica_Raw": gpt_dzielnica,
            "GPT_Ulica_Raw": gpt_ulica
        }

        output_filepath = os.path.join(LOC_OUT_DIR, filename)
        with open(output_filepath, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
        print("-" * 40)
        print(f"Wynik dla pliku: {filename}")
        print(f"  > Finalnie ustalona Dzielnica: {final_dzielnica}")
        print(f"  > Finalnie ustalona Ulica: {final_ulica}")
        print("-" * 40)
        move(filepath, os.path.join(PROCESSED_DIR, filename))
    except Exception as e:
        print(f"BŁĄD podczas przetwarzania pliku {filename}: {e}")
        move(filepath, os.path.join(ERROR_DIR, filename))

def main_loop():
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

if __name__ == '__main__':
    setup()
    if client:
        main_loop()
    else:
        print("Nie można uruchomić pętli głównej z powodu błędu inicjalizacji klienta OpenAI.")