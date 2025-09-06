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
from datetime import datetime # Importujemy do pracy z datami

# --- Konfiguracja ---
ARTIFACTS_DIR = 'model_artifacts_final'
LOC_IN_DIR = 'LOC_IN'
LOC_OUT_DIR = 'LOC_OUT'
PROCESSED_DIR = os.path.join(LOC_IN_DIR, 'processed')
ERROR_DIR = os.path.join(LOC_IN_DIR, 'error')
MAX_LEN = 250

# Rok do porównania dla stanu deweloperskiego (oferty starsze niż styczeń tego roku -> Stan_Deweloperski)
STAN_DEWELOPERSKI_ROK_GRANICZNY = 2024

# Możliwe wartości dla stanu mieszkania, które powinien przewidzieć GPT
POSSIBLE_CONDITIONS = ["Do_Remontu", "Dobry", "Po_Remoncie", "Stan_Deweloperski"]

# ==============================================================================
# === KLUCZOWA POPRAWKA - WPISZ SWÓJ KLUCZ API BEZPOŚREDNIO TUTAJ ===
# ==============================================================================
# PAMIĘTAJ, ABY WSTAWIĆ SWÓJ KLUCZ ZACZYNAJĄCY SIĘ OD "sk-" ZAMIAST TEGO TEKSTU PONIŻEJ
OPENAI_API_KEY = "sk-twoj-dlugi-klucz-api"
# ==============================================================================

# --- Inicjalizacja klienta OpenAI ---
client = None # Ustawiamy None na początku na wszelki wypadek
print("--- Rozpoczęcie inicjalizacji klienta OpenAI ---")
try:
    print("Sprawdzanie klucza API...")
    # Sprawdź, czy klucz API jest ustawiony i nie zawiera domyślnego placeholdera
    if not OPENAI_API_KEY or "twoj-dlugi-klucz-api" in OPENAI_API_KEY:
        print("Błąd: Klucz API OpenAI nie został ustawiony w skrypcie lub zawiera domyślny placeholder!")
        raise ValueError("Klucz API OpenAI nie został ustawiony w skrypcie.")
    print(f"Klucz API wygląda na ustawiony (fragment: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}).") # Ostrożnie z wyświetlaniem fragmentu klucza!
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
    """Inicjalizuje skrypt, ładuje lokalne modele i artefakty."""
    global model, tokenizer, numeric_pipeline, le_dzielnica, le_ulica
    print("--- Inicjalizacja skryptu ---")
    for dir_path in [LOC_IN_DIR, LOC_OUT_DIR, PROCESSED_DIR, ERROR_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Ładowanie lokalnych artefaktów z folderu: {ARTIFACTS_DIR}")
    try:
        model = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, 'final_hierarchical_model.keras'))
        with open(os.path.join(ARTIFACTS_DIR, 'tokenizer.pkl'), 'rb') as f: tokenizer = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'numeric_pipeline.pkl'), 'rb') as f: numeric_pipeline = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'le_dzielnica.pkl'), 'rb') as f: le_dzielnica = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'le_ulica.pkl'), 'rb') as f: le_ulica = pickle.load(f)
        print("Lokalne artefakty załadowane pomyślnie.")
         # Opcjonalne printy słowników
        # print(f"Znane dzielnice w słowniku modelu: {list(le_dzielnica.classes_)[:10]}... ({len(le_dzielnica.classes_)} łącznie)")
        # print(f"Znane ulice w słowniku modelu: {list(le_ulica.classes_)[:10]}... ({len(le_ulica.classes_)} łącznie)")
    except Exception as e:
        print(f"!!! BŁĄD KRYTYCZNY podczas ładowania lokalnych artefaktów: {type(e).__name__} - {e}")
        # Ustaw None dla artefaktów, aby uniknąć dalszych błędów, ale skrypt może nie działać poprawnie bez nich
        model, tokenizer, numeric_pipeline, le_dzielnica, le_ulica = (None,) * 5


def predict_with_local_model(data):
    """Wykonuje predykcję lokalizacji za pomocą modelu TensorFlow."""
    if model is None or tokenizer is None or numeric_pipeline is None or le_dzielnica is None or le_ulica is None:
        print("Błąd: Lokalny model lub artefakty nie zostały poprawnie załadowane. Pomijam predykcję lokalną.")
        return {"Dzielnica": ("Błąd_ładowania_modelu", 0.0), "Ulica": ("Błąd_ładowania_modelu", 0.0)}

    df = pd.DataFrame([data])

    # Przetwarzanie tekstu
    # Upewnij się, że kolumna 'Description' istnieje, zanim ją przetworzysz
    df['Description'] = df.get('Description', '').fillna('') # Użyj .get() i fillna('') na wypadek braku klucza/null
    df['description_clean'] = df['Description'].apply(lambda x: re.sub(r'[^a-ząęółśżżźćń ]', '', str(x).lower()))
    sequences = tokenizer.texts_to_sequences(df['description_clean'])
    X_text = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN)

    # Przetwarzanie cech numerycznych
    # Użyj .get() dla wszystkich kluczowych cech, zapewniając domyślne wartości
    price = data.get('Price', 0) if data.get('Price') is not None else 0 # Domyślnie 0, jeśli Price jest null
    area = data.get('Area', 1) if data.get('Area') is not None and data.get('Area') > 0 else 1 # Domyślnie 1 (lub inny niezerowy), jeśli Area jest null/zero/brak
    num_rooms = data.get('NumberOfRooms', 0) if data.get('NumberOfRooms') is not None else 0
    floor = data.get('Floor', 0) if data.get('Floor') is not None else 0
    floors = data.get('Floors', 0) if data.get('Floors') is not None else 0

    df_numeric_data = {'Area': area, 'Price': price, 'NumberOfRooms': num_rooms, 'Floor': floor, 'Floors': floors}
    df_numeric = pd.DataFrame([df_numeric_data])

    # Obliczanie Price_per_sqm - teraz z już pobranych i oczyszczonych wartości
    df_numeric['Price_per_sqm'] = df_numeric['Price'] / df_numeric['Area']
    # Poprawione użycie replace, aby uniknąć FutureWarning i działać na kopii, a potem przypisać
    df_numeric['Price_per_sqm'] = df_numeric['Price_per_sqm'].replace([np.inf, -np.inf], np.nan).fillna(0) # Uzupełniamy NaN zerem


    numeric_features_cols = ['Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'Price_per_sqm']
    # Upewnij się, że DataFrame numeryczny ma wszystkie oczekiwane kolumny przed transformacją
    for col in numeric_features_cols:
         if col not in df_numeric.columns:
             df_numeric[col] = 0 # Dodaj kolumnę z zerami jeśli jej brakuje

    X_numeric = numeric_pipeline.transform(df_numeric[numeric_features_cols])

    model_input = [X_text, X_numeric]
    # Obsługa potencjalnego błędu predykcji lub zwrócenia nieprawidłowej liczby wyjść przez model
    try:
        pred_results = model.predict(model_input, verbose=0)
        if len(pred_results) != 2:
             raise ValueError(f"Model zwrócił {len(pred_results)} wyjść, oczekiwano 2.")
        pred_dzielnica_proba, pred_ulica_proba = pred_results

        predicted_dzielnica = le_dzielnica.inverse_transform([np.argmax(pred_dzielnica_proba, axis=1)[0]])[0]
        conf_dzielnica = float(np.max(pred_dzielnica_proba, axis=1)[0])
        predicted_ulica = le_ulica.inverse_transform([np.argmax(pred_ulica_proba, axis=1)[0]])[0]
        conf_ulica = float(np.max(pred_ulica_proba, axis=1)[0])

        return {"Dzielnica": (predicted_dzielnica, conf_dzielnica), "Ulica": (predicted_ulica, conf_ulica)}

    except Exception as e:
        print(f"!!! BŁĄD podczas predykcji modelem lokalnym: {type(e).__name__} - {e}")
        return {"Dzielnica": ("Błąd_predykcji", 0.0), "Ulica": ("Błąd_predykcji", 0.0)}


# Zmieniona nazwa funkcji, aby odzwierciedlała nową funkcjonalność
def get_gpt_predictions(description):
    """Wywołuje API OpenAI w celu predykcji lokalizacji, ceny i stanu na podstawie opisu."""
    print(" -> Wewnątrz get_gpt_predictions...")
    if not client:
        print(" -> Klient OpenAI nie jest zainicjowany. Zwracam domyślne None.")
        # Zwracamy słownik ze wszystkimi oczekiwanymi kluczami, nawet jeśli są None
        return {"dzielnica": None, "ulica": None, "price": None, "condition": None}

    # Nowa, bardziej strukturalna lista wiadomości, z prośbą o więcej informacji
    messages = [
        {"role": "system", "content": f"""Jesteś ekspertem od rynku nieruchomości w Polsce. Twoim zadaniem jest precyzyjna identyfikacja najbardziej prawdopodobnej dzielnicy i ulicy (jeśli to oferta z Warszawy) oraz oszacowanie ceny i stanu nieruchomości na podstawie opisu.
Odpowiedz TYLKO w formacie JSON z kluczami "dzielnica", "ulica", "price" (jako liczba całkowita, jeśli to możliwe, bez symboli waluty i separatorów tysięcy) i "condition".
Dla klucza "condition" użyj jednej z następujących, ściśle określonych wartości: {", ".join(POSSIBLE_CONDITIONS)}.
Jeśli dzielnicy (dla Warszawy), ulicy (dla Warszawy), ceny lub stanu nie da się jednoznacznie ustalić na podstawie opisu, zwróć dla danego klucza null.
Pamiętaj o specjalnych przypadkach dla Warszawy: 'Saska Kępa' to część dzielnicy 'Praga-Południe'. 'Stary Mokotów' to część dzielnicy 'Mokotów'.
Cenę zwróć jako pojedynczą liczbę całkowitą (np. 1250000). Ignoruj walutę i separatory tysięcy z opisu. Jeśli cena jest podana w przybliżeniu lub w przedziale, podaj najlepsze oszacowanie jako liczbę całkowitą.
Nie dodawaj żadnego innego tekstu ani komentarzy poza samym obiektem JSON."""},
        {"role": "user", "content": "Przeanalizuj następujący opis nieruchomości i zidentyfikuj lokalizację (dzielnicę i ulicę, jeśli to oferta z Warszawy), cenę oraz stan:"},
        {"role": "user", "content": description} # Przekazujesz SAM OPIS jako oddzielną wiadomość
    ]

    print(" -> Klient OpenAI jest zainicjowany. Przygotowuję wiadomości do API (poszerzone o cenę i stan)...")

    # Możesz zmienić model na gpt-4o, pamiętając o kosztach
    model_to_use = "gpt-4o-mini"
    # model_to_use = "gpt-4o"
    print(f" -> Wywołuję API OpenAI ({model_to_use})...")

    try:
        completion = client.chat.completions.create(
            model=model_to_use,
            response_format={"type": "json_object"}, # To jest kluczowe do wymuszenia JSON
            messages=messages,
            # Dodatkowe parametry, jeśli potrzebne, np. temperature dla kontroli losowości
            # temperature=0.1,
        )
        print(" -> Sukces: Otrzymano odpowiedź z API.")
        raw_gpt_response_content = completion.choices[0].message.content
        print(f" -> Surowa odpowiedź GPT: {raw_gpt_response_content}")

        try:
            response_json = json.loads(raw_gpt_response_content)
            # Pobieramy wszystkie przewidziane wartości, używając .get()
            gpt_dzielnica = response_json.get("dzielnica")
            gpt_ulica = response_json.get("ulica")
            gpt_price_raw = response_json.get("price")
            gpt_condition_raw = response_json.get("condition")

            print(f" -> Zwracam z GPT: Dzielnica='{gpt_dzielnica}', Ulica='{gpt_ulica}', Cena='{gpt_price_raw}', Stan='{gpt_condition_raw}'")

            # Zwracamy słownik z wszystkimi wartościami
            return {
                "dzielnica": gpt_dzielnica,
                "ulica": gpt_ulica,
                "price": gpt_price_raw,
                "condition": gpt_condition_raw
            }

        except json.JSONDecodeError:
             print(f"!!! BŁĄD: API nie zwróciło poprawnego JSON: {raw_gpt_response_content}")
             # Zwróć słownik z None dla wszystkich pól w przypadku błędu parsowania
             return {"dzielnica": None, "ulica": None, "price": None, "condition": None}

    except Exception as e:
        print(f"!!! BŁĄD podczas komunikacji z API OpenAI w get_gpt_predictions: {type(e).__name__} - {e}")
        # Zwróć słownik z None dla wszystkich pól w przypadku błędu API
        return {"dzielnica": None, "ulica": None, "price": None, "condition": None}

def process_file(filepath):
    """
    Wczytuje plik JSON, przetwarza go lokalnym modelem i modelem GPT (jeśli oferta jest z Warszawy),
    ustala finalne predykcje (z uwzględnieniem reguł), zapisuje wyniki i przenosi plik.
    Obsługuje dwa formaty JSON: bezpośredni słownik oferty lub słownik z pojedynczym kluczem zawierającym listę ofert.
    """
    filename = os.path.basename(filepath)
    print(f"\nPrzetwarzanie pliku: {filename}")
    data = None # Inicjalizujemy data jako None na początku

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)

        # === Obsługa różnych formatów JSON ===
        # Sprawdź, czy wczytany JSON jest słownikiem i ma dokładnie jeden klucz na najwyższym poziomie
        if isinstance(loaded_json, dict) and len(loaded_json) == 1:
            top_level_key = list(loaded_json.keys())[0]
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
            # Jeśli JSON nie pasuje do formatu ze zmienną nazwą klucza, sprawdź stary format (bezpośredni słownik oferty)
            # Sprawdzamy obecność kluczy charakterystycznych dla danych oferty
            if isinstance(loaded_json, dict) and ('Description' in loaded_json or 'Price' in loaded_json or 'Area' in loaded_json or 'Location' in loaded_json):
                 data = loaded_json
                 print(f"Odczytano plik {filename} w starym formacie.")
            else:
                 print(f"Błąd formatu pliku {filename}: Nie rozpoznano struktury JSON.")

        # Jeśli zmienna data nadal jest None, oznacza to, że format pliku był niepoprawny
        if data is None:
             raise ValueError(f"Nie udało się poprawnie wczytać danych oferty z pliku {filename} z powodu nieznanego formatu JSON lub braku danych oferty.")
        # ==================================

        # Sprawdzenie czy oferta dotyczy Warszawy na podstawie klucza Location (używamy sparsowanego 'data')
        location_str = data.get('Location', '').lower() # Pobierz lokalizację, domyślnie pusty string
        is_warsaw_offer = 'warszawa' in location_str
        print(f"Oferta z {filename} {'jest' if is_warsaw_offer else 'NIE JEST'} z Warszawy (Lokalizacja: '{data.get('Location', 'brak')}').")

        # Wywołujemy predykcję lokalnego modelu (nawet jeśli oferta nie jest z Warszawy)
        local_predictions = predict_with_local_model(data)
        print(f"Predykcja lokalnego modelu: Dzielnica='{local_predictions['Dzielnica'][0]}', Ulica='{local_predictions['Ulica'][0]}'")

        # Pobieramy Description do przekazania do GPT. Upewniamy się, że klucz 'Description' istnieje.
        description_for_gpt = data.get('Description', None)
        if description_for_gpt is None:
             print(f"UWAGA: W danych oferty z pliku {filename} brak klucza 'Description' lub jego wartość to null.")
             description_for_gpt = "" # Przekazujemy pusty string zamiast None

        # Wywołujemy funkcję GPT (teraz zwraca słownik).
        # Przekazujemy pusty opis, jeśli oferta nie jest z Warszawy, aby GPT zwrócił null dla lokalizacji (zgodnie z prompem).
        # Pozwalamy GPT próbować przewidzieć cenę i stan nawet dla ofert spoza Warszawy,
        # bo prompt jest "od rynku nieruchomości w Polsce", a tylko lokalizacja jest ograniczona do Warszawy.
        gpt_results = get_gpt_predictions(description_for_gpt)

        gpt_dzielnica = gpt_results.get("dzielnica")
        gpt_ulica = gpt_results.get("ulica")
        gpt_price_raw = gpt_results.get("price")
        gpt_condition_raw = gpt_results.get("condition")

        print(f"Sugestia GPT: Dzielnica='{gpt_dzielnica}', Ulica='{gpt_ulica}', Cena='{gpt_price_raw}', Stan='{gpt_condition_raw}'")

        # --- Logika ustalania finalnych wartości ---

        # Lokalizacja - bierzemy predykcję GPT, jeśli oferta jest z Warszawy, GPT coś zwróciło I jest to znana nazwa
        # W przeciwnym razie bierzemy predykcję lokalnego modelu.
        final_dzielnica = local_predictions['Dzielnica'][0] # Domyślnie bierzemy z lokalnego modelu
        final_ulica = local_predictions['Ulica'][0]         # Domyślnie bierzemy z lokalnego modelu

        if is_warsaw_offer: # Nadpisujemy tylko jeśli oferta jest z Warszawy
             if gpt_dzielnica and le_dzielnica is not None and gpt_dzielnica in le_dzielnica.classes_:
                 print(f"GPT zweryfikował/poprawił dzielnicę na: '{gpt_dzielnica}'.")
                 final_dzielnica = gpt_dzielnica
             # else: lokalna predykcja dzielnicy zostaje, jeśli GPT nie zwróciło wartości lub zwróciło nieznaną

             if gpt_ulica and le_ulica is not None and gpt_ulica in le_ulica.classes_:
                 print(f"GPT zweryfikował/poprawił ulicę na: '{gpt_ulica}'.")
                 final_ulica = gpt_ulica
             # else: lokalna predykcja ulicy zostaje, jeśli GPT nie zwróciło wartości lub zwróciło nieznaną

        # Cena - Użyj predykcji GPT, jeśli jest liczbą. W przeciwnym razie użyj ceny z danych wejściowych. W przeciwnym razie null.
        final_price = None
        # Spróbuj przekonwertować surową cenę z GPT na liczbę (może przyjść jako string lub int/float)
        try:
            if gpt_price_raw is not None:
                final_price = float(gpt_price_raw)
                print(f"Ustalono cenę na podstawie predykcji GPT: {final_price}.")
        except (ValueError, TypeError):
            print(f"UWAGA: Nie udało się sparsować ceny z GPT '{gpt_price_raw}' jako liczbę.")
            final_price = None # Jeśli parsowanie się nie udało, cena z GPT jest nieważna

        # Jeśli cena z GPT była nieważna lub null, spróbuj wziąć cenę z danych wejściowych
        if final_price is None and data.get('Price') is not None:
             # Upewnij się, że cena z danych wejściowych jest liczbą
             try:
                 final_price = float(data['Price'])
                 print(f"Ustalono cenę na podstawie danych wejściowych: {final_price}.")
             except (ValueError, TypeError):
                 print(f"UWAGA: Cena z danych wejściowych '{data['Price']}' nie jest liczbą.")
                 final_price = None # Jeśli cena z danych wejściowych jest niepoprawna, pozostaw None


        # Stan mieszkania - Logika:
        # 1. Sprawdź rok budowy. Jeśli istnieje klucz BuiltYear i jego wartość < STAN_DEWELOPERSKI_ROK_GRANICZNY, stan to "Stan_Deweloperski".
        # 2. W przeciwnym razie, bierz predykcję GPT, jeśli pasuje do jednej z 4 kategorii.
        # 3. W przeciwnym razie, stan jest null/nieznany.

        final_condition = None
        built_year = data.get('BuiltYear') # Pobierz rok budowy z danych wejściowych

        if built_year is not None:
            try:
                # Próbujemy sparsować rok budowy jako liczbę całkowitą
                built_year_int = int(built_year)
                # Reguła: Stan Deweloperski jeśli rok budowy jest Mniejszy niż rok graniczny
                if built_year_int < STAN_DEWELOPERSKI_ROK_GRANICZNY:
                     final_condition = "Stan_Deweloperski"
                     print(f"Ustalono Stan_Deweloperski na podstawie roku budowy ({built_year_int} < {STAN_DEWELOPERSKI_ROK_GRANICZNY}).")
            except (ValueError, TypeError):
                print(f"UWAGA: Nie udało się sparsować roku budowy '{built_year}' z danych wejściowych jako liczbę. Reguła roku budowy nie zastosowana.")
                pass # Jeśli rok budowy jest niepoprawny, ignoruj tę regułę

        # Jeśli stan nie został ustalony na podstawie roku budowy, użyj predykcji GPT
        if final_condition is None:
            # Sprawdź, czy surowa predykcja stanu od GPT jest jedną z dozwolonych wartości z naszej globalnej listy
            if gpt_condition_raw in POSSIBLE_CONDITIONS:
                final_condition = gpt_condition_raw
                print(f"Ustalono stan na podstawie predykcji GPT: '{final_condition}'.")
            else:
                # Jeśli GPT zwróciło inną wartość lub null
                print(f"UWAGA: Surowa predykcja stanu od GPT '{gpt_condition_raw}' nie jest jedną z oczekiwanych kategorii ({', '.join(POSSIBLE_CONDITIONS)}) lub jest null.")
                final_condition = None # Możesz tu ustawić np. "Nieznany" jeśli wolisz

        # --- Wyświetlanie finalnych wyników w konsoli ---
        print("-" * 40)
        print(f"Wynik dla pliku: {filename}")
        print(f"  > Lokalizacja (Finalnie): Dzielnica='{final_dzielnica}', Ulica='{final_ulica}'")
        print(f"  > Cena (Finalnie): {final_price}")
        print(f"  > Stan (Finalnie): {final_condition}")
        print("-" * 40)


        # --- Dodawanie wyników do wynikowego JSONa ---
        # Zapisujemy zaktualizowany obiekt 'data' (słownik oferty)
        data['Prediction_Result'] = {
            "Final_Dzielnica": final_dzielnica,
            "Final_Ulica": final_ulica,
            "Final_Price": final_price, # Dodaj finalną cenę
            "Final_Condition": final_condition, # Dodaj finalny stan

            # Zachowujemy też surowe predykcje z lokalnego modelu i GPT dla pełności
            "Lokalny_Model_Dzielnica": local_predictions['Dzielnica'][0],
            "Lokalny_Model_Dzielnica_Prob": round(local_predictions['Dzielnica'][1], 4),
            "Lokalny_Model_Ulica": local_predictions['Ulica'][0],
            "Lokalny_Model_Ulica_Prob": round(local_predictions['Ulica'][1], 4),

            "GPT_Dzielnica_Raw": gpt_dzielnica,
            "GPT_Ulica_Raw": gpt_ulica,
            "GPT_Price_Raw": gpt_price_raw, # Dodaj surową cenę z GPT
            "GPT_Condition_Raw": gpt_condition_raw # Dodaj surowy stan z GPT
        }

        output_filepath = os.path.join(LOC_OUT_DIR, filename)
        with open(output_filepath, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)

        move(filepath, os.path.join(PROCESSED_DIR, filename))

    except Exception as e:
        print(f"BŁĄD podczas przetwarzania pliku {filename}: {type(e).__name__} - {e}") # Dodano typ błędu
        # W przypadku błędu, spróbuj zapisać plik z informacją o błędzie, jeśli obiekt data został wczytany
        if data is not None:
             data['Processing_Error'] = f"{type(e).__name__}: {e}"
             error_output_filepath = os.path.join(ERROR_DIR, filename)
             try:
                  with open(error_output_filepath, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
             except Exception as save_err:
                  print(f"!!! BŁĄD KRYTYCZNY: Nie udało się zapisać pliku z błędem {filename}: {save_err}")

        move(filepath, os.path.join(ERROR_DIR, filename))


def main_loop():
    """Główna pętla skryptu monitorująca folder wejściowy i przetwarzająca pliki."""
    print("\n--- Rozpoczynam pętlę główną skryptu ---")
    while True:
        try:
            # Pobierz listę plików w folderze LOC_IN, filtrując tylko pliki JSON
            files_to_process = [f for f in os.listdir(LOC_IN_DIR) if os.path.isfile(os.path.join(LOC_IN_DIR, f)) and f.lower().endswith('.json')]

            if files_to_process:
                print(f"Znaleziono {len(files_to_process)} plik(ów) JSON do przetworzenia w {LOC_IN_DIR}.")
                for filename in files_to_process:
                    process_file(os.path.join(LOC_IN_DIR, filename))
            #else:
                # Opcjonalne: print co jakiś czas, gdy brak plików
                # print(f"Brak plików JSON do przetworzenia w {LOC_IN_DIR}. Czekam...")

            time.sleep(5) # Odczekaj 5 sekund przed kolejnym sprawdzeniem

        except KeyboardInterrupt:
            print("\nZatrzymywanie skryptu na żądanie użytkownika...")
            break # Zakończ pętlę przy CTRL+C
        except Exception as e:
            # Obsługa nieoczekiwanych błędów w samej pętli głównej
            print(f"!!! BŁĄD w pętli głównej skryptu: {type(e).__name__} - {e}")
            time.sleep(10) # Poczekaj trochę dłużej po błędzie w pętli głównej

# --- Punkt wejścia skryptu ---
if __name__ == '__main__':
    setup() # Najpierw zainicjuj modele lokalne i klienta GPT
    # Uruchom pętlę główną tylko jeśli klient GPT został poprawnie zainicjowany LUB jeśli chcesz przetwarzać tylko lokalnie
    # Obecna logika uruchamia pętlę tylko jeśli klient GPT działa, co jest zgodne z tym, że skrypt jest "hybrydowy".
    # Jeśli chcesz uruchomić pętlę nawet bez działającego GPT (przetwarzając tylko lokalnie), zmień warunek na `if model is not None:`
    if client:
        main_loop()
    else:
        print("Nie można uruchomić pętli głównej, ponieważ klient OpenAI nie został poprawnie zainicjowany.")
        # Jeśli chcesz, możesz tu dodać opcję uruchomienia main_loop() tylko z lokalnym modelem
        # print("Uruchamiam pętlę główną tylko z funkcjonalnością modelu lokalnego (GPT wyłączone).")
        # main_loop() # Uruchom pętlę nawet bez klienta GPT - wymagałoby modyfikacji process_file do obsługi braku klienta GPT bez rzucania błędów