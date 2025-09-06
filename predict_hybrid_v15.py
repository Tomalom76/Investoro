import os
import json
import time
import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from shutil import move
from openai import OpenAI
import gc

# --- Konfiguracja ---
# ZMIANA V15: Zaktualizowano ścieżkę do artefaktów z wersji 15
ARTIFACTS_DIR = 'artifacts_v15_final' 
LOC_IN_DIR = 'LOC_IN'
LOC_OUT_DIR = 'LOC_OUT'
PROCESSED_DIR = os.path.join(LOC_IN_DIR, 'processed')
ERROR_DIR = os.path.join(LOC_IN_DIR, 'error')

POSSIBLE_CONDITIONS = ["Do_Remontu", "Dobry", "Po_Remoncie", "Stan_Deweloperski"]

# ==============================================================================
# === WPISZ SWÓJ KLUCZ API OpenAI PONIŻEJ ===
# ==============================================================================
OPENAI_API_KEY = "sk-..."  # WAŻNE: Wklej tutaj swój klucz API
# ==============================================================================

client = None
try:
    if not OPENAI_API_KEY or "..." in OPENAI_API_KEY:
        raise ValueError("Klucz API OpenAI nie został ustawiony w skrypcie.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("Klient OpenAI zainicjowany pomyślnie.")
except Exception as e:
    print(f"!!! BŁĄD KRYTYCZNY podczas inicjalizacji klienta OpenAI: {e}")

# --- ZMIANA V15: Zaktualizowano listę zmiennych globalnych ---
model, vectorizer, scaler, imputer = (None,) * 4
id_to_name, hierarchy_map = (None,) * 2
city_map, district_map, subdistrict_map, street_map = (None,) * 4
inv_district_map, inv_subdistrict_map, inv_street_map = (None,) * 3
all_known_locations = set()

def setup():
    """Inicjalizuje skrypt, ładuje model v15 i wszystkie niezbędne artefakty."""
    # ZMIANA V15: Przypisanie do zmiennych globalnych
    global model, vectorizer, scaler, imputer, id_to_name, hierarchy_map, all_known_locations
    global city_map, district_map, subdistrict_map, street_map
    global inv_district_map, inv_subdistrict_map, inv_street_map
    
    print("--- Inicjalizacja skryptu (wersja v15) ---")
    for dir_path in [LOC_IN_DIR, LOC_OUT_DIR, PROCESSED_DIR, ERROR_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        
    print(f"Ładowanie artefaktów z folderu: {ARTIFACTS_DIR}")
    try:
        # ZMIANA V15: Upewnij się, że nazwa pliku modelu jest poprawna. Notebook zapisuje go jako .keras
        # W notebooku jest literówka i zapisuje jako `best_location_model_v14.keras` w folderze v14.
        # Założyłem, że docelowo pliki będą w `artifacts_v15_final` i model będzie miał poprawną nazwę.
        # Jeśli nie, zmień poniższą ścieżkę!
        model_path = os.path.join(ARTIFACTS_DIR, 'best_location_model_v15.keras') 
        if not os.path.exists(model_path):
             # Awaryjnie sprawdzam nazwę z notebooka, jeśli istnieje
             fallback_path = os.path.join(ARTIFACTS_DIR, 'best_location_model_v14.keras')
             if os.path.exists(fallback_path):
                 print(f"OSTRZEŻENIE: Nie znaleziono {model_path}. Używam awaryjnej ścieżki: {fallback_path}")
                 model_path = fallback_path
             else:
                 raise FileNotFoundError(f"Nie znaleziono pliku modelu ani w {model_path}, ani w {fallback_path}")
        
        model = tf.keras.models.load_model(model_path)
        vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, 'vectorizer.joblib'))
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
        imputer = joblib.load(os.path.join(ARTIFACTS_DIR, 'imputer.joblib')) # NOWY artefakt
        hierarchy_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'hierarchy_map.joblib')) # NOWY artefakt

        # ZMIANA V15: Ładowanie nowych map
        city_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'city_map.joblib'))
        district_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'district_map.joblib'))
        subdistrict_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'subdistrict_map.joblib'))
        street_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'street_map.joblib'))
        
        # ZMIANA V15: Tworzenie map odwrotnych
        inv_district_map = {v: k for k, v in district_map.items()}
        inv_subdistrict_map = {v: k for k, v in subdistrict_map.items()}
        inv_street_map = {v: k for k, v in street_map.items()}

        df_lok = pd.read_csv('lokalizacja.csv', sep=',', header=None, names=['Id', 'ParentId', 'Name', 'Type', 'FullName'], dtype=str)
        df_lok['Id'] = pd.to_numeric(df_lok['Id'], errors='coerce')
        id_to_name = df_lok.set_index('Id')['Name'].to_dict()
        all_known_locations = set(id_to_name.values())
        
        print("Wszystkie artefakty (v15) załadowane pomyślnie.")
        return True
    except Exception as e:
        print(f"!!! BŁĄD KRYTYCZNY podczas ładowania artefaktów: {e}")
        return False

# ZMIANA V15: Nowa funkcja post-processingu skopiowana z notebooka
def find_best_consistent_path(top_k_districts, top_k_subdistricts, top_k_streets, local_hierarchy_map):
    """Iteruje po top-k predykcjach, aby znaleźć pierwszą spójną hierarchicznie ścieżkę."""
    for dist_id in top_k_districts:
        for sub_id in top_k_subdistricts:
            # Sprawdzenie spójności pod-dzielnicy z dzielnicą
            if sub_id != 0 and local_hierarchy_map.get(sub_id) != dist_id:
                continue
            for street_id in top_k_streets:
                # Sprawdzenie spójności ulicy z pod-dzielnicą (lub dzielnicą, jeśli pod-dzielnicy brak)
                if street_id != 0:
                    expected_parent = sub_id if sub_id != 0 else dist_id
                    if local_hierarchy_map.get(street_id) != expected_parent:
                        continue
                # Znaleziono pierwszą spójną ścieżkę
                return dist_id, sub_id, street_id
    # Jeśli żadna spójna ścieżka nie została znaleziona, zwróć top-1
    return top_k_districts[0], top_k_subdistricts[0], top_k_streets[0]


# ZMIANA V15: Całkowicie przebudowana funkcja predykcji
def predict_location_local(data):
    """Wykonuje predykcję lokalizacji (Dzielnica > Pod-dzielnica > Ulica) za pomocą modelu v15."""
    if not all([model, vectorizer, scaler, imputer, hierarchy_map]): 
        return "Błąd krytyczny: Model lub artefakty v15 niezaładowane"

    # 1. Przygotowanie wejść dla modelu
    df = pd.DataFrame([data])
    
    # Wejście tekstowe
    df['text_features'] = df.get('Title', '').fillna('') + ' ' + df.get('Description', '').fillna('')
    df['text_features'] = df['text_features'].apply(lambda x: re.sub(r'\s+', ' ', str(x).lower()))
    X_text = vectorizer.transform(df['text_features'])

    # Wejście numeryczne (NOWE CECHY)
    df['area'] = pd.to_numeric(df.get('Area'), errors='coerce')
    df['price'] = pd.to_numeric(df.get('Price'), errors='coerce')
    df['price_per_meter'] = (df['price'] / df['area']).replace([np.inf, -np.inf], 0)
    
    # UWAGA: Cechy 'is_central_subdistrict' nie można obliczyć w czasie predykcji,
    # ponieważ wymaga ona znajomości prawdziwej pod-dzielnicy. To potencjalny błąd w logice
    # notebooka. Używamy tu bezpiecznej wartości domyślnej (0).
    df['is_central_subdistrict'] = 0 
    
    numeric_features = ['area', 'price', 'price_per_meter', 'is_central_subdistrict']
    X_num = scaler.transform(imputer.transform(df[numeric_features]))
    
    # Wejście kategoryczne - ID miasta
    location_path = data.get('locationPath', '')
    try:
        path_ids = [int(p) for p in location_path.split(',') if p and p.isdigit()]
        city_id = path_ids[3] if len(path_ids) > 3 else 0 # Miasto to 4. element
    except (ValueError, IndexError):
        city_id = 0

    if city_id == 0 or city_id not in city_map:
        return f"Błąd: Nie można zidentyfikować miasta z locationPath: '{location_path}'"
    
    city_index = city_map.get(city_id)
    X_city_input = np.array([[city_index]])

    # 2. Predykcja modelem (oczekujemy 3 wyjść)
    predictions = model.predict([X_text.toarray(), X_num, X_city_input], verbose=0)
    pred_district_probs, pred_subdistrict_probs, pred_street_probs = predictions[0][0], predictions[1][0], predictions[2][0]

    # 3. Logika post-processingu z Top-K (NOWOŚĆ)
    k = 5
    top_k_district_indices = np.argsort(pred_district_probs)[::-1][:k]
    top_k_subdistrict_indices = np.argsort(pred_subdistrict_probs)[::-1][:k]
    top_k_street_indices = np.argsort(pred_street_probs)[::-1][:k]

    # Konwersja indeksów na oryginalne ID
    top_k_district_ids = [inv_district_map.get(idx, 0) for idx in top_k_district_indices]
    top_k_subdistrict_ids = [inv_subdistrict_map.get(idx, 0) for idx in top_k_subdistrict_indices]
    top_k_street_ids = [inv_street_map.get(idx, 0) for idx in top_k_street_indices]

    # Znalezienie najlepszej spójnej ścieżki
    final_district, final_subdistrict, final_street = find_best_consistent_path(
        top_k_district_ids, top_k_subdistrict_ids, top_k_street_ids, hierarchy_map
    )

    # 4. Zwrócenie wyniku
    city_name = id_to_name.get(city_id, "?")
    district_name = id_to_name.get(final_district, "?") if final_district != 0 else "?"
    subdistrict_name = id_to_name.get(final_subdistrict, "?") if final_subdistrict != 0 else "?"
    street_name = id_to_name.get(final_street, "?") if final_street != 0 else "?"
    
    return f"{city_name} > {district_name} > {subdistrict_name} > {street_name}"

# Ta funkcja nie wymaga zmian
def get_gpt_predictions(description):
    """Wywołuje API OpenAI do predykcji lokalizacji, ceny i stanu."""
    if not client: return {"dzielnica": None, "ulica": None, "price": None, "condition": None}

    messages = [
        {"role": "system", "content": f"""Jesteś ekspertem od rynku nieruchomości w Polsce. Zidentyfikuj dzielnicę i ulicę (jeśli to możliwe), oszacuj cenę i stan. Odpowiedz TYLKO w JSON z kluczami "dzielnica", "ulica", "price" (int), "condition" (jedna z: {", ".join(POSSIBLE_CONDITIONS)}). Jeśli czegoś nie wiesz, zwróć null."""},
        {"role": "user", "content": description}
    ]
    try:
        completion = client.chat.completions.create(model="gpt-4o-mini", response_format={"type": "json_object"}, messages=messages)
        response_json = json.loads(completion.choices[0].message.content)
        return response_json
    except Exception as e:
        print(f"!!! BŁĄD podczas komunikacji z API OpenAI: {e}")
        return {"dzielnica": None, "ulica": None, "price": None, "condition": None}

def process_file(filepath):
    filename = os.path.basename(filepath)
    print(f"\n--- Przetwarzanie pliku: {filename} ---")
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        if not all(k in data for k in ['Title', 'Description', 'Area', 'Price', 'locationPath']): 
            raise ValueError("Brak wymaganych kluczy (w tym locationPath).")

        local_pred_str = predict_location_local(data)
        print(f"  > Predykcja lokalnego modelu v15: {local_pred_str}")

        # Hybrydowa logika z GPT - nie wymaga większych zmian
        gpt_results = get_gpt_predictions(data['Description'])
        print(f"  > Sugestia GPT: Dzielnica='{gpt_results.get('dzielnica')}', Ulica='{gpt_results.get('ulica')}', Cena='{gpt_results.get('price')}', Stan='{gpt_results.get('condition')}'")

        final_loc = local_pred_str
        if not local_pred_str.startswith("Błąd:"):
            parts = [p.strip() for p in local_pred_str.split('>')]
            gpt_dzielnica = gpt_results.get("dzielnica")
            gpt_ulica = gpt_results.get("ulica")
            
            # ZMIANA V15: Dostosowanie do 4-poziomowej lokalizacji
            if len(parts) == 4:
                # Weryfikujemy, czy sugestia GPT jest znaną lokalizacją i podmieniamy
                if gpt_dzielnica and gpt_dzielnica in all_known_locations: parts[1] = gpt_dzielnica
                if gpt_ulica and gpt_ulica in all_known_locations: parts[3] = gpt_ulica
                final_loc = " > ".join(parts)

        final_price = data.get('Price')
        try:
            if gpt_results.get("price") is not None: final_price = float(gpt_results.get("price"))
        except (ValueError, TypeError): pass

        final_condition = None
        # Logika dla BuiltYear jest specyficzna - zachowuję
        if 'BuiltYear' in data and data['BuiltYear']:
            try:
                # Załóżmy, że rok graniczny to 2024
                if int(data['BuiltYear']) >= 2024: final_condition = "Stan_Deweloperski"
            except: pass
        if final_condition is None and gpt_results.get("condition") in POSSIBLE_CONDITIONS:
            final_condition = gpt_results.get("condition")

        print("-" * 20)
        print(f"  > Finalna lokalizacja: {final_loc}")
        print(f"  > Finalna cena: {final_price}")
        print(f"  > Finalny stan: {final_condition}")
        print("-" * 20)

        data['Prediction_Result_v15'] = {
            "Final_Location": final_loc, "Final_Price": final_price, "Final_Condition": final_condition,
            "Local_Model_Raw": local_pred_str, "GPT_Dzielnica_Raw": gpt_results.get("dzielnica"),
            "GPT_Ulica_Raw": gpt_results.get("ulica"), "GPT_Price_Raw": gpt_results.get("price"),
            "GPT_Condition_Raw": gpt_results.get("condition")
        }
        with open(os.path.join(LOC_OUT_DIR, filename), 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
        move(filepath, os.path.join(PROCESSED_DIR, filename))

    except Exception as e:
        print(f"BŁĄD podczas przetwarzania pliku {filename}: {e}")
        move(filepath, os.path.join(ERROR_DIR, filename))

def main_loop():
    print("\n--- Rozpoczynam pętlę główną ---")
    while True:
        try:
            files = [f for f in os.listdir(LOC_IN_DIR) if f.lower().endswith('.json') and os.path.isfile(os.path.join(LOC_IN_DIR, f))]
            if files:
                for filename in files: 
                    process_file(os.path.join(LOC_IN_DIR, filename))
                    gc.collect()
            time.sleep(5)
        except KeyboardInterrupt: 
            print("\nZatrzymywanie...")
            break
        except Exception as e: 
            print(f"BŁĄD w pętli głównej: {e}")
            time.sleep(10)

if __name__ == '__main__':
    if setup() and client:
        main_loop()
    else:
        print("Nie można uruchomić pętli głównej. Sprawdź, czy artefakty v15 są na miejscu i czy klucz API OpenAI jest poprawny.")