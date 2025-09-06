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
ARTIFACTS_DIR = 'Artifacts_Polska' 
LOC_IN_DIR = 'LOC_IN'
LOC_OUT_DIR = 'LOC_OUT'
PROCESSED_DIR = os.path.join(LOC_IN_DIR, 'processed')
ERROR_DIR = os.path.join(LOC_IN_DIR, 'error')

STAN_DEWELOPERSKI_ROK_GRANICZNY = 2024
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

# --- Zmienne globalne na wczytane artefakty ---
model, vectorizer, scaler = (None,) * 3
id_to_name, city_to_districts, district_to_streets = (None,) * 3
city_id_map, district_id_map, street_id_map = (None,) * 3
inv_city_id_map, inv_district_id_map, inv_street_id_map = (None,) * 3
all_known_locations = set()

def setup():
    """Inicjalizuje skrypt, ładuje model lokalizacji i wszystkie niezbędne artefakty."""
    global model, vectorizer, scaler, id_to_name, city_to_districts, district_to_streets, all_known_locations
    global city_id_map, district_id_map, street_id_map, inv_city_id_map, inv_district_id_map, inv_street_id_map
    
    print("--- Inicjalizacja skryptu ---")
    for dir_path in [LOC_IN_DIR, LOC_OUT_DIR, PROCESSED_DIR, ERROR_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        
    print(f"Ładowanie artefaktów z folderu: {ARTIFACTS_DIR}")
    try:
        model = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, 'location_prediction_model.h5'))
        vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, 'vectorizer.joblib'))
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
        
        city_id_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'city_id_map.joblib'))
        district_id_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'district_id_map.joblib'))
        street_id_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'street_id_map.joblib'))
        inv_city_id_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'inv_city_id_map.joblib'))
        inv_district_id_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'inv_district_id_map.joblib'))
        inv_street_id_map = joblib.load(os.path.join(ARTIFACTS_DIR, 'inv_street_id_map.joblib'))

        df_lok = pd.read_csv('lokalizacja.csv', sep=',', header=None, names=['Id', 'ParentId', 'Name', 'AdditionalName', 'FullName'], dtype=str)
        df_lok['Id'] = pd.to_numeric(df_lok['Id'], errors='coerce')
        df_lok['ParentId'] = df_lok['ParentId'].replace('\\N', pd.NA)
        df_lok['ParentId'] = pd.to_numeric(df_lok['ParentId'], errors='coerce')
        
        id_to_name = df_lok.set_index('Id')['Name'].to_dict()
        all_known_locations = set(id_to_name.values())
        
        city_to_districts = df_lok[df_lok['AdditionalName'].str.contains('Dzielnica|Osiedle', na=False)].groupby('ParentId')['Id'].apply(list).to_dict()
        district_to_streets = df_lok[df_lok['AdditionalName'] == 'Ulica'].groupby('ParentId')['Id'].apply(list).to_dict()

        print("Wszystkie artefakty załadowane pomyślnie.")
        return True
    except Exception as e:
        print(f"!!! BŁĄD KRYTYCZNY podczas ładowania artefaktów: {e}")
        return False

def predict_location_local(data):
    """Wykonuje predykcję lokalizacji (Miasto > Dzielnica > Ulica) za pomocą modelu lokalnego."""
    if not all([model, vectorizer, scaler]): return "Błąd ładowania modelu"
    df = pd.DataFrame([data])

    df['text_features'] = df.get('Title', '').fillna('') + ' ' + df.get('Description', '').fillna('')
    df['text_features'] = df['text_features'].apply(lambda x: re.sub(r'\s+', ' ', str(x).lower()))
    X_text = vectorizer.transform(df['text_features'])

    numeric_features = ['Area', 'Price']
    for col in numeric_features: df[col] = pd.to_numeric(df.get(col), errors='coerce').fillna(0)
    X_num = scaler.transform(df[numeric_features])
    
    pred_city, pred_district, pred_street = model.predict([X_text, X_num], verbose=0)
    
    city_pred_index = np.argmax(pred_city[0]); city_id = inv_city_id_map.get(city_pred_index, 0)
    
    valid_districts = city_to_districts.get(city_id, []); district_pred_index = np.argmax(pred_district[0])
    if valid_districts:
        mask = np.zeros_like(pred_district[0]); valid_indices = [district_id_map.get(d) for d in valid_districts if district_id_map.get(d) is not None]
        if valid_indices: mask[valid_indices] = 1; district_pred_index = np.argmax(pred_district[0] * mask)
    district_id = inv_district_id_map.get(district_pred_index, 0)

    valid_streets = district_to_streets.get(district_id, []); street_pred_index = np.argmax(pred_street[0])
    if valid_streets:
        mask = np.zeros_like(pred_street[0]); valid_indices = [street_id_map.get(s) for s in valid_streets if street_id_map.get(s) is not None]
        if valid_indices: mask[valid_indices] = 1; street_pred_index = np.argmax(pred_street[0] * mask)
    street_id = inv_street_id_map.get(street_pred_index, 0)

    city_name = id_to_name.get(city_id, "?")
    district_name = id_to_name.get(district_id, "?") if district_id != 0 else "?"
    street_name = id_to_name.get(street_id, "?") if street_id != 0 else "?"
    
    return f"{city_name} > {district_name} > {street_name}"

def get_gpt_predictions(description):
    """Wywołuje API OpenAI do predykcji lokalizacji, ceny i stanu."""
    if not client: return {"dzielnica": None, "ulica": None, "price": None, "condition": None}

    # ZMIANA: Usunięto z promptu ograniczenie do Warszawy
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
        if not all(k in data for k in ['Title', 'Description', 'Area', 'Price']): raise ValueError("Brak wymaganych kluczy.")

        local_pred_str = predict_location_local(data)
        print(f"  > Predykcja lokalnego modelu: {local_pred_str}")

        gpt_results = get_gpt_predictions(data['Description'])
        print(f"  > Sugestia GPT: Dzielnica='{gpt_results.get('dzielnica')}', Ulica='{gpt_results.get('ulica')}', Cena='{gpt_results.get('price')}', Stan='{gpt_results.get('condition')}'")

        # ZMIANA: Usunięto warunek 'is_warsaw_offer'. Logika działa teraz dla całej Polski.
        final_loc = local_pred_str
        parts = [p.strip() for p in local_pred_str.split('>')]
        gpt_dzielnica = gpt_results.get("dzielnica")
        gpt_ulica = gpt_results.get("ulica")
        
        # Weryfikujemy, czy sugestia GPT jest znaną lokalizacją
        if gpt_dzielnica and gpt_dzielnica in all_known_locations: parts[1] = gpt_dzielnica
        if gpt_ulica and gpt_ulica in all_known_locations: parts[2] = gpt_ulica
        final_loc = " > ".join(parts)

        final_price = data.get('Price')
        try:
            if gpt_results.get("price") is not None: final_price = float(gpt_results.get("price"))
        except (ValueError, TypeError): pass

        final_condition = None
        if data.get('BuiltYear'):
            try:
                if int(data.get('BuiltYear')) >= STAN_DEWELOPERSKI_ROK_GRANICZNY: final_condition = "Stan_Deweloperski"
            except: pass
        if final_condition is None and gpt_results.get("condition") in POSSIBLE_CONDITIONS:
            final_condition = gpt_results.get("condition")

        print("-" * 20)
        print(f"  > Finalna lokalizacja: {final_loc}")
        print(f"  > Finalna cena: {final_price}")
        print(f"  > Finalny stan: {final_condition}")
        print("-" * 20)

        data['Prediction_Result'] = {
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
                for filename in files: process_file(os.path.join(LOC_IN_DIR, filename))
            time.sleep(5)
        except KeyboardInterrupt: print("\nZatrzymywanie..."); break
        except Exception as e: print(f"BŁĄD w pętli głównej: {e}"); time.sleep(10)

if __name__ == '__main__':
    if setup() and client:
        main_loop()
    else:
        print("Nie można uruchomić pętli głównej. Sprawdź, czy artefakty są na miejscu i czy klucz API OpenAI jest poprawny.")