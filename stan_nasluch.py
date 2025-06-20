import pandas as pd
import mlflow
from pycaret.classification import setup, pull, compare_models, create_model, tune_model, finalize_model, save_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
import numpy as np
import os
import time
import shutil
import joblib
import logging # Dodajemy logging
import sys # Dodajemy sys dla loggingu

# --- Konfiguracja ---
DATA_INPUT_DIR_TRAINING = "DATA_STAN"
ARTIFACTS_OUTPUT_DIR = "DATA_STAN_OUT"
PROCESSED_TRAINING_DATA_DIR_NAME = "PROCESSED_DATA_STAN_TRAINING"

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME_TRAINING = 'Investoro_Stan_Mieszkania_Skrypt_Treningowy_Standalone' # Inna nazwa dla odróżnienia

MODEL_FILENAME_OUTPUT = "model_LGBM_stan_final.pkl"
VECTORIZER_FILENAME_OUTPUT = "count_vectorizer.joblib"
INPUT_CSV_FILENAME = "data.csv"

# Konfiguracja loggingu
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# --- Funkcje pomocnicze (takie same jak poprzednio) ---
def preprocess_data_for_training(df_input_original_unmodified):
    logging.info("Rozpoczynam preprocessing danych wejściowych dla treningu...")
    df = df_input_original_unmodified.copy()

    df['BuiltYear'] = pd.to_datetime(df['BuiltYear'], format='%Y', errors='coerce')
    
    initial_rows = len(df)
    df = df.dropna(subset=['Description'])
    logging.info(f"Po dropna na Description: {len(df)} wierszy (usunięto {initial_rows - len(df)}).")
    if df.empty: logging.error("BŁĄD: Brak danych po dropna Description."); return None, None
    initial_rows = len(df)

    df = df.dropna(subset=['Location'])
    logging.info(f"Po dropna na Location: {len(df)} wierszy (usunięto {initial_rows - len(df)}).")
    if df.empty: logging.error("BŁĄD: Brak danych po dropna Location."); return None, None
    initial_rows = len(df)
    
    df = df.dropna(subset=['BuildingCondition'])
    logging.info(f"Po dropna na BuildingCondition (target): {len(df)} wierszy (usunięto {initial_rows - len(df)}).")
    if df.empty: logging.error("BŁĄD: Brak danych po dropna BuildingCondition."); return None, None
    
    df_c = df.copy()

    df_c['Description'] = df_c['Description'].str.slice(0, 3000)
    vectorizer = CountVectorizer(max_features=1500)
    logging.info("Dopasowuję CountVectorizer i transformuję Description...")
    try:
        X_bow = vectorizer.fit_transform(df_c["Description"])
    except Exception as e:
        logging.error(f"BŁĄD podczas fit_transform w CountVectorizer: {e}")
        return None, None
        
    df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out(), index=df_c.index) 
    
    df_c_processed = df_c.drop(columns=['Description']).reset_index(drop=True)
    df_bow_processed = df_bow.reset_index(drop=True) # df_bow był tworzony z indeksem df_c, który mógł nie być ciągły
    
    # Upewnij się, że df_bow_processed ma tyle samo wierszy co df_c_processed
    # To powinno być zagwarantowane, jeśli X_bow powstało z df_c['Description'], a oba potem resetujemy
    if len(df_c_processed) != len(df_bow_processed):
        logging.error(f"BŁĄD: Niezgodność liczby wierszy przed concat! df_c_processed: {len(df_c_processed)}, df_bow_processed: {len(df_bow_processed)}")
        return None, None

    df_prepared = pd.concat([df_c_processed, df_bow_processed], axis=1)
    logging.info(f"Kształt po BoW: {df_prepared.shape}")

    if 'Price' in df_prepared.columns and not df_prepared['Price'].isnull().all() and len(df_prepared) > 1:
        Q1_price = df_prepared["Price"].quantile(0.25)
        Q3_price = df_prepared["Price"].quantile(0.75)
        IQR_price = Q3_price - Q1_price
        lower_bound_price = Q1_price - 1.5 * IQR_price
        upper_bound_price = Q3_price + 1.5 * IQR_price
        df_prep_p = df_prepared[~((df_prepared["Price"] < lower_bound_price) | (df_prepared["Price"] > upper_bound_price))].copy()
    else:
        df_prep_p = df_prepared.copy()
    logging.info(f"Kształt po outlierach Price: {df_prep_p.shape}")
    if df_prep_p.empty: logging.error("BŁĄD: Brak danych po outlierach Price."); return None, None

    if 'PricePerSquareMeter' in df_prep_p.columns and not df_prep_p['PricePerSquareMeter'].isnull().all() and len(df_prep_p) > 1:
        Q1_ppsm = df_prep_p["PricePerSquareMeter"].quantile(0.25)
        Q3_ppsm = df_prep_p["PricePerSquareMeter"].quantile(0.75)
        IQR_ppsm = Q3_ppsm - Q1_ppsm
        lower_bound_ppsm = Q1_ppsm - 1.5 * IQR_ppsm
        upper_bound_ppsm = Q3_ppsm + 1.5 * IQR_ppsm
        df_prep_a = df_prep_p[~((df_prep_p["PricePerSquareMeter"] < lower_bound_ppsm) | (df_prep_p["PricePerSquareMeter"] > upper_bound_ppsm))].copy()
    else:
        df_prep_a = df_prep_p.copy()
    logging.info(f"Kształt po outlierach PricePerSquareMeter: {df_prep_a.shape}")
    if df_prep_a.empty: logging.error("BŁĄD: Brak danych po outlierach PPSM."); return None, None
    
    if 'Area' in df_prep_a.columns and not df_prep_a['Area'].isnull().all() and len(df_prep_a) > 1:
        Q1_area = df_prep_a["Area"].quantile(0.25)
        Q3_area = df_prep_a["Area"].quantile(0.75)
        IQR_area = Q3_area - Q1_area
        lower_bound_area = Q1_area - 1.5 * IQR_area
        upper_bound_area = Q3_area + 1.5 * IQR_area
        df_prepared2 = df_prep_a[~((df_prep_a["Area"] < lower_bound_area) | (df_prep_a["Area"] > upper_bound_area))].copy()
    else:
        df_prepared2 = df_prep_a.copy()
    logging.info(f"Kształt po outlierach Area: {df_prepared2.shape}")
    if df_prepared2.empty: logging.error("BŁĄD: Brak danych po outlierach Area."); return None, None

    df_prepared3 = df_prepared2.copy() 
    df_prepared3 = df_prepared3.dropna(subset=['BuildingCondition'])
    if df_prepared3.empty: logging.error("BŁĄD: Brak danych po dodatkowym dropna na BC w df_prepared3."); return None, None

    df_prepared3.loc[:,'BuiltYear'] = pd.to_datetime(df_prepared3['BuiltYear'], format='%Y', errors='coerce')
    df_prepared3.loc[df_prepared3['TypeOfMarket'] == 'pierwotny', 'BuildingCondition'] = 'DEVELOPER_STATE'
    df_prepared3.loc[df_prepared3['BuiltYear'].dt.year >= 2024, 'BuildingCondition'] = 'DEVELOPER_STATE'
    condition_otodom_after_renovation = (df_prepared3['Link'].str.contains('otodom', case=False, na=False)) & \
                                        (df_prepared3['BuildingCondition'] == 'AFTER_RENOVATION')
    df_prepared3 = df_prepared3[~condition_otodom_after_renovation].copy()
    logging.info(f"Kształt po modyfikacjach BC: {df_prepared3.shape}")
    if df_prepared3.empty: logging.error("BŁĄD: Brak danych po modyfikacjach BC."); return None, None

    logging.info("Rozpoczynam ręczne balansowanie klas...")
    class_counts = df_prepared3['BuildingCondition'].value_counts()
    logging.info(f"Liczebność klas przed zbalansowaniem:\n{class_counts}")
    
    if class_counts.empty or len(class_counts) < 2 or class_counts.min() == 0:
        logging.error("BŁĄD: Niewystarczająca liczba klas lub próbek do zbalansowania.")
        return None, None
        
    min_count = class_counts.min()
    logging.info(f"Balansowanie klas przez undersampling do: {min_count} próbek na klasę.")
    dfs_balanced_list = []
    for condition_class in class_counts.index:
        df_condition = df_prepared3[df_prepared3['BuildingCondition'] == condition_class]
        df_condition_resampled = resample(df_condition,
                                        replace=False,
                                        n_samples=min_count,
                                        random_state=42)
        dfs_balanced_list.append(df_condition_resampled)

    if not dfs_balanced_list:
        logging.error("BŁĄD: Nie udało się stworzyć zbalansowanych próbek.")
        return None, None

    df_balanced_for_setup = pd.concat(dfs_balanced_list).reset_index(drop=True)
    logging.info(f"\nLiczebność klas PO zbalansowaniu:\n{df_balanced_for_setup['BuildingCondition'].value_counts()}")
    logging.info(f"Kształt df_balanced_for_setup: {df_balanced_for_setup.shape}")
    if df_balanced_for_setup.empty: logging.error("BŁĄD: df_balanced_for_setup jest pusty."); return None, None

    string_conversion_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber']
    if 'StreetNumber' in df_balanced_for_setup.columns: 
        string_conversion_cols.append('StreetNumber')

    logging.info(f"Konwertuję kolumny na string w df_balanced_for_setup: {string_conversion_cols}")
    for col in string_conversion_cols:
        if col in df_balanced_for_setup.columns:
            if pd.api.types.is_numeric_dtype(df_balanced_for_setup[col]):
                df_balanced_for_setup.loc[:, col] = df_balanced_for_setup[col].fillna(-1).astype(int).astype(str) 
            else:
                df_balanced_for_setup.loc[:, col] = df_balanced_for_setup[col].fillna('NA_placeholder').astype(str)
        else:
            logging.warning(f"OSTRZEŻENIE: Kolumna '{col}' nie znaleziona w df_balanced_for_setup do konwersji na string.")
            
    logging.info("Zakończono preprocessing dla treningu.")
    return df_balanced_for_setup, vectorizer


def train_pycaret_model(df_train, experiment_name):
    if df_train is None or df_train.empty:
        logging.error("BŁĄD: Dane treningowe (df_train) są puste lub None. Nie można trenować modelu.")
        return None, None

    logging.info("Konfiguruję eksperyment PyCaret (setup)...")
    
    categorical_features_initial = [
        'BuildingType', 'TypeOfMarket', 'OwnerType', 'Type', 'OfferFrom',
        'VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber',
        'RegionNumber', 'Location', 'StreetNumber'
    ]
    numeric_features_initial = [
        'Area', 'Price', 'NumberOfRooms', 'Floor', 'Floors', 'CommunityScore'
    ]
    date_features_initial = ['BuiltYear']

    categorical_features_to_use = [col for col in categorical_features_initial if col in df_train.columns]
    numeric_features_to_use = [col for col in numeric_features_initial if col in df_train.columns]
    date_features_to_use = [col for col in date_features_initial if col in df_train.columns]
    categorical_features_to_use = [col for col in categorical_features_to_use if col not in numeric_features_to_use]

    ignore_features_list_notebook = [
        'SaleId', 'OriginalId', 'PortalId', 'Title', 
        'OfferPrice', 'RealPriceAfterRenovation', 'OriginalPrice',
        'PricePerSquareMeter', 'DateAddedToDatabase', 'DateAdded',
        'DateLastModification', 'DateLastRaises', 'NewestDate',
        'AvailableFrom', 'Link', 'Phone', 'MainImage', 'OtherImages',
        'NumberOfDuplicates', 'NumberOfRaises', 'NumberOfModifications',
        'IsDuplicatePriceLower', 'IsDuplicatePrivateOwner', 'Score', 'ScorePrecision',
        'NumberOfCommunityComments', 'NumberOfCommunityOpinions', 'Archive',
        'SubRegionNumber', 'EncryptedId','id' 
    ]
    explicit_features = set(categorical_features_to_use + numeric_features_to_use + date_features_to_use)
    ignore_features_list_final = [col for col in ignore_features_list_notebook if col in df_train.columns and col not in explicit_features]

    logging.info(f"Setup - Cechy kategoryczne: {categorical_features_to_use}")
    logging.info(f"Setup - Cechy numeryczne: {numeric_features_to_use}")
    logging.info(f"Setup - Cechy daty: {date_features_to_use}")
    logging.info(f"Setup - Cechy ignorowane: {ignore_features_list_final}")

    exp = setup(
        data=df_train,
        target='BuildingCondition',
        verbose=False, 
        session_id=123,
        log_experiment=True,
        experiment_name=experiment_name,
        log_data=False, 
        log_plots=False, 
        fix_imbalance=False, 
        categorical_features=categorical_features_to_use,
        numeric_features=numeric_features_to_use,
        date_features=date_features_to_use,
        ignore_features=ignore_features_list_final,
        ordinal_features={'BuildingType': ['Pozostałe', 'Blok', 'Apartamentowiec', 'Kamienica']},
    )
    
    logging.info("Tworzę model LightGBM...")
    lgbm_model = create_model('lightgbm', verbose=False)
    model_to_finalize = lgbm_model 
    
    logging.info("Finalizuję model...")
    final_model = finalize_model(model_to_finalize)
    logging.info("Model sfinalizowany.")
    
    return final_model, exp

# --- Główna logika skryptu treningowego ---
def main():
    logging.info("Uruchamiam skrypt treningowy stan_nasluch.py...")
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) # Można odkomentować, jeśli ten skrypt też ma logować do MLflow

    os.makedirs(DATA_INPUT_DIR_TRAINING, exist_ok=True)
    os.makedirs(ARTIFACTS_OUTPUT_DIR, exist_ok=True)
    processed_training_data_path = os.path.join(DATA_INPUT_DIR_TRAINING, PROCESSED_TRAINING_DATA_DIR_NAME)
    os.makedirs(processed_training_data_path, exist_ok=True)

    logging.info(f"Nasłuchuję folderu: {DATA_INPUT_DIR_TRAINING} dla pliku {INPUT_CSV_FILENAME}")

    # Zmiana: Pętla wykonuje się tylko raz, jeśli plik istnieje, lub czeka i się kończy, jeśli go nie ma
    # Można też zrobić tak, że skrypt kończy się od razu, jeśli pliku nie ma.
    # Ta wersja będzie nasłuchiwać, ale zakończy się po pierwszym udanym przetworzeniu.
    
    file_processed_this_run = False
    
    while not file_processed_this_run: # Pętla do momentu przetworzenia pliku
        try:
            input_file_path_training = os.path.join(DATA_INPUT_DIR_TRAINING, INPUT_CSV_FILENAME)
            
            if os.path.exists(input_file_path_training):
                logging.info(f"Znaleziono plik {INPUT_CSV_FILENAME} w {DATA_INPUT_DIR_TRAINING}")
                
                df_input_original = pd.read_csv(input_file_path_training, sep=',')
                logging.info(f"Wczytano {input_file_path_training}, kształt: {df_input_original.shape}")

                df_train_data, vectorizer_fitted_on_train = preprocess_data_for_training(df_input_original)

                if df_train_data is None or vectorizer_fitted_on_train is None:
                    logging.error("Preprocessing danych treningowych nie powiódł się. Zatrzymuję przetwarzanie tego pliku.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(input_file_path_training, os.path.join(processed_training_data_path, f"{INPUT_CSV_FILENAME}_preprocess_error_{timestamp}.csv"))
                    logging.info(f"Przeniesiono {input_file_path_training} do {processed_training_data_path} jako plik z błędem preprocessingu.")
                    # Jeśli chcemy, aby skrypt się zatrzymał całkowicie po błędzie, można dodać:
                    # file_processed_this_run = True # Aby wyjść z pętli while
                    # break
                    time.sleep(10) # Jeśli ma próbować dalej, ale z przerwą
                    continue 

                final_model_trained, pycaret_exp_instance = train_pycaret_model(df_train_data, MLFLOW_EXPERIMENT_NAME_TRAINING)
                
                if final_model_trained is None:
                    logging.error("Trening modelu nie powiódł się. Zatrzymuję przetwarzanie tego pliku.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(input_file_path_training, os.path.join(processed_training_data_path, f"{INPUT_CSV_FILENAME}_model_train_error_{timestamp}.csv"))
                    logging.info(f"Przeniesiono {input_file_path_training} do {processed_training_data_path} jako plik z błędem treningu modelu.")
                    # file_processed_this_run = True 
                    # break
                    time.sleep(10)
                    continue

                model_output_path_final = os.path.join(ARTIFACTS_OUTPUT_DIR, MODEL_FILENAME_OUTPUT)
                try:
                    save_model(final_model_trained, model_output_path_final[:-4]) 
                    logging.info(f"Model treningowy zapisany w: {model_output_path_final}")
                except Exception as e:
                    logging.error(f"BŁĄD podczas zapisu modelu treningowego: {e}")

                vectorizer_output_path_final = os.path.join(ARTIFACTS_OUTPUT_DIR, VECTORIZER_FILENAME_OUTPUT)
                try:
                    joblib.dump(vectorizer_fitted_on_train, vectorizer_output_path_final)
                    logging.info(f"Vectorizer treningowy zapisany w: {vectorizer_output_path_final}")
                except Exception as e:
                    logging.error(f"BŁĄD podczas zapisu vectorizera treningowego: {e}")
                
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                shutil.move(input_file_path_training, os.path.join(processed_training_data_path, f"{INPUT_CSV_FILENAME}_processed_{timestamp}.csv"))
                logging.info(f"Przeniesiono {INPUT_CSV_FILENAME} z {DATA_INPUT_DIR_TRAINING} do {processed_training_data_path}")
                
                logging.info(f"Skrypt {__file__} pomyślnie zakończył przetwarzanie pliku {INPUT_CSV_FILENAME}.")
                file_processed_this_run = True # Ustaw flagę, aby wyjść z pętli
                break # Wyjdź z pętli while True
            
            else: # Jeśli plik nie istnieje
                logging.info(f"Oczekuję na plik {INPUT_CSV_FILENAME} w {DATA_INPUT_DIR_TRAINING}... (sprawdzam co 10s)")
                pass

            if not file_processed_this_run: # Jeśli plik nie został jeszcze przetworzony, poczekaj
                time.sleep(10)
            # Jeśli chcesz, aby skrypt kończył się po określonym czasie, jeśli pliku nie ma, dodaj tu licznik.

        except Exception as e:
            logging.error(f"Wystąpił nieoczekiwany błąd w głównej pętli skryptu stan_nasluch.py: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            if 'input_file_path_training' in locals() and os.path.exists(input_file_path_training):
                try:
                    timestamp_err = time.strftime("%Y%m%d-%H%M%S")
                    shutil.move(input_file_path_training, os.path.join(processed_training_data_path, f"{INPUT_CSV_FILENAME}_CRITICAL_ERROR_TRAINING_{timestamp_err}.csv"))
                    logging.info(f"Przeniesiono {input_file_path_training} (powodujący błąd krytyczny) do {processed_training_data_path}")
                except Exception as move_e:
                    logging.error(f"Nie udało się przenieść pliku powodującego błąd krytyczny: {move_e}")
            
            file_processed_this_run = True # Przerwij pętlę również w przypadku krytycznego błędu
            logging.info("Przerywam działanie skryptu stan_nasluch.py z powodu błędu krytycznego.")
            break 
    
    if not file_processed_this_run and not os.path.exists(os.path.join(DATA_INPUT_DIR_TRAINING, INPUT_CSV_FILENAME)):
         logging.info(f"Nie znaleziono pliku {INPUT_CSV_FILENAME} w {DATA_INPUT_DIR_TRAINING} podczas tej sesji. Skrypt kończy działanie.")

    logging.info(f"Skrypt stan_nasluch.py zakończył działanie.")


if __name__ == "__main__":
    main()