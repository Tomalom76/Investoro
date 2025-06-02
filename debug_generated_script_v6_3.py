import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import time
import os
import logging
import sys
from pycaret.regression import (setup, pull, compare_models, plot_model, load_model, tune_model, finalize_model, save_model, predict_model, get_config)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error

import mlflow
import time
import os
import logging
notebook_mlflow_uri_check = locals().get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
current_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
script_local_mlruns = os.path.join(current_script_dir, 'mlruns_SCRIPT_EXEC')
if not os.path.exists(script_local_mlruns): os.makedirs(script_local_mlruns, exist_ok=True)
script_tracking_uri = f"file:///{os.path.abspath(script_local_mlruns).replace(os.sep, '/')}"
use_notebook_uri = False
if notebook_mlflow_uri_check and not notebook_mlflow_uri_check.startswith('file:///C:/Users/Ai/Desktop/Tomek/Investoro/mlruns_DIRECT_LOCAL_TEST'):
    use_notebook_uri = True
effective_tracking_uri = notebook_mlflow_uri_check if use_notebook_uri else script_tracking_uri
mlflow.set_tracking_uri(effective_tracking_uri)
notebook_mlflow_exp_name = None
if 'MLFLOW_EXPERIMENT_NAME' in locals(): notebook_mlflow_exp_name = MLFLOW_EXPERIMENT_NAME
current_experiment_name = notebook_mlflow_exp_name if notebook_mlflow_exp_name else 'Predykcja_Cen_Mieszkan_Automatyczna_Watcher_v20'
experiment = mlflow.get_experiment_by_name(current_experiment_name)
if experiment is None: experiment_id = mlflow.create_experiment(current_experiment_name)
else: experiment_id = experiment.experiment_id
mlflow.set_experiment(experiment_name=current_experiment_name)
logging.info(f'MLflow: Ustawiono URI na: {mlflow.get_tracking_uri()}')
logging.info(f'MLflow: Ustawiono eksperyment na: {current_experiment_name}')

with mlflow.start_run(run_name=f'Automated_Run_NB63_{{time.strftime("%Y%m%d-%H%M%S")}}') as run:
    # Logowanie parametrów wejściowych
    mlflow.log_param('input_csv_path', r'C:\Tomek\Projekty\Investoro\DATA\data.csv')
    mlflow.log_param('notebook_path', r'C:\Tomek\Projekty\Investoro\EDA_Projekt1_v6_3FULL.ipynb')
    mlflow_tags_from_notebook_exec = {}
    if 'MLFLOW_TAGS' in locals() and isinstance(MLFLOW_TAGS, dict):
        mlflow_tags_from_notebook_exec = MLFLOW_TAGS
    mlflow.set_tags(mlflow_tags_from_notebook_exec)
    
    def convert_column_types(df_to_convert):
        df_copy = df_to_convert.copy()
        str_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber']
        for col in str_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str)
        if 'BuiltYear' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['BuiltYear']):
             df_copy['BuiltYear'] = pd.to_datetime(df_copy['BuiltYear'], format='%Y', errors='coerce')
        return df_copy
    
    import pandas as pd
    import mlflow
    # from pycaret.datasets import get_data 
    from pycaret.regression import setup, pull, compare_models, plot_model, load_model, tune_model, finalize_model, save_model, predict_model, get_config
    import pymysql
    from sqlalchemy import create_engine
    import numpy as np
    # from scipy.stats import skewnorm 
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer
    # from joblib import parallel_backend 
    MLFLOW_EXPERIMENT_NAME = 'Investoro_Ceny'
    MLFLOW_TAGS = {'data': 'Investoro_ceny', 'library': 'pycaret'}
    
    mlflow.set_tracking_uri("http://localhost:5000")
    # tę komórkę uruchom jeśli czerpiesz dane z pliku .csv
    df_original  = pd.read_csv(r'C:\Tomek\Projekty\Investoro\DATA\data.csv', sep=',') # ZMODYFIKOWANE PRZEZ SKRYPT
    df_original .head(10)
    df_original .sample(10)
    df_original [df_original .duplicated()]
    df_original .nunique()
    df_corr_temp = df_original.copy()
    if pd.api.types.is_string_dtype(df_corr_temp['BuiltYear']):
        df_corr_temp['BuiltYear_Num'] = pd.to_datetime(df_corr_temp['BuiltYear'], format='%Y', errors='coerce').dt.year
    elif pd.api.types.is_datetime64_any_dtype(df_corr_temp['BuiltYear']):
         df_corr_temp['BuiltYear_Num'] = df_corr_temp['BuiltYear'].dt.year
    else:
        df_corr_temp['BuiltYear_Num'] = pd.to_numeric(df_corr_temp['BuiltYear'], errors='coerce') # Ostateczna próba
    
    cols_for_corr = ['Area', 'Price', 'BuiltYear_Num', 'Floor', 'Floors', 'CommunityScore', 'CountyNumber', 'CommunityNumber',
                       'RegionNumber','KindNumber']
    # Upewnij się, że wszystkie kolumny istnieją i są numeryczne
    valid_cols_for_corr = [col for col in cols_for_corr if col in df_corr_temp.columns and pd.api.types.is_numeric_dtype(df_corr_temp[col])]
    correlation_matrix = df_corr_temp[valid_cols_for_corr].corr()
    df_original .isnull()
    df_original .isnull().sum()
    df_cleaned = df_original.copy()
    print(f"Rozmiar df_cleaned przed czyszczeniem: {df_cleaned.shape}")
    
    df_cleaned.dropna(subset=['Area', 'Price', 'Location'], inplace=True)
    print(f"Rozmiar df_cleaned po usunięciu NaN z Area, Price, Location: {df_cleaned.shape}")
    
    Q1_price = df_cleaned["Price"].quantile(0.25)
    Q3_price = df_cleaned["Price"].quantile(0.75)
    IQR_price = Q3_price - Q1_price
    lower_bound_price = Q1_price - 1.5 * IQR_price
    upper_bound_price = Q3_price + 1.5 * IQR_price
    df_cleaned = df_cleaned[~((df_cleaned["Price"] < lower_bound_price) | (df_cleaned["Price"] > upper_bound_price))]
    print(f"Rozmiar df_cleaned po usunięciu outlierów z Price: {df_cleaned.shape}")
    if "PricePerSquareMeter" in df_cleaned.columns and df_cleaned["PricePerSquareMeter"].isnull().sum() < len(df_cleaned) * 0.8: 
        df_cleaned.dropna(subset=['PricePerSquareMeter'], inplace=True) 
        Q1_ppsm = df_cleaned["PricePerSquareMeter"].quantile(0.25)
        Q3_ppsm = df_cleaned["PricePerSquareMeter"].quantile(0.75)
        IQR_ppsm = Q3_ppsm - Q1_ppsm
        lower_bound_ppsm = Q1_ppsm - 1.5 * IQR_ppsm
        upper_bound_ppsm = Q3_ppsm + 1.5 * IQR_ppsm
        df_cleaned = df_cleaned[~((df_cleaned["PricePerSquareMeter"] < lower_bound_ppsm) | (df_cleaned["PricePerSquareMeter"] > upper_bound_ppsm))]
        print(f"Rozmiar df_cleaned po usunięciu outlierów z PricePerSquareMeter: {df_cleaned.shape}")
    else:
        print("Kolumna 'PricePerSquareMeter' nie użyta do usuwania outlierów (brak lub za dużo NaN).")
    Q1_area = df_cleaned["Area"].quantile(0.25)
    Q3_area = df_cleaned["Area"].quantile(0.75)
    IQR_area = Q3_area - Q1_area
    lower_bound_area = Q1_area - 1.5 * IQR_area
    upper_bound_area = Q3_area + 1.5 * IQR_area
    df_cleaned = df_cleaned[~((df_cleaned["Area"] < lower_bound_area) | (df_cleaned["Area"] > upper_bound_area))]
    print(f"Rozmiar df_cleaned po usunięciu outlierów z Area: {df_cleaned.shape}")
    df_cleaned['BuiltYear'] = pd.to_datetime(df_cleaned['BuiltYear'], format='%Y', errors='coerce')
    print("Konwersja BuiltYear na datetime w df_cleaned zakończona.")
    print("Informacje o df_cleaned po wszystkich krokach czyszczenia:")
    df_cleaned.info()
    print("\nBraki danych w df_cleaned (%):")
    print("\nPierwsze wiersze df_cleaned:")
    
    print(f"Rozmiar df_cleaned przed podziałem na train/holdout: {df_cleaned.shape}")
    train_df = df_cleaned.sample(frac=0.9, random_state=42)
    holdout_df = df_cleaned.drop(train_df.index)
    
    print(f"Rozmiar zbioru treningowego (train_df): {train_df.shape}")
    print(f"Rozmiar zbioru holdout (holdout_df): {holdout_df.shape}")
    def convert_column_types(df_to_convert):
        df_copy = df_to_convert.copy()
        str_cols = ['VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber', 'StreetNumber'] # Dodano StreetNumber
        for col in str_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str)
        
        # BuiltYear powinno być już datetime, ale upewnijmy się
        if 'BuiltYear' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['BuiltYear']):
             df_copy['BuiltYear'] = pd.to_datetime(df_copy['BuiltYear'], format='%Y', errors='coerce')
        return df_copy
    
    train_df = convert_column_types(train_df)
    holdout_df = convert_column_types(holdout_df)
    
    print("\\nTypy danych w train_df po konwersji:")
    train_df.info()
    print("\\nTypy danych w holdout_df po konwersji:")
    holdout_df.info()
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    location_vectorizer = TfidfVectorizer(
        max_features=100, 
        stop_words=None,
        ngram_range=(1, 1),
        min_df=5,
        max_df=0.95
    )
    print("Przetwarzanie TF-IDF dla zbioru treningowego...")
    train_df_copy = train_df.copy() # Pracujemy na kopii
    train_df_copy['Location_Clean'] = train_df_copy['Location'].fillna('').astype(str)
    train_location_tfidf_features = location_vectorizer.fit_transform(train_df_copy['Location_Clean'])
    
    try:
        feature_names = location_vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = location_vectorizer.get_feature_names_() 
        
    train_location_tfidf_df = pd.DataFrame(
        train_location_tfidf_features.toarray(),
        columns=['loc_tfidf_' + name for name in feature_names],
        index=train_df_copy.index # Ważne, aby zachować oryginalny indeks
    )
    print(f"Utworzono {train_location_tfidf_df.shape[1]} cech TF-IDF dla zbioru treningowego.")
    
    train_df_processed = pd.concat(
        [train_df_copy.drop(columns=['Location', 'Location_Clean'], errors='ignore'), train_location_tfidf_df], 
        axis=1
    )
    # WAŻNE: Usuń wiersze gdzie 'Price' < 20000 (lub inna wartość) dopiero PO przetworzeniu TF-IDF, 
    # aby uniknąć problemów z niedopasowaniem indeksów przy konkatenacji.
    train_df_processed = train_df_processed[train_df_processed['Price'] >= 20000] 
    print(f"Rozmiar train_df_processed po usunięciu cen < 20000: {train_df_processed.shape}")
    
    print("Przetwarzanie TF-IDF dla zbioru holdout...")
    holdout_df_copy = holdout_df.copy() # Pracujemy na kopii
    holdout_df_copy['Location_Clean'] = holdout_df_copy['Location'].fillna('').astype(str)
    holdout_location_tfidf_features = location_vectorizer.transform(holdout_df_copy['Location_Clean']) # Użyj transform
    
    holdout_location_tfidf_df = pd.DataFrame(
        holdout_location_tfidf_features.toarray(),
        columns=['loc_tfidf_' + name for name in feature_names],
        index=holdout_df_copy.index # Ważne, aby zachować oryginalny indeks
    )
    print(f"Utworzono {holdout_location_tfidf_df.shape[1]} cech TF-IDF dla zbioru holdout.")
    
    holdout_df_processed = pd.concat(
        [holdout_df_copy.drop(columns=['Location', 'Location_Clean'], errors='ignore'), holdout_location_tfidf_df],
        axis=1
    )
    print(f"Rozmiar holdout_df_processed: {holdout_df_processed.shape}")
    
    categorical_features_initial = [
        'BuildingType', 'BuildingCondition', 'TypeOfMarket', 'OwnerType', 'Type', 'OfferFrom',
        'VoivodeshipNumber', 'CountyNumber', 'CommunityNumber', 'KindNumber', 'RegionNumber'
    ]
    numeric_features_initial = [
        'Area', 'NumberOfRooms', 'Floor', 'Floors', 'CommunityScore'
    ]
    date_features_initial = ['BuiltYear']
    
    categorical_features_to_use = [col for col in categorical_features_initial if col in train_df_processed.columns]
    numeric_features_to_use = [col for col in numeric_features_initial if col in train_df_processed.columns]
    # Kolumny loc_tfidf_* zostaną dodane do numeric_features wewnątrz setup
    date_features_to_use = [col for col in date_features_initial if col in train_df_processed.columns]
    
    ignore_features_list_setup = [ # (...) lista jak w oryginalnym kodzie
        'SaleId', 'OriginalId', 'PortalId', 'Title', 'Description',
        'OfferPrice', 'RealPriceAfterRenovation', 'OriginalPrice',
        'PricePerSquareMeter', 'DateAddedToDatabase', 'DateAdded',
        'DateLastModification', 'DateLastRaises', 'NewestDate',
        'AvailableFrom', 'Link', 'Phone', 'MainImage', 'OtherImages',
        'NumberOfDuplicates', 'NumberOfRaises', 'NumberOfModifications',
        'IsDuplicatePriceLower', 'IsDuplicatePrivateOwner', 'Score', 'ScorePrecision',
        'NumberOfCommunityComments', 'NumberOfCommunityOpinions', 'Archive',
        'SubRegionNumber', 'EncryptedId',
        'StreetNumber' # StreetNumber jest teraz stringiem, jeśli ma za dużo wartości, można go tu dodać
    ]
    ignore_features_final = [col for col in ignore_features_list_setup if col in train_df_processed.columns]
    
    print("--- Informacje przed PyCaret setup ---")
    print("Liczba kolumn w train_df_processed:", len(train_df_processed.columns.tolist()))
    print("Cechy kategoryczne:", categorical_features_to_use)
    print("Cechy numeryczne (początkowe):", numeric_features_to_use)
    print("Cechy daty:", date_features_to_use)
    print("Ignorowane cechy:", ignore_features_final)
    print("------------------------------------")
    import os
    # Utwórz dedykowany katalog dla tego testu, jeśli nie istnieje
    current_directory = os.getcwd() 
    local_mlruns_path = os.path.join(current_directory, "mlruns_DIRECT_LOCAL_TEST") 
    
    if not os.path.exists(local_mlruns_path):
        os.makedirs(local_mlruns_path)
        print(f"Utworzono katalog: {local_mlruns_path}")
    else:
        print(f"Katalog już istnieje: {local_mlruns_path}")
    
    absolute_mlruns_path = os.path.abspath(local_mlruns_path)
    tracking_uri = f"file:///{absolute_mlruns_path.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow tracking URI ustawione na: {mlflow.get_tracking_uri()}")
    
    # MLFLOW_EXPERIMENT_NAME powinno być zdefiniowane wcześniej w Twoim notebooku
    # np. MLFLOW_EXPERIMENT_NAME = 'Investoro_Ceny'
    
    try:
        # Sprawdź, czy eksperyment istnieje
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            # Jeśli nie istnieje, stwórz go.
            # Dla logowania typu 'file://', MLflow sam zarządzi lokalizacją artefaktów
            # w podkatalogach struktury 'mlruns'.
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            print(f"Utworzono nowy eksperyment MLflow: '{MLFLOW_EXPERIMENT_NAME}' o ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Znaleziono istniejący eksperyment: '{MLFLOW_EXPERIMENT_NAME}' o ID: {experiment_id}")
        
        # Ustaw eksperyment jako aktywny
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
        print(f"Aktywny eksperyment MLflow ustawiony na: '{MLFLOW_EXPERIMENT_NAME}'")
    
    except Exception as e:
        print(f"Błąd podczas ustawiania/tworzenia eksperymentu MLflow: {e}")
        import traceback
        print(traceback.format_exc())
    
    reg_exp = None 
    try:
        loc_tfidf_cols = [col for col in train_df_processed.columns if 'loc_tfidf_' in col]
        reg_exp = setup(
            data=train_df_processed, 
            target='Price',
            log_experiment=True,
            experiment_name=MLFLOW_EXPERIMENT_NAME,
            categorical_features=categorical_features_to_use,
            numeric_features=numeric_features_to_use + loc_tfidf_cols, # Dodajemy cechy TF-IDF
            date_features=date_features_to_use,
            ignore_features=ignore_features_final,
            # ... (reszta parametrów jak w oryginalnym kodzie) ...
        )
    except Exception as e:
        print(f"Błąd podczas setup PyCaret: {e}")
        import traceback
        print(traceback.format_exc())
    
    if reg_exp:
        best_model_from_compare = compare_models() # Zapisz najlepszy model z compare_models
        compare_metrics_df = pull()
    else:
        print("Nie udało się zainicjować eksperymentu PyCaret.")
    from pycaret.regression import get_config
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Wyciągnij dane oryginalne i przetransformowane
    df_raw = train_df_processed.copy()
    df_transformed = get_config("X_train").copy()
    df_transformed["Price"] = get_config("y_train")
    
    # Rysowanie wykresów
    #fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    #sns.histplot(df_raw["Price"], ax=axes[0])
    #axes[0].set_title("Raw Data")
    
    #sns.histplot(df_transformed["Price"], ax=axes[1])
    #axes[1].set_title("Transformed Data")
    
    #plt.tight_layout()
    #plt.show()
    reg_exp.plot_model(best_model_from_compare, plot='feature')
    if reg_exp and best_model_from_compare:
        print("Predykcja na zbiorze testowym (z podziału PyCaret) przy użyciu dostrojonego modelu:")
        predict_model(best_model_from_compare) # To wyświetli metryki na wewnętrznym zbiorze testowym
        test_set_metrics_after_tuning = pull()
    
    best_final_model = None
    if reg_exp and best_model_from_compare:
        best_final_model = finalize_model(best_model_from_compare, experiment_custom_tags={"step": "final_tuned"})
        print("Sfinalizowany model (po strojeniuu):")
    elif reg_exp and 'best_model_from_compare' in locals() and best_model_from_compare is not None:
        print("Strojenie nie powiodło się lub zostało pominięte. Finalizuję najlepszy model z compare_models.")
        best_final_model = finalize_model(best_model_from_compare, experiment_custom_tags={"step": "final_compare"})
    else:
        print("Nie można sfinalizować modelu.")
    save_model(best_final_model, r'C:\\Tomek\\Projekty\\Investoro\\DATA_OUT\\0_best_LGBM_model_via_script_watch') # ZMODYFIKOWANE PRZEZ SKRYPT
    
    # ----- POCZĄTEK FINALNEGO PRZETWARZANIA I ZAPISU (dla notebooka v6.3) -----
    
    # === DODANY BLOK: Zapis TfidfVectorizer ===
    # Ten kod wykona się wewnątrz exec(), po wykonaniu kodu z notebooka
    if 'location_vectorizer' in locals() and 'DATA_OUT_DIR' in locals() and 'os' in locals() and 'joblib' in locals() and 'mlflow' in locals():
        try:
            # Używamy stałej nazwy pliku, tej samej, której oczekuje skrypt predykcyjny
            vectorizer_save_path = os.path.join(DATA_OUT_DIR, 'location_vectorizer.joblib') 
            joblib.dump(location_vectorizer, vectorizer_save_path)
            logging.info(f"ZAPISANO (z watcher.py): TfidfVectorizer do: {vectorizer_save_path}")
            if mlflow.active_run():
                mlflow.log_artifact(vectorizer_save_path, "location_vectorizer_artifact") # Nazwa artefaktu w MLflow
                logging.info(f"ZAPISANO (z watcher.py): TfidfVectorizer jako artefakt MLflow.")
        except Exception as e_vec_save:
            logging.error(f"BŁĄD (z watcher.py): Nie udało się zapisać TfidfVectorizer: {e_vec_save}")
    else:
        missing_vars_for_vec_save = [var for var in ['location_vectorizer', 'DATA_OUT_DIR', 'os', 'joblib', 'mlflow'] if var not in locals()]
        logging.warning(f"Nie znaleziono wymaganych zmiennych/modułów do zapisu TfidfVectorizer w watcher.py. Brakujące: {missing_vars_for_vec_save}")
    # === KONIEC BLOKU: Zapis TfidfVectorizer ===
    
    if 'reg_exp' in locals() and \
       'best_final_model' in locals() and \
       'df_original' in locals() and isinstance(df_original, pd.DataFrame) and \
       'location_vectorizer' in locals() and \
       'convert_column_types' in locals() :
    
        logging.info("Przygotowywanie df_original do finalnej predykcji...")
        
        df_to_predict_on = df_original.copy()
    
        df_to_predict_on.dropna(subset=['Area', 'Price', 'Location'], inplace=True) # DODANO 'Price' TUTAJ
        logging.info(f"Rozmiar df_to_predict_on po usunięciu NaN z Area, Price, Location: {df_to_predict_on.shape}")
    
        if not df_to_predict_on.empty:
            Q1_area_orig = df_to_predict_on["Area"].quantile(0.25)
            Q3_area_orig = df_to_predict_on["Area"].quantile(0.75)
            IQR_area_orig = Q3_area_orig - Q1_area_orig
            lower_bound_area_orig = Q1_area_orig - 1.5 * IQR_area_orig
            upper_bound_area_orig = Q3_area_orig + 1.5 * IQR_area_orig
            df_to_predict_on = df_to_predict_on[
                ~((df_to_predict_on["Area"] < lower_bound_area_orig) | (df_to_predict_on["Area"] > upper_bound_area_orig))
            ]
            logging.info(f"Rozmiar df_to_predict_on po usunięciu outlierów z Area: {df_to_predict_on.shape}")
        
        if not df_to_predict_on.empty:
            df_to_predict_on = convert_column_types(df_to_predict_on)
            logging.info("Konwersja typów danych w df_to_predict_on zakończona.")
    
        df_original_processed_for_script = pd.DataFrame() 
        if not df_to_predict_on.empty:
            df_to_predict_on['Location_Clean'] = df_to_predict_on['Location'].fillna('').astype(str)
            original_location_tfidf_features = location_vectorizer.transform(df_to_predict_on['Location_Clean'])
            
            try:
                feature_names_loc_script = location_vectorizer.get_feature_names_out()
            except AttributeError:
                feature_names_loc_script = location_vectorizer.get_feature_names_()
    
            original_location_tfidf_df_script = pd.DataFrame(
                original_location_tfidf_features.toarray(),
                columns=['loc_tfidf_' + name for name in feature_names_loc_script],
                index=df_to_predict_on.index
            )
            df_original_processed_for_script = pd.concat(
                [df_to_predict_on.drop(columns=['Location', 'Location_Clean'], errors='ignore'), original_location_tfidf_df_script],
                axis=1
            )
            logging.info(f"Utworzono {original_location_tfidf_df_script.shape[1]} cech TF-IDF dla df_original_processed_for_script.")
            logging.info(f"Rozmiar df_original_processed_for_script: {df_original_processed_for_script.shape}")
        
        if not df_original_processed_for_script.empty:
            logging.info("Wykonywanie predykcji na całym przetworzonym df_original...")
            data_for_final_pred_script = df_original_processed_for_script.copy()
            
            actual_prices_for_metric_calc = None
            if 'Price' in df_original_processed_for_script.columns:
                actual_prices_for_metric_calc = df_original_processed_for_script['Price'].copy() 
            
            if 'Price' in data_for_final_pred_script.columns:
                data_for_final_pred_script = data_for_final_pred_script.drop(columns=['Price'])
                logging.info("Usunięto kolumnę 'Price' z danych przekazywanych do finalnego predict_model.")
            
            predictions_on_original_df = predict_model(best_final_model, data=data_for_final_pred_script)
            
            if 'prediction_label' in predictions_on_original_df.columns:
                if actual_prices_for_metric_calc is not None:
                    metrics_df = pd.DataFrame({
                        'actual': actual_prices_for_metric_calc,
                        'predicted': predictions_on_original_df.loc[actual_prices_for_metric_calc.index, 'prediction_label']
                    })
                    metrics_df.dropna(inplace=True)
    
                    if not metrics_df.empty:
                        r2_orig = r2_score(metrics_df['actual'], metrics_df['predicted'])
                        mae_orig = mean_absolute_error(metrics_df['actual'], metrics_df['predicted'])
                        logging.info(f"Metryki na przetworzonym df_original (po usunięciu NaN w Price i predykcjach): R2 = {r2_orig:.4f}, MAE = {mae_orig:.2f}")
                        if mlflow.active_run():
                            mlflow.log_metric("df_original_final_R2", r2_orig)
                            mlflow.log_metric("df_original_final_MAE", mae_orig)
                    else:
                        logging.warning("Nie można obliczyć metryk - brak danych po usunięciu NaN z cen rzeczywistych lub predykcji.")
                
                output_df = df_original.copy() 
                
                predictions_to_add_script = pd.DataFrame(index=df_original_processed_for_script.index)
                if 'SaleId' in df_original_processed_for_script.columns:
                     predictions_to_add_script['SaleId'] = df_original_processed_for_script['SaleId']
                else: 
                     if df_original_processed_for_script.index.name == 'SaleId':
                         predictions_to_add_script['SaleId'] = df_original_processed_for_script.index.to_series()
                     else:
                         logging.warning("Nie można jednoznacznie zidentyfikować 'SaleId' w df_original_processed_for_script. Predykcje mogą nie zostać poprawnie połączone.")
    
                predictions_to_add_script['PredictedPrice_LGBM'] = predictions_on_original_df.loc[predictions_to_add_script.index, 'prediction_label']
    
                if 'SaleId' in predictions_to_add_script.columns and 'SaleId' in output_df.columns:
                    try:
                        output_df['SaleId'] = output_df['SaleId'].astype(str)
                        predictions_to_add_script['SaleId'] = predictions_to_add_script['SaleId'].astype(str)
                        predictions_to_add_script.drop_duplicates(subset=['SaleId'], keep='first', inplace=True)
                        output_df = pd.merge(output_df, predictions_to_add_script, on='SaleId', how='left')
                    except Exception as e_merge:
                        logging.error(f"Błąd podczas łączenia predykcji z oryginalnym DataFrame: {e_merge}")
                else:
                     logging.error("Nie można połączyć predykcji - brak kolumny 'SaleId' w jednym z DataFrame'ów.")
    
                if 'PredictedPrice_LGBM' in output_df.columns:
                    output_df['PredictedPrice_LGBM_num'] = pd.to_numeric(output_df['PredictedPrice_LGBM'], errors='coerce').round(0)
                    output_df['PredictedPrice_LGBM_formatted'] = output_df['PredictedPrice_LGBM_num'].apply(
                        lambda x: f'{x:,.0f}' if pd.notna(x) else '' 
                    )
                    output_df['PredictedPrice_LGBM'] = output_df['PredictedPrice_LGBM_formatted']
                    output_df.drop(columns=['PredictedPrice_LGBM_num', 'PredictedPrice_LGBM_formatted'], inplace=True, errors='ignore')
    
                    cols_script = list(output_df.columns)
                    if 'Price' in cols_script and 'PredictedPrice_LGBM' in cols_script:
                        try:
                            price_idx_script = cols_script.index('Price')
                            if 'PredictedPrice_LGBM' in cols_script: 
                                cols_script.remove('PredictedPrice_LGBM')
                            cols_script.insert(price_idx_script + 1, 'PredictedPrice_LGBM')
                            output_df = output_df[cols_script]
                            logging.info("Przesunięto kolumnę 'PredictedPrice_LGBM' obok 'Price'.")
                        except ValueError as ve_reorder: 
                            logging.error(f"Błąd podczas przesuwania kolumny (ValueError): {ve_reorder}")
                        except Exception as e_reorder: 
                             logging.error(f"Inny błąd podczas przesuwania kolumny: {e_reorder}")
                    elif 'PredictedPrice_LGBM' in cols_script:
                        logging.info("Kolumna 'Price' nie istnieje, 'PredictedPrice_LGBM' pozostaje na końcu.")
                    else:
                        logging.warning("Nie znaleziono kolumny 'PredictedPrice_LGBM' do przesunięcia.")
                else:
                     logging.warning("Kolumna 'PredictedPrice_LGBM' nie została dodana do output_df po merge.")
    
                try:
                    output_path_final_script = r'C:\Tomek\Projekty\Investoro\DATA_OUT\data_out_20250530_180939.csv'
                    output_df.to_csv(output_path_final_script, index=False, sep=',')
                    logging.info("Finalny DataFrame (oryginalne dane + predykcje) zapisany do: %s", output_path_final_script)
                    if mlflow.active_run():
                       mlflow.log_artifact(output_path_final_script, "output_data_on_full_original")
                       logging.info(f"Zalogowano {output_path_final_script} jako artefakt MLflow.")
                except Exception as e_save_final:
                    logging.error("Błąd podczas zapisywania finalnego DataFrame: %s", e_save_final)
                    import traceback; logging.error(traceback.format_exc())
            else:
                logging.error("Kolumna 'prediction_label' nie została znaleziona w wynikach predict_model na df_original_processed_for_script.")
        else:
            logging.warning("DataFrame df_original_processed_for_script jest pusty po przetworzeniu, nie wykonano finalnej predykcji.")
    else:
        logging.warning("Nie znaleziono wymaganych obiektów ('reg_exp', 'best_final_model', 'df_original', 'location_vectorizer', 'convert_column_types') do przeprowadzenia finalnego przetwarzania.")
    
    # ----- KONIEC FINALNEGO PRZETWARZANIA I ZAPISU (dla notebooka v6.3) -----
