import joblib
import os

# --- Konfiguracja ---
SOURCE_DIR = "DATA_OUT"
MODEL_PIPELINE_FILENAME = "0_best_LGBM_model_via_script_watch.pkl"

def inspect_pipeline():
    """
    Ta funkcja ma tylko jedno zadanie: wczytać pipeline i wypisać jego budowę,
    a w szczególności listę cech, których oczekuje finalny model.
    """
    model_path = os.path.join(SOURCE_DIR, MODEL_PIPELINE_FILENAME)
    
    print("--- ROZPOCZYNAM INSPEKCJĘ MODELU ---")
    print(f"Próbuję wczytać plik: {model_path}\n")

    if not os.path.exists(model_path):
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku modelu w '{model_path}'.")
        return

    try:
        pipeline = joblib.load(model_path)
        print(">>> Pipeline wczytany pomyślnie. Analizuję jego strukturę...\n")

        # Wypisujemy wszystkie kroki (transformery) w pipeline
        print("--- KROKI (TRANSFORMERY) Wewnątrz PIPELINE'U ---")
        for i, (name, step) in enumerate(pipeline.steps):
            print(f"Krok {i}: Nazwa = '{name}', Obiekt = {type(step)}")
        
        # Próbujemy dostać się do ostatniego kroku, którym powinien być finalny model
        final_model_step = pipeline.steps[-1][1]
        print(f"\n>>> Finalny model to obiekt typu: {type(final_model_step)}\n")

        # Wypisujemy listę cech, na których ten finalny model został wytrenowany
        if hasattr(final_model_step, 'feature_name_'):
            print("--- LISTA OCZEKIWANYCH KOLUMN (CECH) PRZEZ FINALNY MODEL ---")
            expected_features = final_model_step.feature_name_
            print(f"Model oczekuje DOKŁADNIE {len(expected_features)} kolumn.")
            # Zapiszmy tę listę do pliku, żeby mieć ją na przyszłość
            with open("oczekiwane_kolumny.txt", "w") as f:
                for feature in expected_features:
                    f.write(f"{feature}\n")
            print("\n>>> ZAPISAŁEM PEŁNĄ LISTĘ OCZEKIWANYCH KOLUMN DO PLIKU: 'oczekiwane_kolumny.txt' <<<")
            print(">>> To jest 'Święty Graal', którego szukaliśmy. To jest przepis na DataFrame, który musimy stworzyć. <<<")

        else:
            print("!!! OSTRZEŻENIE: Finalny model nie ma atrybutu 'feature_name_'. To może oznaczać, że jest to inny typ obiektu. !!!")
            if hasattr(final_model_step, 'feature_names_in_'):
                 print("--- Znalazłem alternatywną listę cech ('feature_names_in_') ---")
                 print(final_model_step.feature_names_in_)


    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas inspekcji: {e}")

if __name__ == "__main__":
    inspect_pipeline()
    input("\nNaciśnij Enter, aby zakończyć.")