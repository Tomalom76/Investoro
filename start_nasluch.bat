@echo off
echo Aktywowanie srodowiska Conda 'projekt1'...

CALL C:\Tomek\Conda\condabin\activate.bat projekt1

REM Sprawdzenie, czy aktywacja sie powiodla (opcjonalne)
IF ERRORLEVEL 1 (
    echo Blad podczas aktywacji srodowiska Conda.
    pause
    exit /b
)

echo Uruchamianie skryptu nasluchujacego model_watcher_v1_2.py...
python model_watcher_v1_2.py

echo Zatrzymywanie skryptu (jesli doszlo do tego punktu bez bledu)...
REM Dezaktywacja srodowiska (opcjonalne, zamkniecie okna i tak to zrobi)
REM CALL conda deactivate

pause