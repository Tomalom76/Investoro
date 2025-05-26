@echo off
echo Aktywowanie srodowiska Conda 'projekt1'...

CALL C:\Tomek\Conda\condabin\activate.bat projekt1

REM Sprawdzenie, czy aktywacja sie powiodla (opcjonalne)
IF ERRORLEVEL 1 (
    echo Blad podczas aktywacji srodowiska Conda.
    pause
    exit /b
)

echo Uruchamianie skryptu nasluchujacego json_predictor_v3_3.py...
python json_predictor_v3_3.py

echo Zatrzymywanie skryptu (jesli doszlo do tego punktu bez bledu)...
REM Dezaktywacja srodowiska (opcjonalne, zamkniecie okna i tak to zrobi)
REM CALL conda deactivate

pause