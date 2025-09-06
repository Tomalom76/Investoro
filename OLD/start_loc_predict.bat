@echo off
echo Aktywowanie srodowiska Conda 'projekt1'...

CALL C:\miniconda3\condabin\activate.bat projekt1

REM Sprawdzenie, czy aktywacja sie powiodla (opcjonalne)
IF ERRORLEVEL 1 (
    echo Blad podczas aktywacji srodowiska Conda.
    pause
    exit /b
)

echo Uruchamianie skryptu nasluchujacego predict_hybrid.py...
python predict_hybrid.py

echo Zatrzymywanie skryptu (jesli doszlo do tego punktu bez bledu)...
REM Dezaktywacja srodowiska (opcjonalne, zamkniecie okna i tak to zrobi)
REM CALL conda deactivate

pause