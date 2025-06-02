@echo off
REM Plik wsadowy do uruchomienia skryptu json_state.py (predykcje na JSON)

echo Aktywowanie srodowiska Conda "projekt1"...
call C:\Tomek\Conda\condabin\activate.bat projekt1

echo Uruchamianie skryptu json_state.py...
python json_state.py

echo.
echo Skrypt zakonczyl dzialanie lub zostal przerwany.
pause