@echo off
REM Plik wsadowy do uruchomienia skryptu stan_nasluch.py

echo Aktywowanie srodowiska Conda...
call C:\Tomek\Conda\condabin\activate.bat projekt1
REM REM Lub jeśli masz Minicondę:
REM REM call C:\Sciezka\Do\Twojej\Miniconda3\Scripts\activate.bat projekt1
REM REM Upewnij się, że ścieżka do activate.bat jest poprawna!


echo Uruchamianie skryptu stan_nasluch.py...
python stan_nasluch.py

echo.
echo Skrypt zakonczyl dzialanie lub zostal przerwany.
pause