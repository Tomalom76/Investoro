@echo off
REM Plik wsadowy do uruchomienia głównego skryptu orkiestrującego start_all.py

call conda activate projekt1
call C:\Tomek\Conda\condabin\activate.bat projekt1

echo Uruchamianie skryptu start_all.py...
python start_all.py

echo.
echo Skrypt start_all.py zakonczyl dzialanie lub zostal przerwany.
pause