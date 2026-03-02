@echo off
SETLOCAL

:: Change to the script directory
cd /d "%~dp0"

:: Activate the virtual environment
CALL venv\Scripts\activate

:: Start the application
echo Starting the application...
python app.py
pause

ENDLOCAL
