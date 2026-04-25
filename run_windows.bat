@echo off
SETLOCAL EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================
echo   YouDub WebUI
echo ============================================
echo.

:: Check if venv exists
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo         Please run setup_windows.bat first to set up the environment.
    echo.
    pause
    EXIT /B 1
)

:: Activate the virtual environment
echo [1/3] Activating virtual environment...
CALL venv\Scripts\activate.bat

:: Verify activation by checking VIRTUAL_ENV
IF NOT DEFINED VIRTUAL_ENV (
    echo [ERROR] Failed to activate virtual environment.
    echo         Try running setup_windows.bat again.
    echo.
    pause
    EXIT /B 1
)
echo        Virtual environment activated.

:: Check Python availability
echo.
echo [2/3] Checking Python...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not available in the virtual environment.
    echo         Try running setup_windows.bat again.
    echo.
    pause
    EXIT /B 1
)
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version.split()[0])"') do set PYVER=%%i
echo        Found Python %PYVER%

:: Check critical dependencies
python -c "import gradio" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Gradio is not installed.
    echo         Please run setup_windows.bat to install dependencies.
    echo.
    pause
    EXIT /B 1
)

:: Handle share mode via environment variable
set SHARE_FLAG=
if "%YOUDUB_SHARE%"=="1" (
    set SHARE_FLAG=1
)

echo.
echo [3/3] Starting YouDub WebUI...
echo.

if "!SHARE_FLAG!"=="1" (
    echo [INFO] Share mode enabled (YOUDUB_SHARE=1^)
    echo        A public URL will be generated for remote access.
    echo.
)

echo ============================================
echo   YouDub WebUI is starting...
echo   Access: http://localhost:7860
echo   Press Ctrl+C to stop the application
echo ============================================
echo.

:: Auto-open browser after a short delay
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:7860"

:: Launch the application
if "!SHARE_FLAG!"=="1" (
    python -c "from app import app; app.launch(share=True)"
) else (
    python app.py
)

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with an error.
    echo         Check the error messages above for details.
    echo.
)

pause
ENDLOCAL
