@echo off
SETLOCAL EnableDelayedExpansion

cd /d "%~dp0"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to change to script directory: "%~dp0"
    echo Please run this script from a local drive, not a network path.
    pause
    EXIT /B 1
)

set LOGFILE=%~dp0run.log
echo [%date% %time%] YouDub Startup Started > "%LOGFILE%"

title YouDub WebUI

echo ============================================
echo   YouDub WebUI
echo ============================================
echo.

:: ============================================
:: [1/6] Check virtual environment
:: ============================================
echo [1/6] Checking virtual environment...
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo.
    echo [ERROR] Virtual environment not found.
    echo.
    echo ============================================
    echo   Troubleshooting
    echo   1. Run setup_windows.bat first to install dependencies
    echo   2. Verify setup_windows.bat completed successfully
    echo   3. Check permissions to create the venv directory
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: Virtual environment not found >> "%LOGFILE%"
    pause
    EXIT /B 1
)
echo        Virtual environment found.
echo [%date% %time%] Virtual environment found >> "%LOGFILE%"

:: ============================================
:: [2/6] Activate virtual environment
:: ============================================
echo.
echo [2/6] Activating virtual environment...
CALL venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to activate virtual environment.
    echo.
    echo ============================================
    echo   Troubleshooting
    echo   1. Delete the venv folder, then re-run setup_windows.bat
    echo   2. Verify your Python installation is working
    echo   3. Run this script as Administrator
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: Failed to activate virtual environment >> "%LOGFILE%"
    pause
    EXIT /B 1
)
IF NOT DEFINED VIRTUAL_ENV (
    echo.
    echo [ERROR] Virtual environment activation failed: VIRTUAL_ENV not set.
    echo.
    echo ============================================
    echo   Troubleshooting
    echo   1. Re-run setup_windows.bat
    echo   2. Check if antivirus is blocking activation
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: VIRTUAL_ENV not set after activation >> "%LOGFILE%"
    pause
    EXIT /B 1
)
echo        Virtual environment activated.
echo [%date% %time%] Virtual environment activated >> "%LOGFILE%"

:: ============================================
:: [3/6] Check Python version
:: ============================================
echo.
echo [3/6] Checking Python version...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Python is not available in the virtual environment.
    echo.
    echo ============================================
    echo   Troubleshooting
    echo   1. Delete the venv folder
    echo   2. Re-run setup_windows.bat
    echo   3. Verify Python 3.8+ is installed on your system
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: Python not available >> "%LOGFILE%"
    pause
    EXIT /B 1
)

for /f "tokens=*" %%i in ('python -c "import sys; ver=sys.version.split()[0].split('.'); maj=int(ver[0]); min=int(ver[1]); print(maj); print(min)"') do (
  if not defined PYMAJOR (
    set PYMAJOR=%%i
  ) else (
    set PYMINOR=%%i
  )
)
echo        Found Python %PYMAJOR%.%PYMINOR%.x

:: Check via batch comparison (avoids cmd.exe > parsing issue with Python inline code)
set VERSION_OK=0
if !PYMAJOR! GEQ 4 set VERSION_OK=1
if !PYMAJOR! EQU 3 if !PYMINOR! GEQ 8 set VERSION_OK=1
if not !VERSION_OK! EQU 0 goto VERSION_PASS
echo.
echo [ERROR] Python version is too old.
echo.
echo ============================================
echo   Current version: %PYMAJOR%.%PYMINOR%
echo   Required version: 3.8 or higher
echo.
echo   Troubleshooting
echo   1. Download the latest Python from https://www.python.org/downloads/
echo   2. Select "Add Python to PATH" during installation
echo   3. Re-run setup_windows.bat
echo ============================================
echo.
echo [%date% %time%] ERROR: Python %PYMAJOR%.%PYMINOR% is too old (requires 3.8+) >> "%LOGFILE%"
pause
EXIT /B 1
:VERSION_PASS
echo        Python version check passed.
echo [%date% %time%] Python %PYMAJOR%.%PYMINOR% version check passed >> "%LOGFILE%"

:: ============================================
:: [4/6] Check critical dependencies
:: ============================================
echo.
echo [4/6] Checking dependencies...

echo        Checking gradio...
python -c "import gradio; print(f'Gradio {gradio.__version__}')" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Gradio is not installed or is corrupted.
    echo.
    echo ============================================
    echo   Troubleshooting
    echo   1. Run setup_windows.bat to reinstall dependencies
    echo   2. Or manually: pip install gradio
    echo   3. Check network connection and retry pip install
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: Gradio not installed >> "%LOGFILE%"
    pause
    EXIT /B 1
)
for /f "tokens=*" %%i in ('python -c "import gradio; print(gradio.__version__)"') do set GRADIO_VER=%%i
echo        Gradio %GRADIO_VER% installed

echo        Checking torch...
python -c "import torch; print(f'Torch {torch.__version__}')" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] PyTorch is not installed or is corrupted.
    echo        Some features may not work properly.
    echo.
    echo   Troubleshooting
    echo   Run setup_windows.bat to reinstall PyTorch
    echo.
    echo [%date% %time%] WARNING: PyTorch not found >> "%LOGFILE%"
) else (
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.__version__)"') do set TORCH_VER=%%i
    echo        PyTorch %TORCH_VER% installed
    echo [%date% %time%] PyTorch %TORCH_VER% found >> "%LOGFILE%"
)
echo        Dependency check complete.
echo [%date% %time%] Critical dependencies check passed >> "%LOGFILE%"

:: ============================================
:: [5/6] Check configuration
:: ============================================
echo.
echo [5/6] Checking configuration file...
IF NOT EXIST ".env" (
    echo [WARNING] .env configuration file not found.
    echo        Default settings will be used. Some features may be limited.
    echo.
    echo   Tip
    echo   Copy .env.example to .env and fill in the required API keys
    echo.
    echo [%date% %time%] WARNING: .env file not found >> "%LOGFILE%"
) else (
    echo        .env configuration file found.
    echo [%date% %time%] .env file found >> "%LOGFILE%"
)

:: ============================================
:: [6/6] Prepare to launch
:: ============================================
echo.
echo [6/6] Preparing to launch application...

:: Handle share mode via environment variable
set SHARE_FLAG=
if "%YOUDUB_SHARE%"=="1" (
    set SHARE_FLAG=1
)

if "!SHARE_FLAG!"=="1" (
    echo        Share mode enabled (YOUDUB_SHARE=1)
    echo        A public URL will be provided for remote access
    echo [%date% %time%] Share mode enabled >> "%LOGFILE%"
)

echo.
echo ============================================
echo   Ready to launch!
echo ============================================
echo.
echo   URL: http://127.0.0.1:19876
echo   Stop: Press Ctrl+C
echo.
if "!SHARE_FLAG!"=="1" (
echo   Share mode: Enabled - external URL will be shown after launch
echo.
)
echo ============================================
echo.
echo [%date% %time%] All checks passed, starting application >> "%LOGFILE%"

:: Auto-open browser after a short delay
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:19876"

:: Record startup time for quick-failure detection
set START_TIME=%time%

:: Launch the application (inbrowser=False to prevent Gradio from opening its own browser)
if "!SHARE_FLAG!"=="1" (
    python -c "from app import app; app.launch(share=True, inbrowser=False)"
) else (
    python -c "from app import app; app.launch(server_port=19876, inbrowser=False)"
)

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================
    echo   [ERROR] Application exited unexpectedly
    echo ============================================
    echo.
    echo   See error messages above for details.
    echo.
    echo   Possible causes
    echo   1. Check if port 19876 is already in use
    echo   2. Check run.log for detailed logs
    echo   3. Run setup_windows.bat to reinstall dependencies
    echo   4. Verify .env configuration is correct
    echo.
    echo [%date% %time%] ERROR: Application exited with error code %ERRORLEVEL% >> "%LOGFILE%"
    echo.
    pause
    EXIT /B 1
)

echo.
echo ============================================
echo   Application exited normally
echo ============================================
echo.
echo [%date% %time%] Application exited normally >> "%LOGFILE%"
echo.
pause
ENDLOCAL
