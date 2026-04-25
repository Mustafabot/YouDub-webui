@echo off
SETLOCAL EnableDelayedExpansion

set LOGFILE=install.log
echo [%date% %time%] YouDub Setup Started > %LOGFILE%

echo ============================================
echo   YouDub Windows Setup
echo ============================================
echo.

:: Check for Python 3.8+
echo [1/7] Checking Python version...
py -c "import sys; ver=tuple(map(int,sys.version.split()[0].split('.'))); sys.exit(0 if ver>=(3,8) else 1)" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.8+ is required but not found.
    echo Please download and install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo [%date% %time%] ERROR: Python 3.8+ not found >> %LOGFILE%
    pause
    EXIT /B 1
)
for /f "tokens=*" %%i in ('py -c "import sys; print(sys.version.split()[0])"') do set PYVER=%%i
echo         Found Python %PYVER%
echo [%date% %time%] Python %PYVER% found >> %LOGFILE%

:: Create a virtual environment if it doesn't exist
echo.
echo [2/7] Setting up virtual environment...
IF NOT EXIST "venv" (
    echo         Creating virtual environment...
    py -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        echo [%date% %time%] ERROR: venv creation failed >> %LOGFILE%
        pause
        EXIT /B 1
    )
    echo [%date% %time%] Virtual environment created >> %LOGFILE%
) else (
    echo         Virtual environment already exists.
    echo [%date% %time%] Virtual environment already exists >> %LOGFILE%
)

:: Activate the virtual environment
CALL venv\Scripts\activate
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    echo [%date% %time%] ERROR: venv activation failed >> %LOGFILE%
    pause
    EXIT /B 1
)

:: Upgrade pip
echo.
echo [3/7] Upgrading pip...
python -m pip install --upgrade pip >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] pip upgrade failed, continuing with current version...
    echo [%date% %time%] WARNING: pip upgrade failed >> %LOGFILE%
) else (
    echo         pip upgraded successfully.
    echo [%date% %time%] pip upgraded >> %LOGFILE%
)

:: Install requirements
echo.
echo [4/7] Installing requirements from requirements.txt...
pip install -r requirements.txt >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install requirements. Check install.log for details.
    echo [%date% %time%] ERROR: requirements install failed >> %LOGFILE%
    pause
    EXIT /B 1
)
echo         Requirements installed successfully.
echo [%date% %time%] Requirements installed >> %LOGFILE%

:: Install TTS (optional, non-blocking)
echo.
echo [5/7] Installing TTS (optional)...
pip install TTS >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] TTS installation failed. You can install it manually later with: pip install TTS
    echo [%date% %time%] WARNING: TTS install failed >> %LOGFILE%
) else (
    echo         TTS installed successfully.
    echo [%date% %time%] TTS installed >> %LOGFILE%
)

:: Detect CUDA and install PyTorch
echo.
echo [6/7] Installing PyTorch (detecting CUDA version)...
set TORCH_INDEX=
set TORCH_LABEL=

nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo         NVIDIA GPU detected.
    echo [%date% %time%] NVIDIA GPU detected >> %LOGFILE%
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set DRIVER_VER=%%i
    echo         Driver version: !DRIVER_VER!

    :: Try to get CUDA version from nvidia-smi
    set CUDA_VER=
    for /f "tokens=2 delims=," %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv 2^>nul') do (
        for /f "tokens=*" %%b in ("%%a") do set CUDA_CAP=%%b
    )

    :: Use nvidia-smi output to detect CUDA version
    for /f "tokens=*" %%v in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do set CUDA_LINE=%%v

    if defined CUDA_LINE (
        for /f "tokens=4" %%c in ("!CUDA_LINE!") do set CUDA_VER=%%c
    )

    if defined CUDA_VER (
        echo         CUDA version: !CUDA_VER!
        echo [%date% %time%] CUDA version: !CUDA_VER! >> %LOGFILE%

        :: Extract major version
        set CUDA_MAJOR=
        for /f "tokens=1 delims=." %%m in ("!CUDA_VER!") do set CUDA_MAJOR=%%m

        if "!CUDA_MAJOR!"=="12" (
            set TORCH_INDEX=https://download.pytorch.org/whl/cu121
            set TORCH_LABEL=CUDA 12.x (cu121)
        ) else if "!CUDA_MAJOR!"=="11" (
            set TORCH_INDEX=https://download.pytorch.org/whl/cu118
            set TORCH_LABEL=CUDA 11.x (cu118)
        ) else (
            echo [WARNING] Unsupported CUDA version !CUDA_VER!, falling back to CPU.
            echo [%date% %time%] WARNING: Unsupported CUDA !CUDA_VER!, using CPU >> %LOGFILE%
            set TORCH_LABEL=CPU
        )
    ) else (
        echo [WARNING] Could not detect CUDA version, falling back to CPU.
        echo [%date% %time%] WARNING: CUDA version undetected, using CPU >> %LOGFILE%
        set TORCH_LABEL=CPU
    )
) else (
    echo         No NVIDIA GPU detected, installing CPU version.
    echo [%date% %time%] No NVIDIA GPU, using CPU >> %LOGFILE%
    set TORCH_LABEL=CPU
)

echo         Installing PyTorch for !TORCH_LABEL!...
if defined TORCH_INDEX (
    pip install torch torchvision torchaudio --index-url !TORCH_INDEX! >> %LOGFILE% 2>&1
) else (
    pip install torch torchvision torchaudio >> %LOGFILE% 2>&1
)
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch. Check install.log for details.
    echo [%date% %time%] ERROR: PyTorch install failed >> %LOGFILE%
    pause
    EXIT /B 1
)
echo         PyTorch (!TORCH_LABEL!) installed successfully.
echo [%date% %time%] PyTorch !TORCH_LABEL! installed >> %LOGFILE%

:: Create .env from .env.example if not exists
echo.
echo [7/7] Checking .env configuration...
IF NOT EXIST ".env" (
    IF EXIST ".env.example" (
        copy .env.example .env >nul 2>&1
        IF %ERRORLEVEL% NEQ 0 (
            echo [WARNING] Failed to create .env from .env.example.
            echo [%date% %time%] WARNING: .env creation failed >> %LOGFILE%
        ) else (
            echo         Created .env from .env.example. Please review and edit it.
            echo [%date% %time%] .env created from .env.example >> %LOGFILE%
        )
    ) else (
        echo [WARNING] .env.example not found. Please create .env manually.
        echo [%date% %time%] WARNING: .env.example not found >> %LOGFILE%
    )
) else (
    echo         .env already exists.
    echo [%date% %time%] .env already exists >> %LOGFILE%
)

echo.
echo ============================================
echo   Setup complete!
echo   Log file: %LOGFILE%
echo ============================================
echo [%date% %time%] Setup complete >> %LOGFILE%
pause
ENDLOCAL
