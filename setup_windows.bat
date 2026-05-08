@echo off
SETLOCAL EnableDelayedExpansion

set LOGFILE=install.log
echo [%date% %time%] YouDub Setup Started > %LOGFILE%

echo ============================================
echo   YouDub Windows Setup
echo ============================================
echo.

:: Auto-detect pip mirror for China users
set PIP_MIRROR=
set PIP_MIRROR_HOST=
echo [0/9] Detecting fastest pip mirror...
for %%m in (
    "https://pypi.tuna.tsinghua.edu.cn/simple|pypi.tuna.tsinghua.edu.cn"
    "https://mirrors.aliyun.com/pypi/simple/|mirrors.aliyun.com"
    "https://mirrors.cloud.tencent.com/pypi/simple|mirrors.cloud.tencent.com"
    "https://repo.huaweicloud.com/repository/pypi/simple|repo.huaweicloud.com"
) do (
    for /f "tokens=1,2 delims=|" %%a in ("%%m") do (
        if "!PIP_MIRROR!"=="" (
            curl -s --connect-timeout 3 %%a >nul 2>&1
            if !ERRORLEVEL! EQU 0 (
                set PIP_MIRROR=%%a
                set PIP_MIRROR_HOST=%%b
                echo         Found mirror: %%b
                echo [%date% %time%] Using mirror: %%b >> %LOGFILE%
            )
        )
    )
)
if "!PIP_MIRROR!"=="" (
    echo         No mirror detected, using default PyPI.
    echo [%date% %time%] No mirror, using default PyPI >> %LOGFILE%
    set PIP_MIRROR_OPTS=
) else (
    set PIP_MIRROR_OPTS=-i !PIP_MIRROR! --trusted-host !PIP_MIRROR_HOST!
)

:: Check for Python 3.8+
echo [1/9] Checking Python version...
for /f "tokens=*" %%i in ('py -c "import sys; ver=sys.version.split()[0].split('.'); print(int(ver[0])); print(int(ver[1]))"') do (
  if not defined PYMAJOR (
    set PYMAJOR=%%i
  ) else (
    set PYMINOR=%%i
  )
)
:: Check via batch comparison (avoids cmd.exe > parsing issue)
set PYVER_OK=0
if !PYMAJOR! GEQ 4 set PYVER_OK=1
if !PYMAJOR! EQU 3 if !PYMINOR! GEQ 8 set PYVER_OK=1
if not !PYVER_OK! EQU 0 goto PYVER_PASS
echo.
echo [ERROR] Python 3.8 or higher is required but not found.
echo.
echo ============================================
echo   Troubleshooting
echo   1. Download and install Python from https://www.python.org/downloads/
echo   2. Select "Add Python to PATH" during installation
echo   3. Re-run this script after installation is complete
echo ============================================
echo.
echo [%date% %time%] ERROR: Python 3.8+ not found >> %LOGFILE%
pause
EXIT /B 1
:PYVER_PASS
echo         Found Python %PYMAJOR%.%PYMINOR%.x
echo [%date% %time%] Python %PYMAJOR%.%PYMINOR% found >> %LOGFILE%

:: Create a virtual environment if it doesn't exist
echo.
echo [2/9] Setting up virtual environment...
IF NOT EXIST "venv" (
    echo         Creating virtual environment...
    py -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        echo.
        echo ============================================
        echo   Troubleshooting
        echo   1. Run this script as Administrator
        echo   2. Check if disk space is sufficient
        echo   3. Check write permissions to the venv directory
        echo   4. Temporarily disable antivirus and retry
        echo ============================================
        echo.
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
    echo.
    echo [ERROR] Failed to activate virtual environment.
    echo.
    echo ============================================
    echo   Troubleshooting
    echo   1. Delete the venv folder, then re-run this script
    echo   2. Run this script as Administrator
    echo   3. Verify your Python installation is working
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: venv activation failed >> %LOGFILE%
    pause
    EXIT /B 1
)

:: Upgrade pip
echo.
echo [3/9] Upgrading pip...
python -m pip install --upgrade pip !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] pip upgrade failed, continuing with current version...
    echo [%date% %time%] WARNING: pip upgrade failed >> %LOGFILE%
) else (
    echo         pip upgraded successfully.
    echo [%date% %time%] pip upgraded >> %LOGFILE%
)

:: Install requirements
echo.
echo [4/9] Installing requirements from requirements.txt...
pip install -r requirements.txt !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to install requirements.
    echo.
    echo ============================================
    echo   Troubleshooting
    echo   1. Check install.log for detailed errors
    echo   2. Verify network connection is stable
    echo   3. Try manually: pip install -r requirements.txt
    echo   4. Or install using system Python pip
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: requirements install failed >> %LOGFILE%
    pause
    EXIT /B 1
)
echo         Requirements installed successfully.
echo [%date% %time%] Requirements installed >> %LOGFILE%

:: Install TTS (optional, non-blocking)
echo.
echo [5/9] Installing TTS (optional)...
pip install TTS !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] TTS install failed. TTS features will be unavailable. You can install manually later: pip install TTS
    echo [%date% %time%] WARNING: TTS install failed >> %LOGFILE%
) else (
    echo         TTS installed successfully.
    echo [%date% %time%] TTS installed >> %LOGFILE%
)

:: Install IndexTTS (optional, non-blocking, from GitHub)
echo.
echo [6/9] Installing IndexTTS (optional, from GitHub)...
echo         Downloading from https://github.com/index-tts/index-tts.git ...
pip install git+https://github.com/index-tts/index-tts.git !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] IndexTTS install failed. You can install later via WebUI "Download missing models" button.
    echo [%date% %time%] WARNING: IndexTTS install failed >> %LOGFILE%
) else (
    echo         IndexTTS installed successfully.
    echo [%date% %time%] IndexTTS installed >> %LOGFILE%
)

:: Detect CUDA and install PyTorch
echo.
echo [7/9] Installing PyTorch (detecting CUDA version)...
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
            echo [WARNING] Unsupported CUDA version !CUDA_VER!, falling back to CPU version.
            echo [%date% %time%] WARNING: Unsupported CUDA !CUDA_VER!, using CPU >> %LOGFILE%
            set TORCH_LABEL=CPU
        )
    ) else (
        echo [WARNING] Could not detect CUDA version, falling back to CPU version.
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
    pip install torch torchvision torchaudio --index-url !TORCH_INDEX! !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
) else (
    pip install torch torchvision torchaudio !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
)
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to install PyTorch.
    echo.
    echo ============================================
    echo   Troubleshooting
    echo   1. Check install.log for detailed errors
    echo   2. Verify network connection is stable
    echo   3. If using CUDA version, verify drivers are correct
    echo   4. Try manual install: pip install torch torchvision torchaudio
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: PyTorch install failed >> %LOGFILE%
    pause
    EXIT /B 1
)
echo         PyTorch (!TORCH_LABEL!) installed successfully.
echo [%date% %time%] PyTorch !TORCH_LABEL! installed >> %LOGFILE%

:: Verify numba/numpy compatibility
echo.
echo [8/9] Verifying dependency compatibility...
python -c "import numba; import numpy; print(f'Numba {numba.__version__} + NumPy {numpy.__version__}')" >> %LOGFILE% 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo         numba/numpy compatibility verified.
    echo [%date% %time%] Dependency check passed >> %LOGFILE%
) else (
    echo [WARNING] Dependency check failed. Attempting auto-fix...
    echo [%date% %time%] WARNING: Dependency check failed >> %LOGFILE%
    python -m pip install "numpy<2.4" "numba>=0.63" !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
    IF %ERRORLEVEL% EQU 0 (
        echo         Fixed dependencies automatically.
        echo [%date% %time%] Dependencies fixed >> %LOGFILE%
    ) else (
        echo [WARNING] Could not auto-fix dependencies. Check install.log for details.
        echo [%date% %time%] WARNING: Auto-fix failed >> %LOGFILE%
    )
)

:: Create .env from .env.example if not exists
echo.
echo [9/9] Checking .env configuration...
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
        echo [WARNING] .env.example not found. Please create .env file manually.
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
