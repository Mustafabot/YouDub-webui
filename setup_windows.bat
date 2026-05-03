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
echo [0/8] Detecting fastest pip mirror...
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
echo [1/7] Checking Python version...
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
echo [错误] 未找到 Python 3.8 或更高版本。
echo.
echo ============================================
echo   解决方案：
echo   1. 从 https://www.python.org/downloads/ 下载安装 Python
echo   2. 安装时勾选 "Add Python to PATH"
echo   3. 安装完成后重新运行此脚本
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
echo [2/7] Setting up virtual environment...
IF NOT EXIST "venv" (
    echo         Creating virtual environment...
    py -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo [错误] 创建虚拟环境失败。
        echo.
        echo ============================================
        echo   解决方案
        echo   1. 以管理员权限运行此脚本
        echo   2. 检查磁盘空间是否充足
        echo   3. 检查是否有权限创建 venv 目录
        echo   4. 暂时关闭杀毒软件后重试
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
    echo [错误] 激活虚拟环境失败。
    echo.
    echo ============================================
    echo   解决方案
    echo   1. 删除 venv 文件夹，重新运行此脚本
    echo   2. 以管理员权限运行此脚本
    echo   3. 检查 Python 安装是否完整
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: venv activation failed >> %LOGFILE%
    pause
    EXIT /B 1
)

:: Upgrade pip
echo.
echo [3/7] Upgrading pip...
python -m pip install --upgrade pip !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [错误] pip 升级失败，将使用当前版本...
    echo [%date% %time%] WARNING: pip upgrade failed >> %LOGFILE%
) else (
    echo         pip upgraded successfully.
    echo [%date% %time%] pip upgraded >> %LOGFILE%
)

:: Install requirements
echo.
echo [4/7] Installing requirements from requirements.txt...
pip install -r requirements.txt !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 安装依赖项失败。
    echo.
    echo ============================================
    echo   解决方案
    echo   1. 查看 install.log 了解详细错误
    echo   2. 检查网络连接是否稳定
    echo   3. 或手动执行: pip install -r requirements.txt
    echo   4. 若使用代理，请检查系统 pip 配置
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
echo [5/7] Installing TTS (optional)...
pip install TTS !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [错误] TTS 安装失败。暂不使用 TTS 功能，可稍后手动执行: pip install TTS
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
            echo [错误] 不支持的 CUDA 版本 !CUDA_VER!，将使用 CPU 版本。
            echo [%date% %time%] WARNING: Unsupported CUDA !CUDA_VER!, using CPU >> %LOGFILE%
            set TORCH_LABEL=CPU
        )
    ) else (
        echo [错误] 无法检测 CUDA 版本，将使用 CPU 版本。
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
    echo [错误] 安装 PyTorch 失败。
    echo.
    echo ============================================
    echo   解决方案
    echo   1. 查看 install.log 了解详细错误
    echo   2. 检查网络连接是否稳定
    echo   3. 若使用 CUDA 版本，请确认驱动已正确
    echo   4. 可尝试手动安装: pip install torch torchvision torchaudio
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
echo [7/8] Verifying dependency compatibility...
python -c "import numba; import numpy; print(f'Numba {numba.__version__} + NumPy {numpy.__version__}')" >> %LOGFILE% 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo         numba/numpy compatibility verified.
    echo [%date% %time%] Dependency check passed >> %LOGFILE%
) else (
    echo [错误] 依赖检查失败。正在尝试重新安装修复...
    echo [%date% %time%] WARNING: Dependency check failed >> %LOGFILE%
    python -m pip install "numpy<2.4" "numba>=0.63" !PIP_MIRROR_OPTS! >> %LOGFILE% 2>&1
    IF %ERRORLEVEL% EQU 0 (
        echo         Fixed dependencies automatically.
        echo [%date% %time%] Dependencies fixed >> %LOGFILE%
    ) else (
        echo [错误] 无法自动修复依赖问题。请查看 install.log 了解详情。
        echo [%date% %time%] WARNING: Auto-fix failed >> %LOGFILE%
    )
)

:: Create .env from .env.example if not exists
echo.
echo [8/8] Checking .env configuration...
IF NOT EXIST ".env" (
    IF EXIST ".env.example" (
        copy .env.example .env >nul 2>&1
        IF %ERRORLEVEL% NEQ 0 (
            echo [错误] 无法从 .env.example 创建 .env 文件。
            echo [%date% %time%] WARNING: .env creation failed >> %LOGFILE%
        ) else (
            echo         Created .env from .env.example. Please review and edit it.
            echo [%date% %time%] .env created from .env.example >> %LOGFILE%
        )
    ) else (
        echo [错误] 未找到 .env.example 文件，请手动创建 .env 文件。
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
