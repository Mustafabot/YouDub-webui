@echo off
SETLOCAL EnableDelayedExpansion

cd /d "%~dp0"
IF %ERRORLEVEL% NEQ 0 (
    echo [错误] 无法切换到脚本目录: "%~dp0"
    echo 请确认从本地磁盘运行此脚本，不要使用网络路径。
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
    echo [错误] 未找到虚拟环境
    echo.
    echo ============================================
    echo   解决方案
    echo   1. 请先运行 setup_windows.bat 安装依赖
    echo   2. 确认 setup_windows.bat 是否成功完成
    echo   3. 检查是否有权限创建 venv 目录
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: Virtual environment not found >> "%LOGFILE%"
    pause
    EXIT /B 1
)
echo        虚拟环境存在。
echo [%date% %time%] Virtual environment found >> "%LOGFILE%"

:: ============================================
:: [2/6] Activate virtual environment
:: ============================================
echo.
echo [2/6] 激活虚拟环境...
CALL venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 虚拟环境激活失败。
    echo.
    echo ============================================
    echo   解决方案
    echo   1. 删除 venv 文件夹，重新运行 setup_windows.bat
    echo   2. 检查 Python 安装是否完整
    echo   3. 以管理员权限运行脚本
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: Failed to activate virtual environment >> "%LOGFILE%"
    pause
    EXIT /B 1
)
IF NOT DEFINED VIRTUAL_ENV (
    echo.
    echo [错误] 虚拟环境激活失败，环境变量未设置。
    echo.
    echo ============================================
    echo   解决方案
    echo   1. 重新运行 setup_windows.bat
    echo   2. 检查是否有杀毒软件拦截
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: VIRTUAL_ENV not set after activation >> "%LOGFILE%"
    pause
    EXIT /B 1
)
echo        虚拟环境已激活。
echo [%date% %time%] Virtual environment activated >> "%LOGFILE%"

:: ============================================
:: [3/6] Check Python version
:: ============================================
echo.
echo [3/6] 检查 Python 版本...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 虚拟环境中 Python 不可用。
    echo.
    echo ============================================
    echo   解决方案
    echo   1. 删除 venv 文件夹
    echo   2. 重新运行 setup_windows.bat
    echo   3. 确认系统已安装 Python 3.8+
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
echo        检查 Python %PYMAJOR%.%PYMINOR%.x

:: Check via batch comparison (avoids cmd.exe > parsing issue with Python inline code)
set VERSION_OK=0
if !PYMAJOR! GEQ 4 set VERSION_OK=1
if !PYMAJOR! EQU 3 if !PYMINOR! GEQ 8 set VERSION_OK=1
if not !VERSION_OK! EQU 0 goto VERSION_PASS
echo.
echo [错误] Python 版本过低。
echo.
echo ============================================
echo   当前版本: %PYMAJOR%.%PYMINOR%
echo   要求版本: 3.8 或更高
echo.
echo   解决方案：
echo   1. 从 https://www.python.org/downloads/ 下载最新 Python
echo   2. 安装时勾选 "Add Python to PATH"
echo   3. 重新运行 setup_windows.bat
echo ============================================
echo.
echo [%date% %time%] ERROR: Python %PYMAJOR%.%PYMINOR% is too old (requires 3.8+) >> "%LOGFILE%"
pause
EXIT /B 1
:VERSION_PASS
echo        Python 版本符合要求。
echo [%date% %time%] Python %PYMAJOR%.%PYMINOR% version check passed >> "%LOGFILE%"

:: ============================================
:: [4/6] Check critical dependencies
:: ============================================
echo.
echo [4/6] 检查依赖项...

echo        检查 gradio...
python -c "import gradio; print(f'Gradio {gradio.__version__}')" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] Gradio 未安装或损坏。
    echo.
    echo ============================================
    echo   解决方案
    echo   1. 运行 setup_windows.bat 重新安装依赖
    echo   2. 或手动执行: pip install gradio
    echo   3. 检查网络连接和 pip 镜像配置
    echo ============================================
    echo.
    echo [%date% %time%] ERROR: Gradio not installed >> "%LOGFILE%"
    pause
    EXIT /B 1
)
for /f "tokens=*" %%i in ('python -c "import gradio; print(gradio.__version__)"') do set GRADIO_VER=%%i
echo        Gradio %GRADIO_VER% 已安装

echo        检查 torch...
python -c "import torch; print(f'Torch {torch.__version__}')" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] PyTorch 未安装或损坏。
    echo        部分功能可能无法正常运行。
    echo.
    echo   解决方案
    echo   运行 setup_windows.bat 重新安装 PyTorch
    echo.
    echo [%date% %time%] WARNING: PyTorch not found >> "%LOGFILE%"
) else (
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.__version__)"') do set TORCH_VER=%%i
    echo        PyTorch %TORCH_VER% 已安装
    echo [%date% %time%] PyTorch %TORCH_VER% found >> "%LOGFILE%"
)
echo        依赖项检查完成。
echo [%date% %time%] Critical dependencies check passed >> "%LOGFILE%"

:: ============================================
:: [5/6] Check configuration
:: ============================================
echo.
echo [5/6] 检查配置文件...
IF NOT EXIST ".env" (
    echo [警告] 未找到 .env 配置文件。
    echo        将使用默认配置，部分功能可能受限。
    echo.
    echo   提示：
    echo   可从 .env.example 复制创建 .env 文件
    echo   并填入需要的 API 密钥
    echo.
    echo [%date% %time%] WARNING: .env file not found >> "%LOGFILE%"
) else (
    echo        .env 配置文件存在。
    echo [%date% %time%] .env file found >> "%LOGFILE%"
)

:: ============================================
:: [6/6] Prepare to launch
:: ============================================
echo.
echo [6/6] 准备启动应用...

:: Handle share mode via environment variable
set SHARE_FLAG=
if "%YOUDUB_SHARE%"=="1" (
    set SHARE_FLAG=1
)

if "!SHARE_FLAG!"=="1" (
    echo        分享模式已开启 (YOUDUB_SHARE=1^)
    echo        将生成公网 URL 供远程访问。
    echo [%date% %time%] Share mode enabled >> "%LOGFILE%"
)

echo.
echo ============================================
echo   启动完成：
echo ============================================
echo.
echo   访问地址: http://127.0.0.1:19876
echo   停止应用: 按 Ctrl+C
echo.
if "!SHARE_FLAG!"=="1" (
echo   分享模式: 启用，将显示公网 URL。
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
    echo   [错误] 应用程序异常退出
    echo ============================================
    echo.
    echo   请查看上方的错误信息了解详情。
    echo.
    echo   常见解决方案：
    echo   1. 检查端口 19876 是否被占用
    echo   2. 查看 run.log 了解详细日志
    echo   3. 运行 setup_windows.bat 重新安装依赖
    echo   4. 确认 .env 配置是否正确
    echo.
    echo [%date% %time%] ERROR: Application exited with error code %ERRORLEVEL% >> "%LOGFILE%"
    echo.
    pause
    EXIT /B 1
)

echo.
echo ============================================
echo   应用程序已正常退出
-echo ============================================
echo.
echo [%date% %time%] Application exited normally >> "%LOGFILE%"
echo.
pause
ENDLOCAL
