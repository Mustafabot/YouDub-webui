import os
import sys
import json
import time
import socket
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

CONFIG_FILE = Path(__file__).parent / "config.json"
ENV_FILE = Path(__file__).parent.parent / ".env"
PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "bin"

load_dotenv(ENV_FILE, override=True)

DEFAULT_CONFIG = {
    "OPENAI_API_KEY": "",
    "OPENAI_API_BASE": "https://api.openai.com/v1",
    "MODEL_NAME": "gpt-3.5-turbo",
    "OPENAI_API_TEMPERATURE": 1.0,
    "OPENAI_API_TOP_P": 1.0,
    "OPENAI_API_MAX_TOKENS": "",
    "OPENAI_API_EXTRA_BODY": "",
    "HF_TOKEN": "",
    "HF_ENDPOINT": "",
    "PIP_INDEX_URL": "",
    "DOWNLOAD_TIMEOUT": 120,
    "BYTEDANCE_APPID": "",
    "BYTEDANCE_ACCESS_TOKEN": "",
    "BILI_SESSDATA": "",
    "BILI_BILI_JCT": "",
    "BILI_BASE64": "",
    "FFMPEG_PATH": "",
}

REQUIRED_CONFIG = {
    "OPENAI_API_KEY": "翻译服务",
    "HF_TOKEN": "说话者分离",
    "BYTEDANCE_APPID": "语音合成",
    "BYTEDANCE_ACCESS_TOKEN": "语音合成",
    "BILI_SESSDATA": "B站上传",
    "BILI_BILI_JCT": "B站上传",
}

_ffmpeg_cache = {"path": None, "version": None}


def get_bundled_ffmpeg_path():
    """获取项目内置 FFmpeg 的路径"""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    
    ffmpeg_filename = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    ffmpeg_path = BIN_DIR / ffmpeg_filename
    
    if ffmpeg_path.exists() and ffmpeg_path.is_file():
        return str(ffmpeg_path)
    
    return None


def get_ffmpeg_path():
    """
    获取 FFmpeg 可执行文件路径
    优先级：
    1. 配置文件中的 FFMPEG_PATH (最高优先级，保持兼容)
    2. 项目内置 bin/ 目录下的 FFmpeg
    3. 系统 PATH 中的 FFmpeg
    4. 常见安装目录 (保持兼容)
    """
    if _ffmpeg_cache["path"] is not None:
        return _ffmpeg_cache["path"]
    
    configured_path = get_config("FFMPEG_PATH")
    if configured_path:
        configured_path = Path(configured_path)
        if configured_path.exists() and configured_path.is_file():
            _ffmpeg_cache["path"] = str(configured_path)
            return _ffmpeg_cache["path"]
    
    bundled_path = get_bundled_ffmpeg_path()
    if bundled_path:
        _ffmpeg_cache["path"] = bundled_path
        return bundled_path
    
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        _ffmpeg_cache["path"] = system_ffmpeg
        return _ffmpeg_cache["path"]
    
    common_paths = [
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for path in common_paths:
        if Path(path).exists():
            _ffmpeg_cache["path"] = path
            return path
    
    return None


def get_ffmpeg_version():
    """获取 FFmpeg 版本信息"""
    if _ffmpeg_cache["version"] is not None:
        return _ffmpeg_cache["version"]
    
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        return None
    
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            _ffmpeg_cache["version"] = version_line
            return version_line
    except Exception:
        pass
    
    return None


def check_ffmpeg_available():
    """检查 FFmpeg 是否可用并能正常执行"""
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        return False, "FFmpeg 未找到，请运行自动下载或在配置中设置 FFMPEG_PATH"
    
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, f"FFmpeg 执行失败，请检查文件完整性。路径: {ffmpeg_path}"
    except Exception as e:
        return False, f"FFmpeg 无法执行: {str(e)}。路径: {ffmpeg_path}"
    
    return True, f"FFmpeg 可用: {ffmpeg_path}"


def ensure_ffmpeg_available(auto_download=True):
    """
    确保 FFmpeg 可用，不可用时尝试自动下载
    
    Args:
        auto_download: 是否允许自动下载
    
    Returns:
        (是否可用, 信息或错误)
    """
    available, msg = check_ffmpeg_available()
    if available:
        return True, msg
    
    if not auto_download:
        return False, msg
    
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    
    try:
        from download_ffmpeg import download_ffmpeg
        logger.info("FFmpeg 未找到，正在尝试自动下载...")
        success, result = download_ffmpeg()
        if success:
            _ffmpeg_cache["path"] = None
            _ffmpeg_cache["version"] = None
            return check_ffmpeg_available()
        else:
            return False, f"自动下载失败: {result}。请手动安装 FFmpeg 或在配置中设置 FFMPEG_PATH。"
    except ImportError as e:
        return False, f"无法加载下载模块: {e}。请手动安装 FFmpeg 或在配置中设置 FFMPEG_PATH。"


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if config.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = config["HF_ENDPOINT"]
            os.environ["HUGGINGFACE_HUB_URL"] = config["HF_ENDPOINT"]
        return config
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    os.environ.update({k: str(v) for k, v in config.items() if v})
    if config.get("HF_ENDPOINT"):
        os.environ["HUGGINGFACE_HUB_URL"] = config["HF_ENDPOINT"]


def get_config(key, default=None):
    config = load_config()
    value = config.get(key)
    if value:
        return value
    env_value = os.environ.get(key)
    if env_value:
        return env_value
    return default


def set_config(key, value):
    config = load_config()
    config[key] = value
    save_config(config)
    os.environ[key] = value


def validate_config():
    missing = {}
    for key, feature in REQUIRED_CONFIG.items():
        if not get_config(key):
            missing[key] = feature
    return missing


_network_cache = {"online": None, "timestamp": 0}
_NETWORK_CACHE_TTL = 30


def check_network():
    now = time.time()
    if _network_cache["online"] is not None and now - _network_cache["timestamp"] < _NETWORK_CACHE_TTL:
        return _network_cache["online"]
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        _network_cache["online"] = True
    except (socket.timeout, socket.error, OSError):
        _network_cache["online"] = False
    _network_cache["timestamp"] = now
    return _network_cache["online"]


def get_offline_capabilities():
    return {
        "available": ["人声分离", "语音识别", "视频合成"],
        "unavailable": ["视频下载", "字幕翻译", "语音合成-火山引擎", "上传B站"],
    }


def is_offline_mode():
    """检查是否处于离线模式"""
    return not check_network()


def get_hf_local_files_only():
    """根据网络状态返回 HuggingFace 模型加载时的 local_files_only 参数"""
    return is_offline_mode()


def get_config_status():
    status = {}
    for key in DEFAULT_CONFIG:
        value = get_config(key)
        status[key] = {
            "set": bool(value),
            "required": key in REQUIRED_CONFIG,
            "feature": REQUIRED_CONFIG.get(key, ""),
        }
    
    ffmpeg_available, ffmpeg_msg = check_ffmpeg_available()
    status["FFMPEG"] = {
        "set": ffmpeg_available,
        "required": True,
        "feature": "视频下载/分离/合成",
        "message": ffmpeg_msg,
        "path": get_ffmpeg_path(),
        "can_download": True,
    }

    try:
        from .model_manager import check_all_models_status
        model_statuses = check_all_models_status()
        downloaded = sum(1 for s in model_statuses.values() if s["downloaded"])
        total = len(model_statuses)
        status["MODELS"] = {
            "set": downloaded == total,
            "required": True,
            "feature": "AI模型",
            "message": f"已下载 {downloaded}/{total} 个模型",
            "downloaded": downloaded,
            "total": total,
            "models": model_statuses,
        }
    except Exception as e:
        status["MODELS"] = {
            "set": False,
            "required": True,
            "feature": "AI模型",
            "message": f"模型状态检查失败: {e}",
        }

    return status
