import os
import json
import time
import socket
from pathlib import Path
from dotenv import load_dotenv

CONFIG_FILE = Path(__file__).parent / "config.json"
ENV_FILE = Path(__file__).parent.parent / ".env"

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
    "BYTEDANCE_APPID": "",
    "BYTEDANCE_ACCESS_TOKEN": "",
    "BILI_SESSDATA": "",
    "BILI_BILI_JCT": "",
    "BILI_BASE64": "",
}

REQUIRED_CONFIG = {
    "OPENAI_API_KEY": "翻译服务",
    "HF_TOKEN": "说话者分离",
    "BYTEDANCE_APPID": "语音合成",
    "BYTEDANCE_ACCESS_TOKEN": "语音合成",
    "BILI_SESSDATA": "B站上传",
    "BILI_BILI_JCT": "B站上传",
}

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    os.environ.update({k: str(v) for k, v in config.items() if v})

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

def get_config_status():
    status = {}
    for key in DEFAULT_CONFIG:
        value = get_config(key)
        status[key] = {
            "set": bool(value),
            "required": key in REQUIRED_CONFIG,
            "feature": REQUIRED_CONFIG.get(key, ""),
        }
    return status
