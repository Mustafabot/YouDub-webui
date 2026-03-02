import os
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "OPENAI_API_KEY": "",
    "OPENAI_API_BASE": "https://api.openai.com/v1",
    "MODEL_NAME": "gpt-3.5-turbo",
    "HF_TOKEN": "",
    "HF_ENDPOINT": "",
    "BYTEDANCE_APPID": "",
    "BYTEDANCE_ACCESS_TOKEN": "",
    "BILI_SESSDATA": "",
    "BILI_BILI_JCT": "",
    "BILI_BASE64": "",
}

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    os.environ.update({k: v for k, v in config.items() if v})

def get_config(key, default=None):
    config = load_config()
    return config.get(key, os.environ.get(key, default))

def set_config(key, value):
    config = load_config()
    config[key] = value
    save_config(config)
    os.environ[key] = value
