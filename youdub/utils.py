import re
import string
import subprocess
import sys
import time
import os
import urllib.request
from typing import List, Optional, Tuple
import numpy as np
from scipy.io import wavfile
from loguru import logger

def sanitize_filename(filename: str) -> str:
    # Define a set of valid characters
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

    # Keep only valid characters
    sanitized_filename = ''.join(c for c in filename if c in valid_chars)

    # Replace multiple spaces with a single space
    sanitized_filename = re.sub(' +', ' ', sanitized_filename)

    return sanitized_filename


def save_wav(wav: np.ndarray, output_path: str, sample_rate=24000):
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))

def save_wav_norm(wav: np.ndarray, output_path: str, sample_rate=24000):
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))
    
def normalize_wav(wav_path: str) -> None:
    sample_rate, wav = wavfile.read(wav_path)
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(wav_path, sample_rate, wav_norm.astype(np.int16))


PIP_MIRRORS = [
    ("清华 TUNA",     "https://pypi.tuna.tsinghua.edu.cn/simple",        "pypi.tuna.tsinghua.edu.cn"),
    ("阿里云",        "https://mirrors.aliyun.com/pypi/simple/",         "mirrors.aliyun.com"),
    ("腾讯云",        "https://mirrors.cloud.tencent.com/pypi/simple",   "mirrors.cloud.tencent.com"),
    ("华为云",        "https://repo.huaweicloud.com/repository/pypi/simple", "repo.huaweicloud.com"),
    ("官方源",        "https://pypi.org/simple/",                        "pypi.org"),
]


def _detect_best_mirror() -> Optional[str]:
    for name, url, host in PIP_MIRRORS:
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"pip 镜像 {name} ({host}) 可达")
            return url
        except Exception:
            logger.debug(f"pip 镜像 {name} ({host}) 不可达")
    return None


def install_package_with_mirrors(package_spec: str, timeout: int = 300) -> bool:
    pip_index_url = os.environ.get("PIP_INDEX_URL", "")
    if pip_index_url:
        mirrors_to_try = [("用户配置", pip_index_url, "")]
    else:
        best = _detect_best_mirror()
        if best:
            mirrors_to_try = [(name, url, host) for name, url, host in PIP_MIRRORS]
        else:
            mirrors_to_try = [("官方源(默认)", "https://pypi.org/simple/", "pypi.org")]

    for name, index_url, host in mirrors_to_try:
        for attempt in range(1, 3):
            try:
                cmd = [sys.executable, "-m", "pip", "install", package_spec]
                cmd.extend(["-i", index_url, "--timeout", "60"])
                if host:
                    cmd.extend(["--trusted-host", host])
                logger.info(f"[{name}] 安装 {package_spec} (尝试 {attempt}/2)...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0:
                    logger.info(f"[{name}] 安装 {package_spec} 成功")
                    return True
                error_text = (result.stderr or "") + (result.stdout or "")
                logger.warning(f"[{name}] 安装失败: {error_text[:200]}")
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 10))
            except subprocess.TimeoutExpired:
                logger.warning(f"[{name}] 安装超时 (尝试 {attempt}/2)")
            except Exception as e:
                logger.warning(f"[{name}] 安装异常: {e}")
                if attempt < 2:
                    time.sleep(5)
    logger.error(f"所有镜像均安装失败: {package_spec}")
    return False