# ============================================================================
# 通用工具模块 (utils.py)
# 功能：
#   - sanitize_filename: 清理文件名，移除 Windows 不允许的字符
#   - save_wav: 将 numpy 音频数组保存为 WAV 文件（含归一化处理）
#   - normalize_wav: 读取 WAV 文件并重新归一化音量
#   - install_package_with_mirrors: 使用多镜像源安装 pip 包（自动检测可用镜像）
# ============================================================================

# ---------------------------------------------------------------------------
# 标准库导入
# ---------------------------------------------------------------------------
import re          # 正则表达式，用于文件名清理
import string      # 字符串常量（如标点符号）
import subprocess  # 子进程执行（pip install）
import sys         # Python 解释器路径（sys.executable）
import time        # 重试间隔等待
import os          # 环境变量读取（PIP_INDEX_URL）
import urllib.request  # HTTP 请求，用于检测镜像源可达性
from typing import List, Optional, Tuple  # 类型注解

# ---------------------------------------------------------------------------
# 第三方库导入
# ---------------------------------------------------------------------------
import numpy as np          # 数值计算，处理音频数组
from scipy.io import wavfile  # SciPy WAV 读写
from loguru import logger    # 日志


# ============================================================================
# 文件名清理
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """
    清理文件名字符串，移除 Windows 文件系统不允许的字符。

    清理规则：
    1. 移除 Windows 不允许的字符：< > : " / \ | ?
    2. 移除首尾的空格和点号（Windows 文件名不允许首尾空格/点）
    3. 多个连续空格合并为一个空格
    4. 如果清理后为空，返回默认名 'untitled'

    Args:
        filename: 原始文件名（可能包含非法字符）

    Returns:
        str: 清理后的合法文件名，清理后为空则返回 'untitled'
    """
    # 移除 Windows 不允许的文件名字符：< > : " / \ | ? *
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '', filename)

    # 移除首尾空格和点号（Windows 不允许文件名以空格或点开头/结尾）
    sanitized = sanitized.strip(' .')

    # 将多个连续空格合并为一个空格
    sanitized = re.sub(r' +', ' ', sanitized)

    # 如果清理后为空字符串，返回默认名称
    return sanitized if sanitized else 'untitled'


# ============================================================================
# WAV 音频保存（峰值归一化）
# ============================================================================

def save_wav(wav: np.ndarray, output_path: str, sample_rate=24000):
    """
    将 numpy 音频波形数组保存为 WAV 文件，自动进行峰值归一化。

    处理流程：
    1. 计算音频数据的峰值幅度（绝对值最大值）
    2. 如果峰值接近零（静音），直接写入全零数组
    3. 如果有声音，将所有采样值归一化到 [-32767, 32767] 的 int16 范围
    4. 写入 WAV 文件

    Args:
        wav: numpy 音频数据数组，形状为 (samples,) 或 (samples, channels)
        output_path: 输出 WAV 文件的路径字符串
        sample_rate: 采样率（Hz），默认 24000
    """
    # 计算音频数据中的最大幅值（绝对值峰值）
    peak = np.max(np.abs(wav))

    # 如果峰值非常接近零（静音或几乎静音），直接写出零数组
    if peak < 1e-10:
        # 用 dtype=np.int16 的零数组写入，避免写入全零 float 数组导致格式问题
        wavfile.write(output_path, sample_rate, np.zeros_like(wav, dtype=np.int16))
        return

    # 峰值归一化：将所有采样值缩放到 int16 范围 [-32767, 32767]
    # 32767 是 int16 的最大正数值（相比 32768 留一位防止溢出）
    wav_norm = wav * (32767 / peak)

    # 写入 WAV 文件，转换为 int16 类型
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))


def normalize_wav(wav_path: str) -> None:
    """
    读取 WAV 文件并重新归一化其音量到标准范围。

    用于修复音量过小或过大的 WAV 文件。
    实现方式：读取 -> save_wav（含峰值归一化）-> 覆盖写入原路径。

    Args:
        wav_path: WAV 文件的路径字符串
    """
    # 读取 WAV 文件：sample_rate 为采样率，wav 为音频数据数组
    sample_rate, wav = wavfile.read(wav_path)
    # 调用 save_wav 进行峰值归一化后重新写入（覆盖原文件）
    save_wav(wav, wav_path, sample_rate)


# ============================================================================
# pip 多镜像源安装
# ============================================================================

# 预定义的 pip 镜像源列表（名称、URL、信任主机名）
# 按优先级排序（越靠前越快，以国内用户为考量）
PIP_MIRRORS = [
    ("清华 TUNA",     "https://pypi.tuna.tsinghua.edu.cn/simple",        "pypi.tuna.tsinghua.edu.cn"),
    ("阿里云",        "https://mirrors.aliyun.com/pypi/simple/",         "mirrors.aliyun.com"),
    ("腾讯云",        "https://mirrors.cloud.tencent.com/pypi/simple",   "mirrors.cloud.tencent.com"),
    ("华为云",        "https://repo.huaweicloud.com/repository/pypi/simple", "repo.huaweicloud.com"),
    ("官方源",        "https://pypi.org/simple/",                        "pypi.org"),
]


def _detect_best_mirror() -> Optional[str]:
    """
    检测最快的可用 pip 镜像源。

    遍历 PIP_MIRRORS 列表，对每个镜像源发送 HEAD 请求（超时 5 秒），
    返回第一个可达的镜像源 URL。
    按列表顺序检测，因此列表顺序提供了优先级（通常清华最快）。

    Returns:
        Optional[str]: 可达镜像源的 URL 字符串，如果全部不可达则返回 None
    """
    # 遍历所有预配置的镜像源
    for name, url, host in PIP_MIRRORS:
        try:
            # 发送 HEAD 请求（比 GET 更轻量，只检查可达性）
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"pip 镜像 {name} ({host}) 可达")
            return url  # 返回第一个可达的镜像源
        except Exception:
            # 连接失败则跳过，尝试下一个镜像
            logger.debug(f"pip 镜像 {name} ({host}) 不可达")

    # 所有镜像都不可达
    return None


def install_package_with_mirrors(package_spec: str, timeout: int = 300, extra_args: list = None) -> bool:
    """
    使用多个 pip 镜像源尝试安装 Python 包。

    策略：
    1. 如果环境变量 PIP_INDEX_URL 已设置，优先使用用户自定义镜像
    2. 否则自动检测可达的最快镜像源
    3. 每个镜像尝试 2 次安装（第 1 次失败后短暂等待再试第 2 次）
    4. 当前镜像完全失败后，自动切换到下一个镜像

    Args:
        package_spec: pip 包规范字符串（如 "requests==2.28.0" 或 "torch"）
        timeout: 每次 pip install 的超时时间（秒），默认 300 秒
        extra_args: 额外传递给 pip install 的参数列表（如 ["--no-deps"]）

    Returns:
        bool: True 表示至少有一个镜像成功安装，False 表示所有镜像都失败
    """
    # ---- 确定要尝试的镜像列表 ----
    # 检查环境变量中是否有用户自定义的 pip 镜像源
    pip_index_url = os.environ.get("PIP_INDEX_URL", "")
    if pip_index_url:
        # 用户自定义镜像：只尝试这一个
        mirrors_to_try = [("用户配置", pip_index_url, "")]
    else:
        # 没有用户自定义镜像：自动检测可达镜像
        best = _detect_best_mirror()
        if best:
            # 检测到有可达镜像，尝试所有镜像（按列表顺序）
            mirrors_to_try = [(name, url, host) for name, url, host in PIP_MIRRORS]
        else:
            # 所有镜像检测都失败，使用官方源作为最后的兜底
            mirrors_to_try = [("官方源(默认)", "https://pypi.org/simple/", "pypi.org")]

    # ---- 遍历镜像列表进行安装尝试 ----
    for name, index_url, host in mirrors_to_try:
        for attempt in range(1, 3):  # 每个镜像最多尝试 2 次
            try:
                # 构建 pip install 命令
                cmd = [sys.executable, "-m", "pip", "install", package_spec]
                cmd.extend(["-i", index_url, "--timeout", "60"])  # 指定镜像源和 HTTP 超时
                if host:
                    cmd.extend(["--trusted-host", host])  # 信任非 HTTPS 源

                # 添加额外参数（如 --no-deps）
                if extra_args:
                    cmd.extend(extra_args)

                # 记录本次安装尝试
                logger.info(f"[{name}] 安装 {package_spec} (尝试 {attempt}/2)...")

                # 执行 pip install 子进程
                result = subprocess.run(
                    cmd,
                    capture_output=True,  # 捕获 stdout 和 stderr
                    text=True,            # 以文本形式返回
                    timeout=timeout       # 安装超时
                )

                # 检查安装结果
                if result.returncode == 0:
                    logger.info(f"[{name}] 安装 {package_spec} 成功")
                    return True  # 安装成功，直接返回

                # 安装失败：记录错误信息（截取前 200 字符避免日志过长）
                error_text = (result.stderr or "") + (result.stdout or "")
                logger.warning(f"[{name}] 安装失败: {error_text[:200]}")

                # 如果还有第二次尝试机会，等待后重试（指数退避：2^attempt 秒）
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 10))

            except subprocess.TimeoutExpired:
                # pip install 超时
                logger.warning(f"[{name}] 安装超时 (尝试 {attempt}/2)")

            except Exception as e:
                # 其他异常（如网络断开、进程被终止等）
                logger.warning(f"[{name}] 安装异常: {e}")
                if attempt < 2:
                    # 等待 5 秒后重试
                    time.sleep(5)

    # ---- 所有镜像均失败 ----
    logger.error(f"所有镜像均安装失败: {package_spec}")
    return False
