# ============================================================================
# 配置管理模块 (config.py)
# 功能: 管理 YouDub-webui 的所有配置项，包括:
#   - FFmpeg/FFprobe 查找与可用性检测（多级查找 + 缓存）
#   - config.json 的读取/写入（原子写入保证文件完整性）
#   - 网络连通性检测（带 TTL 缓存）
#   - 配置状态汇总（给 UI 展示）
#   - 各配置项的默认值、必填项校验
# ============================================================================

# ---------------------------------------------------------------------------
# 标准库导入
# ---------------------------------------------------------------------------
import os       # 环境变量读写、路径操作
import sys      # 平台判断（win/linux）、sys.path 管理
import json     # config.json 的序列化/反序列化
import time     # 网络缓存 TTL 时间戳
import socket   # 网络连通性检测（TCP 连接测试）
import shutil   # 文件复制（损坏配置备份）、查找系统 PATH 中的可执行文件
import subprocess  # 执行 FFmpeg/FFprobe 版本查询
from pathlib import Path  # 跨平台路径对象

# ---------------------------------------------------------------------------
# 第三方库导入
# ---------------------------------------------------------------------------
from dotenv import load_dotenv  # 加载 .env 文件到环境变量
from loguru import logger       # 结构化日志

# ============================================================================
# 全局路径常量
# ============================================================================

# config.json 配置文件路径，位于 youdub/ 目录下
CONFIG_FILE = Path(__file__).parent / "config.json"
# .env 环境变量文件路径，位于项目根目录
ENV_FILE = Path(__file__).parent.parent / ".env"
# 项目根目录（youdub/ 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent
# AI 模型存放根目录
MODEL_ROOT = PROJECT_ROOT / "models"
# 内置可执行文件目录（FFmpeg 等）
BIN_DIR = PROJECT_ROOT / "bin"

# 加载 .env 文件，已存在的环境变量会被覆盖（override=True）
load_dotenv(ENV_FILE, override=True)

# ============================================================================
# 默认配置字典
# 所有配置项的默认值集中在此，UI 初始化时以此为准
# ============================================================================
DEFAULT_CONFIG = {
    # ---------- OpenAI 翻译服务 ----------
    "OPENAI_API_KEY": "",                    # OpenAI API 密钥
    "OPENAI_API_BASE": "https://api.openai.com/v1",  # API 端点地址（可更换为代理）
    "MODEL_NAME": "gpt-3.5-turbo",           # 翻译使用的模型名
    "OPENAI_API_TEMPERATURE": 1.0,           # 生成温度（随机性）
    "OPENAI_API_TOP_P": 1.0,                 # Top-P 采样参数
    "OPENAI_API_MAX_TOKENS": "",             # 最大生成 token 数（空 = 不限制）
    "OPENAI_API_EXTRA_BODY": "",             # 额外请求体参数（JSON 字符串）

    # ---------- 翻译提示词 ----------
    "TRANSLATION_SYSTEM_PROMPT": "",          # 翻译系统提示词
    "TRANSLATION_FEWSHOT_EXAMPLES": "",       # 翻译少样本示例
    "SUMMARY_SYSTEM_PROMPT": "",             # 摘要系统提示词
    "SUMMARY_TRANSLATION_PROMPT": "",        # 摘要翻译提示词

    # ---------- HuggingFace ----------
    "HF_TOKEN": "",                          # HuggingFace 访问令牌
    "HF_ENDPOINT": "",                       # HuggingFace 镜像端点

    # ---------- pip 镜像 ----------
    "PIP_INDEX_URL": "",                     # pip 镜像源 URL

    # ---------- 下载超时 ----------
    "DOWNLOAD_TIMEOUT": 120,                 # 下载超时秒数

    # ---------- 火山引擎 TTS ----------
    "BYTEDANCE_APPID": "",                   # 火山引擎应用 ID
    "BYTEDANCE_ACCESS_TOKEN": "",            # 火山引擎访问令牌

    # ---------- B站上传 ----------
    "BILI_SESSDATA": "",                     # B站登录凭据 sessdata
    "BILI_BILI_JCT": "",                     # B站登录凭据 bili_jct
    "BILI_BASE64": "",                       # B站登录凭据 base64

    # ---------- FFmpeg ----------
    "FFMPEG_PATH": "",                       # 自定义 FFmpeg 路径

    # ---------- TTS 变速参数 ----------
    "TTS_STRETCH_MIN_SPEED": 0.6,            # TTS 最小拉伸速度
    "TTS_STRETCH_MAX_SPEED": 1.3,            # TTS 最大拉伸速度

    # ---------- IndexTTS 模型 ----------
    "INDEXTTS_MODEL_DIR": "models/index-tts",          # IndexTTS 模型目录
    "INDEXTTS_USE_FP16": True,                         # 是否使用 FP16 推理（节省显存）
    "INDEXTTS_USE_DEEPSPEED": False,                   # 是否使用 DeepSpeed 加速
}

# ============================================================================
# 必填配置项声明
# 键 = 配置名，值 = 所属功能（用于错误提示）
# ============================================================================
REQUIRED_CONFIG = {
    "OPENAI_API_KEY": "翻译服务",
    "HF_TOKEN": "说话者分离",
    "BYTEDANCE_APPID": "语音合成",
    "BYTEDANCE_ACCESS_TOKEN": "语音合成",
    "BILI_SESSDATA": "B站上传",
    "BILI_BILI_JCT": "B站上传",
}

# ============================================================================
# FFmpeg/FFprobe 路径与版本缓存
# 避免重复查找，提高性能
# ============================================================================
_ffmpeg_cache = {"path": None, "version": None}   # FFmpeg 路径与版本缓存
_ffprobe_cache = {"path": None, "version": None}  # FFprobe 路径与版本缓存


# ============================================================================
# FFmpeg 查找与检测
# ============================================================================

def get_bundled_ffmpeg_path():
    """
    获取项目内置 bin/ 目录下的 FFmpeg 路径。

    流程：
    1. 确保 bin/ 目录存在
    2. 根据操作系统确定可执行文件名（.exe 后缀 vs 无后缀）
    3. 检查文件是否存在

    Returns:
        str 或 None: 找到返回绝对路径字符串，否则返回 None
    """
    # 确保 bin/ 目录存在
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    # Windows 下需要 .exe 后缀
    ffmpeg_filename = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    # 拼接完整路径
    ffmpeg_path = BIN_DIR / ffmpeg_filename
    # 检查文件是否存在且为普通文件（非目录）
    if ffmpeg_path.exists() and ffmpeg_path.is_file():
        return str(ffmpeg_path)  # 返回字符串路径
    return None  # 未找到


def get_ffmpeg_path():
    """
    获取 FFmpeg 可执行文件路径。

    查找优先级（从高到低）：
    1. 配置文件中用户手动设置的 FFMPEG_PATH（最高优先级，保持向后兼容）
    2. 项目内置 bin/ 目录下的 FFmpeg（自动下载存放位置）
    3. 系统 PATH 环境变量中的 FFmpeg（shutil.which）
    4. 常见安装目录（Windows: Program Files, C:\\ffmpeg; Linux: /usr/local/bin, /usr/bin）

    使用全局缓存 _ffmpeg_cache["path"] 避免重复查找。

    Returns:
        str 或 None: 找到返回路径字符串，否则返回 None
    """
    # 如果缓存中已有路径，直接返回
    if _ffmpeg_cache["path"] is not None:
        return _ffmpeg_cache["path"]

    # 优先级 1: 用户配置的 FFMPEG_PATH
    configured_path = get_config("FFMPEG_PATH")
    if configured_path:
        configured_path = Path(configured_path)  # 转为 Path 对象便于校验
        if configured_path.exists() and configured_path.is_file():
            _ffmpeg_cache["path"] = str(configured_path)
            return _ffmpeg_cache["path"]

    # 优先级 2: 项目内置 bin/ 目录
    bundled_path = get_bundled_ffmpeg_path()
    if bundled_path:
        _ffmpeg_cache["path"] = bundled_path
        return bundled_path

    # 优先级 3: 系统 PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        _ffmpeg_cache["path"] = system_ffmpeg
        return _ffmpeg_cache["path"]

    # 优先级 4: 常见安装目录（硬编码兜底）
    common_paths = [
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",  # Windows 标准安装
        r"C:\ffmpeg\bin\ffmpeg.exe",                # Windows 备选安装
        "/usr/local/bin/ffmpeg",                    # Linux/macOS 常见位置
        "/usr/bin/ffmpeg",                          # Linux 系统包管理器安装位置
    ]
    for path in common_paths:
        if Path(path).exists():
            _ffmpeg_cache["path"] = path
            return path

    # 所有方式均未找到
    return None


def get_ffprobe_path():
    """
    获取 FFprobe 可执行文件路径（FFmpeg 套件中的媒体探测工具）。

    查找优先级（从高到低）：
    1. 配置中 FFMPEG_PATH 同目录下的 ffprobe（FFmpeg 通常附带 ffprobe）
    2. 项目内置 bin/ 目录下的 ffprobe
    3. 系统 PATH 中的 ffprobe
    4. 常见安装目录

    使用全局缓存 _ffprobe_cache["path"] 避免重复查找。

    Returns:
        str 或 None: 找到返回路径字符串，否则返回 None
    """
    # 如果缓存中已有路径，直接返回
    if _ffprobe_cache["path"] is not None:
        return _ffprobe_cache["path"]

    # 优先级 1: 配置的 FFMPEG_PATH 同目录下的 ffprobe
    configured_path = get_config("FFMPEG_PATH")
    if configured_path:
        configured_path = Path(configured_path)
        # 在同一父目录下查找 ffprobe（与 ffmpeg 同目录）
        ffprobe_path = configured_path.parent / (
            "ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"
        )
        if ffprobe_path.exists() and ffprobe_path.is_file():
            _ffprobe_cache["path"] = str(ffprobe_path)
            return _ffprobe_cache["path"]

    # 优先级 2: 项目内置 bin/ 目录
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    ffprobe_filename = "ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"
    bundled_ffprobe = BIN_DIR / ffprobe_filename
    if bundled_ffprobe.exists() and bundled_ffprobe.is_file():
        _ffprobe_cache["path"] = str(bundled_ffprobe)
        return _ffprobe_cache["path"]

    # 优先级 3: 系统 PATH
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        _ffprobe_cache["path"] = system_ffprobe
        return _ffprobe_cache["path"]

    # 优先级 4: 常见安装目录
    common_paths = [
        r"C:\Program Files\FFmpeg\bin\ffprobe.exe",  # Windows 标准安装
        r"C:\ffmpeg\bin\ffprobe.exe",                # Windows 备选安装
        "/usr/local/bin/ffprobe",                    # Linux/macOS
        "/usr/bin/ffprobe",                          # Linux 系统包安装
    ]
    for path in common_paths:
        if Path(path).exists():
            _ffprobe_cache["path"] = path
            return path

    # 所有方式均未找到
    return None


def get_ffmpeg_version():
    """
    获取 FFmpeg 的版本信息。

    通过执行 `ffmpeg -version` 命令并解析第一行输出来获取版本字符串。
    结果会缓存到 _ffmpeg_cache["version"] 中，避免重复执行子进程。

    Returns:
        str 或 None: 版本信息字符串（如 "ffmpeg version 4.4 ..."），失败返回 None
    """
    # 如果版本信息已缓存，直接返回
    if _ffmpeg_cache["version"] is not None:
        return _ffmpeg_cache["version"]

    # 先确保能找到 FFmpeg 路径
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        return None  # 找不到 FFmpeg，无法获取版本

    # 执行 ffmpeg -version 命令获取版本信息
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],   # 命令参数
            capture_output=True,          # 捕获 stdout/stderr
            text=True,                    # 以文本模式返回（而非 bytes）
            timeout=5                     # 超时时间 5 秒
        )
        if result.returncode == 0:
            # 取第一行作为版本信息（标准格式：ffmpeg version x.y.z ...）
            version_line = result.stdout.split("\n")[0]
            _ffmpeg_cache["version"] = version_line  # 写入缓存
            return version_line
    except Exception:
        # 任何异常（文件不存在、超时、无权限等）都不抛出，静默失败
        pass

    return None  # 获取失败


def check_ffmpeg_available():
    """
    检查 FFmpeg 是否可用并能正常执行。

    同时检查 FFmpeg 和 FFprobe 两个工具：
    1. 查找 FFmpeg 路径
    2. 执行 ffmpeg -version 验证可执行
    3. 查找 FFprobe 路径
    4. 执行 ffprobe -version 验证可执行

    返回值是 (bool, str) 元组，便于上层直接用于 UI 提示。

    Returns:
        tuple[bool, str]:
            - (True, 成功消息)  两者都可用
            - (False, 错误消息) 任一不可用，消息包含具体原因和提示
    """
    # ---- 检查 FFmpeg ----
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        return False, "FFmpeg 未找到，请运行自动下载或在配置中设置 FFMPEG_PATH"

    # 执行 ffmpeg -version 验证其可执行
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

    # ---- 检查 FFprobe ----
    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path:
        return False, (
            f"FFprobe 未找到。FFmpeg 可用但缺少 FFprobe。"
            f"请重新运行自动下载或手动安装完整 FFmpeg 包。FFmpeg 路径: {ffmpeg_path}"
        )

    # 执行 ffprobe -version 验证其可执行
    try:
        result = subprocess.run(
            [ffprobe_path, "-version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, f"FFprobe 执行失败，请检查文件完整性。路径: {ffprobe_path}"
    except Exception as e:
        return False, f"FFprobe 无法执行: {str(e)}。路径: {ffprobe_path}"

    # 两者均可用
    return True, f"FFmpeg 可用: {ffmpeg_path}, FFprobe 可用: {ffprobe_path}"


def ensure_ffmpeg_available(auto_download=True):
    """
    确保 FFmpeg 可用，若不可用且允许自动下载则尝试自动下载。

    流程：
    1. 先调用 check_ffmpeg_available() 检查当前状态
    2. 如果已可用，直接返回
    3. 如果不可用且 auto_download=True，尝试调用 download_ffmpeg 模块下载
    4. 下载后清除缓存，重新检查可用性

    Args:
        auto_download: 是否允许自动下载（默认 True）

    Returns:
        tuple[bool, str]:
            - (True, 消息)  可用
            - (False, 消息) 不可用，消息包含具体原因
    """
    # 先检查当前 FFmpeg 是否可用
    available, msg = check_ffmpeg_available()
    if available:
        return True, msg

    # 如果不可用且不允许自动下载，直接返回失败
    if not auto_download:
        return False, msg

    # 将 scripts/ 目录加入 sys.path，以便导入 download_ffmpeg 模块
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # 尝试导入并执行自动下载
    try:
        from download_ffmpeg import download_ffmpeg
        logger.info("FFmpeg 未找到，正在尝试自动下载...")
        success, result = download_ffmpeg()
        if success:
            # 下载成功后清除缓存，确保重新查找
            _ffmpeg_cache["path"] = None
            _ffmpeg_cache["version"] = None
            _ffprobe_cache["path"] = None
            _ffprobe_cache["version"] = None
            # 重新检查可用性
            return check_ffmpeg_available()
        else:
            return False, f"自动下载失败: {result}。请手动安装 FFmpeg 或在配置中设置 FFMPEG_PATH。"
    except ImportError as e:
        return False, f"无法加载下载模块: {e}。请手动安装 FFmpeg 或在配置中设置 FFMPEG_PATH。"


# ============================================================================
# config.json 读写（原子写入）
# ============================================================================

def load_config():
    """
    加载配置文件 config.json。

    流程：
    1. 检查 CONFIG_FILE 是否存在
    2. 存在则读取 JSON 内容
    3. JSON 损坏时：备份损坏文件、返回默认配置
    4. 若配置中有 HF_ENDPOINT，同步设置环境变量
    5. 文件不存在则返回 DEFAULT_CONFIG 副本

    Returns:
        dict: 配置字典
    """
    if CONFIG_FILE.exists():
        # 尝试读取并解析 JSON
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            # JSON 损坏：记录错误、备份文件、返回默认配置
            logger.error(f"配置文件 {CONFIG_FILE} 损坏: {e}，将使用默认配置")
            import shutil as _shutil
            backup_path = str(CONFIG_FILE) + ".backup"
            _shutil.copy2(CONFIG_FILE, backup_path)
            logger.info(f"已备份损坏配置到 {backup_path}")
            return DEFAULT_CONFIG.copy()

        # 如果配置中设置了 HF_ENDPOINT，同步到环境变量
        if config.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = config["HF_ENDPOINT"]
            os.environ["HUGGINGFACE_HUB_URL"] = config["HF_ENDPOINT"]

        return config

    # 配置文件不存在，返回默认配置的副本
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """
    保存配置到 config.json（原子写入，避免写入中断导致文件损坏）。

    流程：
    1. 先写入到临时文件 config.json.tmp
    2. 写入成功后通过 tmp_path.replace() 原子替换原文件
    3. 将所有配置项同步到环境变量
    4. 特殊处理 HF_ENDPOINT 的额外环境变量

    Args:
        config: 要保存的配置字典
    """
    # 原子写入：先写临时文件
    tmp_path = CONFIG_FILE.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # replace() 原子替换原文件（比 rename 更安全）
    tmp_path.replace(CONFIG_FILE)

    # 将所有配置项同步到 os.environ，便于运行时读取
    os.environ.update({k: str(v) for k, v in config.items() if v is not None})

    # 特殊处理：HF_ENDPOINT 需要设置两个环境变量
    if config.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = config["HF_ENDPOINT"]
        os.environ["HUGGINGFACE_HUB_URL"] = config["HF_ENDPOINT"]


# ============================================================================
# 配置项读写接口
# ============================================================================

def get_config(key, default=None):
    """
    获取配置项的值。

    读取顺序：
    1. 从 config.json 中读取
    2. 若 JSON 中不存在或为 None，从环境变量读取
    3. 若都未找到，返回 default

    Args:
        key: 配置项名称（字符串）
        default: 当配置项不存在时的默认返回值

    Returns:
        配置项的值（类型由配置内容决定）
    """
    config = load_config()  # 加载当前配置
    value = config.get(key)  # 从配置字典中获取
    if value is not None:
        return value  # JSON 中有值，直接返回
    # JSON 中没有值，尝试从环境变量获取
    env_value = os.environ.get(key)
    if env_value is not None:
        return env_value
    # 都未找到，返回默认值
    return default


def set_config(key, value):
    """
    设置配置项的值并持久化保存。

    流程：
    1. 加载当前配置
    2. 修改指定键的值
    3. 保存到 config.json（原子写入）
    4. 同步到环境变量

    Args:
        key: 配置项名称（字符串）
        value: 配置项的值
    """
    config = load_config()      # 加载当前配置
    config[key] = value          # 修改值
    save_config(config)          # 持久化保存
    os.environ[key] = value      # 同步到环境变量，确保本次运行时生效


# ============================================================================
# 配置校验
# ============================================================================

def validate_config():
    """
    校验所有必填配置项是否已填写。

    遍历 REQUIRED_CONFIG 字典，检查每个必填项的值是否为空。
    用于 UI 启动前的配置完整性检查。

    Returns:
        dict: 缺失的配置项字典，格式 {配置名: 所属功能}
              如果全部已填写，返回空字典 {}
    """
    missing = {}
    for key, feature in REQUIRED_CONFIG.items():
        if not get_config(key):  # 值为空（None/空字符串/False）
            missing[key] = feature  # 记录缺失项及其所属功能
    return missing


# ============================================================================
# 网络连通性检测（带 TTL 缓存）
# ============================================================================

# 网络缓存 TTL（秒）：30 秒内重复检测直接返回缓存结果
_NETWORK_CACHE_TTL = 30
_network_cache = {"online": None, "timestamp": 0}  # 缓存：是否在线 + 时间戳


def check_network():
    """
    检测当前网络是否可用（能否访问互联网）。

    检测方式：尝试 TCP 连接 Google DNS (8.8.8.8:53)。
    使用带 TTL 的缓存，避免频繁检测。

    缓存机制：
    - 30 秒内重复调用直接返回缓存结果
    - 超过 30 秒重新检测并更新缓存

    Returns:
        bool: True 表示网络可用，False 表示不可用
    """
    now = time.time()  # 当前时间戳

    # 如果缓存有效且未过期，直接返回缓存结果
    if _network_cache["online"] is not None and now - _network_cache["timestamp"] < _NETWORK_CACHE_TTL:
        return _network_cache["online"]

    # 缓存过期或无缓存，重新检测
    try:
        # TCP 连接 Google DNS（8.8.8.8:53），超时 5 秒
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        _network_cache["online"] = True   # 连接成功，网络可用
    except (socket.timeout, socket.error, OSError):
        # 超时或无网络错误
        _network_cache["online"] = False  # 连接失败，网络不可用

    _network_cache["timestamp"] = now  # 更新时间戳
    return _network_cache["online"]


def get_offline_capabilities():
    """
    获取离线模式下的可用/不可用功能列表。

    在无网络时，部分功能（如视频下载、翻译、TTS 云服务、B站上传）无法使用。
    此函数用于 UI 提示用户哪些功能在离线模式下受限。

    Returns:
        dict: 包含 "available"（可用功能列表）和 "unavailable"（不可用功能列表）
    """
    return {
        # 本地处理功能，不需要网络
        "available": ["人声分离", "语音识别", "视频合成"],
        # 需要网络的在线服务
        "unavailable": ["视频下载", "字幕翻译", "语音合成-火山引擎", "上传B站"],
    }


def is_offline_mode():
    """
    判断当前是否处于离线模式。

    简单地反转网络检测结果。

    Returns:
        bool: True 表示离线模式，False 表示在线模式
    """
    return not check_network()


def get_hf_local_files_only():
    """
    获取 HuggingFace 是否应仅使用本地文件。
    离线时 HF 不应尝试下载模型。

    此函数供 HuggingFace 的 snapshot_download 等 API 使用。

    Returns:
        bool: True 表示仅使用本地缓存，False 表示可以联网下载
    """
    return is_offline_mode()


# ============================================================================
# 配置状态汇总（用于 UI 展示）
# ============================================================================

def get_config_status():
    """
    汇总所有配置项的设置状态，供 UI 展示配置完整性。

    返回值包含三类信息：
    1. 各配置项是否已设置、是否为必填、所属功能
    2. FFmpeg 可用性状态（路径、版本、是否可以自动下载）
    3. AI 模型下载状态（已下载数 / 总数 + 每个模型详情）

    Returns:
        dict: 配置状态字典
            - 键为配置名，值为 {"set": bool, "required": bool, "feature": str}
            - 特殊键 "FFMPEG" 包含 FFmpeg 检测结果
            - 特殊键 "MODELS" 包含模型下载状态
    """
    # ---- 遍历所有配置项，记录设置状态 ----
    status = {}
    for key in DEFAULT_CONFIG:
        value = get_config(key)
        status[key] = {
            "set": bool(value),                             # 是否已设置（非空）
            "required": key in REQUIRED_CONFIG,             # 是否为必填项
            "feature": REQUIRED_CONFIG.get(key, ""),        # 所属功能名称
        }

    # ---- FFmpeg 状态 ----
    ffmpeg_available, ffmpeg_msg = check_ffmpeg_available()
    status["FFMPEG"] = {
        "set": ffmpeg_available,               # FFmpeg 是否可用
        "required": True,                      # FFmpeg 始终必填
        "feature": "视频下载/分离/合成",         # 涉及功能
        "message": ffmpeg_msg,                 # 可用性消息
        "path": get_ffmpeg_path(),             # FFmpeg 路径
        "can_download": True,                  # 是否支持自动下载
    }

    # ---- AI 模型状态（动态导入避免循环依赖） ----
    try:
        from .model_manager import check_all_models_status
        model_statuses = check_all_models_status()
        downloaded = sum(1 for s in model_statuses.values() if s["downloaded"])  # 已下载数
        total = len(model_statuses)             # 模型总数
        status["MODELS"] = {
            "set": downloaded == total,         # 是否全部已下载
            "required": True,                   # 模型始终必填
            "feature": "AI模型",
            "message": f"已下载 {downloaded}/{total} 个模型",
            "downloaded": downloaded,            # 已下载数量
            "total": total,                      # 总数量
            "models": model_statuses,            # 每个模型的详情
        }
    except Exception as e:
        # 模型状态检查失败（如 model_manager 导入出错），记录错误但不中断
        status["MODELS"] = {
            "set": False,
            "required": True,
            "feature": "AI模型",
            "message": f"模型状态检查失败: {e}",
        }

    return status


# ============================================================================
# 模块初始化
# ============================================================================

def _init_logging_once():
    """
    一次性初始化日志系统。

    在模块被导入时自动调用，确保 logging 在程序早期就完成配置。
    使用 try-except 包裹，防止 log_config 导入或初始化失败导致程序崩溃。
    """
    try:
        from .log_config import init_logging
        init_logging()
    except Exception:
        # 日志初始化失败不应影响程序启动
        pass


# 模块导入时自动初始化日志系统
_init_logging_once()
