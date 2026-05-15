"""
model_manager.py — AI 模型管理模块

本模块负责 YouDub 流水线中所有 AI 模型的注册、状态检查和下载管理。
共管理 6 个模型：
    1. demucs_htdemucs_ft     — 音频分离（人声/伴奏）
    2. whisper_large_v3        — 语音识别
    3. whisper_align           — 语音对齐（多语言）
    4. pyannote_segmentation   — 说话者分离（分割）
    5. pyannote_embedding      — 说话者嵌入（为 TTS 匹配音色）
    6. indextts                — 零样本 TTS 声音克隆

核心机制：
    - MODEL_REGISTRY 字典统一注册所有模型
    - 每个模型绑定 check_fn（检查是否已下载）和 download_fn（下载函数）
    - 下载支持 HuggingFace / ModelScope 双源回退
    - 支持 HF_ENDPOINT 镜像加速
"""

import os
import gc
import sys
import time
import subprocess
from pathlib import Path
from loguru import logger

from .config import get_config, PROJECT_ROOT, MODEL_ROOT
from .utils import install_package_with_mirrors


# ──────────────────────────────────────────────
# IndexTTS 包管理
# ──────────────────────────────────────────────

def _install_indextts_package():
    """安装 indextts Python 库（使用多镜像回退）

    由于 IndexTTS-2 是一个外部 GitHub 仓库，需要通过 pip 安装。
    使用 utils.install_package_with_mirrors 自动尝试多个国内镜像源。

    Raises:
        RuntimeError: 所有镜像源均安装失败时抛出，附手动安装指引
    """
    if install_package_with_mirrors("git+https://github.com/index-tts/index-tts.git", timeout=600):
        logger.info("indextts 库安装成功")
        return True
    raise RuntimeError(
        "indextts 库安装失败。请尝试手动安装:\n"
        "  1. 配置 pip 镜像: pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple\n"
        "  2. 手动安装: pip install git+https://github.com/index-tts/index-tts.git"
    )


def _check_indextts_package_installed() -> bool:
    """检查 indextts 库是否已安装

    通过尝试导入 indextts.infer_v2.IndexTTS2 来判断。

    Returns:
        bool: 库已安装返回 True，否则返回 False
    """
    try:
        from indextts.infer_v2 import IndexTTS2
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────
# 模型文件路径常量
# ──────────────────────────────────────────────

# Whisper 模型下载根目录（在 MODEL_ROOT/whisper 下）
WHISPER_DOWNLOAD_ROOT = str(MODEL_ROOT / "whisper")

# Demucs 模型存放目录
DEMUCS_MODEL_DIR = MODEL_ROOT / "demucs"
# HuggingFace 默认缓存目录
HF_CACHE_DIR = str(MODEL_ROOT / "huggingface" / "hub")


# ──────────────────────────────────────────────
# 下载辅助函数
# ──────────────────────────────────────────────

def _apply_hf_endpoint():
    """应用 HuggingFace 镜像加速配置

    从配置或环境变量中读取 HF_ENDPOINT，如果存在则设置到环境变量中。
    这会使得 huggingface_hub 库的 download/snapshot 等操作走镜像站，
    从而加速国内用户的模型下载。

    同时还设置了 HF_HUB_DOWNLOAD_TIMEOUT=120 以避免超时。
    """
    endpoint = get_config("HF_ENDPOINT")
    if not endpoint:
        endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_HUB_URL"] = endpoint
        host = endpoint.replace("https://", "").replace("http://", "")
        os.environ["HF_MIRROR"] = host
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"


def _snapshot_download_with_retry(repo_id, max_retries=3, **kwargs):
    """带指数退避重试的 HuggingFace 模型下载

    封装 huggingface_hub.snapshot_download，在网络不稳定时自动重试。
    重试间隔：5s -> 10s -> 20s ... 最大 60s。

    Args:
        repo_id: HuggingFace 仓库 ID（如 "Systran/faster-whisper-large-v3"）
        max_retries: 最大重试次数（默认 3）
        **kwargs: 传递给 snapshot_download 的额外参数

    Returns:
        下载成功的路径

    Raises:
        RuntimeError: 所有重试均失败时抛出
    """
    from huggingface_hub import snapshot_download
    _apply_hf_endpoint()
    for attempt in range(1, max_retries + 1):
        try:
            return snapshot_download(repo_id, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                wait = min(2 ** attempt * 5, 60)
                logger.warning(f"下载 {repo_id} 失败 (尝试 {attempt}/{max_retries}): {e}")
                logger.info(f"等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"下载 {repo_id} 失败(已重试{max_retries}次): {e}")
    return None


# ──────────────────────────────────────────────
# 各模型下载状态检查函数
# ──────────────────────────────────────────────

def _check_whisper_model_cached() -> bool:
    """检查 Whisper large-v3 模型是否已下载到本地缓存

    检测 Systran/faster-whisper-large-v3 的 HuggingFace 快照目录是否存在。
    使用 HuggingFace Hub 的缓存目录结构：models--{org}--{name}/snapshots/

    Returns:
        bool: 模型已缓存返回 True
    """
    model_dir = Path(WHISPER_DOWNLOAD_ROOT) / "models--Systran--faster-whisper-large-v3"
    if not model_dir.exists():
        return False
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return False
    return True


def _check_demucs_model_cached() -> bool:
    """检查 Demucs (htdemucs_ft) 模型是否已下载

    通过读取 Demucs 的 yaml 配置文件，获取模型签名列表，
    然后逐一检查 checkpoints 目录下是否存在对应的 checkpoint 文件。

    Returns:
        bool: 所有模型文件已存在返回 True
    """
    import yaml
    try:
        import demucs.pretrained
        remote_dir = demucs.pretrained.REMOTE_ROOT
        yaml_file = remote_dir / "htdemucs_ft.yaml"
        if not yaml_file.exists():
            return False
        with open(yaml_file) as f:
            bag = yaml.safe_load(f)
        sigs = bag.get("models", bag.get("model", None))
        if sigs is None:
            return False
        if isinstance(sigs, str):
            sigs = [sigs]
        checkpoints_dir = DEMUCS_MODEL_DIR / "checkpoints"
        if not checkpoints_dir.exists():
            return False
        for sig in sigs:
            found = False
            for f in checkpoints_dir.iterdir():
                if f.name.startswith(sig):
                    found = True
                    break
            if not found:
                return False
        return True
    except Exception as e:
        logger.debug(f"Error checking demucs model: {e}")
        return False


def _check_indextts_model_cached() -> bool:
    """检查 IndexTTS-2 模型文件是否已下载

    通过检查配置的 INDEXTTS_MODEL_DIR 下是否存在 config.yaml 来判断。
    如果路径是相对路径，则相对于项目根目录 PROJECT_ROOT 解析。

    Returns:
        bool: 模型文件已存在返回 True
    """
    model_dir = get_config('INDEXTTS_MODEL_DIR', 'models/index-tts')
    if not os.path.isabs(model_dir):
        model_dir = str(PROJECT_ROOT / model_dir)
    cfg_path = os.path.join(model_dir, 'config.yaml')
    return os.path.exists(cfg_path)


# ──────────────────────────────────────────────
# 各模型下载函数
# ──────────────────────────────────────────────

def _download_indextts():
    """下载 IndexTTS-2 模型

    下载流程：
        1. 先检查并安装 indextts Python 库（如果未安装）
        2. 从 HuggingFace 下载 IndexTeam/IndexTTS-2
        3. HF 失败时回退到 ModelScope 下载
        4. 执行垃圾回收

    模型文件保存在 INDEXTTS_MODEL_DIR（默认 models/index-tts）。

    Raises:
        RuntimeError: HF 和 ModelScope 均下载失败时抛出
    """
    _apply_hf_endpoint()

    if not _check_indextts_package_installed():
        logger.info("indextts 库未安装，开始安装...")
        _install_indextts_package()

    model_dir = get_config('INDEXTTS_MODEL_DIR', 'models/index-tts')
    if not os.path.isabs(model_dir):
        model_dir = str(PROJECT_ROOT / model_dir)

    logger.info("Downloading IndexTTS-2 model...")

    try:
        _snapshot_download_with_retry(
            "IndexTeam/IndexTTS-2",
            max_retries=3,
            local_dir=model_dir,
            local_files_only=False,
            resume_download=True,
        )
    except Exception as hf_error:
        logger.warning(f"从 HuggingFace 下载 IndexTTS-2 失败: {hf_error}，尝试 ModelScope...")
        try:
            from modelscope import snapshot_download as ms_snapshot_download
            ms_snapshot_download(
                model_id="IndexTeam/IndexTTS-2",
                local_dir=model_dir,
            )
        except Exception as ms_error:
            raise RuntimeError(
                f"下载 IndexTTS-2 模型失败。\n"
                f"HuggingFace 错误: {hf_error}\n"
                f"ModelScope 错误: {ms_error}\n"
                f"请手动下载: huggingface-cli download IndexTeam/IndexTTS-2 --local-dir={model_dir}"
            )

    gc.collect()
    logger.info("IndexTTS-2 model downloaded successfully")


def _download_demucs():
    """下载 Demucs (htdemucs_ft) 音频分离模型

    下载流程：
        1. 解析 htdemucs_ft.yaml 获取模型签名列表
        2. 从 files.txt 构建签名到 URL 的映射
        3. 使用 torch.hub.load_state_dict_from_url 逐个下载 checkpoint
        4. 支持断点重试（3 次）
        5. 下载完成后清理缓存和 GPU 显存

    Raises:
        RuntimeError: yaml 中没有 models 列表，或某个签名重试 3 次后仍失败
    """
    import torch
    import yaml
    import time
    from urllib.parse import urlparse
    from demucs.pretrained import REMOTE_ROOT, ROOT_URL
    _apply_hf_endpoint()

    logger.info("Downloading Demucs (htdemucs_ft) model...")

    yaml_file = REMOTE_ROOT / "htdemucs_ft.yaml"
    with open(yaml_file, encoding="utf-8") as f:
        bag = yaml.safe_load(f)
    signatures = bag.get("models", [])
    if not signatures:
        raise RuntimeError("htdemucs_ft.yaml 中没有找到 models 列表")

    root_url_prefix = ""
    url_map = {}
    for line in (REMOTE_ROOT / "files.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("root:"):
            root_url_prefix = line.split(":", 1)[1].strip()
        else:
            sig = line.split("-", 1)[0]
            url_map[sig] = ROOT_URL + root_url_prefix + line

    checkpoints_dir = DEMUCS_MODEL_DIR / "checkpoints"

    for sig in signatures:
        if sig not in url_map:
            logger.warning(f"找不到签名 {sig} 的 URL，跳过")
            continue
        url = url_map[sig]
        filename = os.path.basename(urlparse(url).path)
        cached_file = checkpoints_dir / filename

        for attempt in range(3):
            try:
                logger.info(f"下载中 ({attempt+1}/3): {sig}")
                pkg = torch.hub.load_state_dict_from_url(
                    url, map_location="cpu", check_hash=False
                )
                del pkg
                break
            except Exception as e:
                logger.warning(f"下载 {sig} 失败 (尝试 {attempt+1}/3): {e}")
                if cached_file.exists():
                    try:
                        cached_file.unlink()
                    except Exception:
                        pass
                if attempt == 2:
                    raise RuntimeError(f"下载 {sig} 失败: {e}")
                wait = min(2 ** attempt * 3, 30)
                time.sleep(wait)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Demucs model downloaded successfully")


# ──────────────────────────────────────────────
# 语音对齐模型（多语言）配置
# ──────────────────────────────────────────────

# 各语言对应的 HuggingFace 对齐模型 ID
ALIGN_HF_MODELS = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",       # 日语
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", # 中文
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",         # 荷兰语
    "ko": "kresnik/wav2vec2-large-xlsr-korean",                   # 韩语
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",        # 俄语
}


def _check_whisper_align_cached() -> bool:
    """检查所有语言的 WhisperX 对齐模型是否已下载

    遍历 ALIGN_HF_MODELS 字典，检查每个模型在 HuggingFace 缓存目录中
    是否存在且包含 refs 和 snapshots 快照。

    Returns:
        bool: 所有语言的对齐模型都已缓存返回 True
    """
    for lang, model_id in ALIGN_HF_MODELS.items():
        org, name = model_id.split("/", 1)
        model_dir = Path(HF_CACHE_DIR) / f"models--{org}--{name}"
        if not model_dir.exists():
            return False
        refs_dir = model_dir / "refs"
        snapshots_dir = model_dir / "snapshots"
        if not refs_dir.exists() or not snapshots_dir.exists():
            return False
        if not list(refs_dir.iterdir()):
            return False
    return True


def _download_whisper():
    """下载 Whisper large-v3 语音识别模型

    从 HuggingFace 下载 Systran/faster-whisper-large-v3，
    模型缓存到 WHISPER_DOWNLOAD_ROOT 目录。
    支持断点续传和最多 3 次重试。
    """
    from huggingface_hub import snapshot_download
    _apply_hf_endpoint()
    logger.info("Downloading Whisper large-v3 model...")
    _snapshot_download_with_retry(
        "Systran/faster-whisper-large-v3",
        max_retries=3,
        cache_dir=WHISPER_DOWNLOAD_ROOT,
        local_files_only=False,
        resume_download=True,
    )
    logger.info("Whisper model downloaded successfully")


def _download_align():
    """下载所有语言的 WhisperX 对齐模型

    遍历 ALIGN_HF_MODELS，逐个下载每个语言的对齐模型到 HF_CACHE_DIR。
    即使某个语言下载失败，仍然继续尝试下载其他语言。
    下载完成后执行垃圾回收。
    """
    from huggingface_hub import snapshot_download
    _apply_hf_endpoint()
    logger.info("Downloading align models...")
    for lang, model_id in ALIGN_HF_MODELS.items():
        logger.info(f"Downloading align model for {lang} ({model_id})...")
        try:
            _snapshot_download_with_retry(
                model_id,
                max_retries=2,
                cache_dir=HF_CACHE_DIR,
                local_files_only=False,
                resume_download=True,
            )
            logger.info(f"Align model for {lang} downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download align model for {lang}: {e}")
    gc.collect()
    logger.info("Align models download completed")


def _check_pyannote_segmentation_cached() -> bool:
    """检查 pyannote 说话者分割模型是否已缓存

    检查 pyannote/speaker-diarization-community-1 的 HuggingFace 缓存目录。

    Returns:
        bool: 模型已缓存返回 True
    """
    model_dir = Path(HF_CACHE_DIR) / "models--pyannote--speaker-diarization-community-1"
    if not model_dir.exists():
        return False
    refs_dir = model_dir / "refs"
    snapshots_dir = model_dir / "snapshots"
    if not refs_dir.exists() or not snapshots_dir.exists():
        return False
    return bool(list(refs_dir.iterdir()))


def _check_pyannote_embedding_cached() -> bool:
    """检查 pyannote 说话者嵌入模型是否已缓存

    检查 pyannote/embedding 的 HuggingFace 缓存目录。

    Returns:
        bool: 模型已缓存返回 True
    """
    model_dir = Path(HF_CACHE_DIR) / "models--pyannote--embedding"
    if not model_dir.exists():
        return False
    refs_dir = model_dir / "refs"
    snapshots_dir = model_dir / "snapshots"
    if not refs_dir.exists() or not snapshots_dir.exists():
        return False
    return bool(list(refs_dir.iterdir()))


def _download_pyannote_segmentation():
    """下载 pyannote 说话者分割模型

    从 HuggingFace 下载 pyannote/speaker-diarization-community-1。
    注意：此模型需要 HF_TOKEN 认证才能下载（受限制模型）。
    支持最多 3 次重试。

    Raises:
        ValueError: 未设置 HF_TOKEN 时抛出
    """
    from huggingface_hub import snapshot_download
    _apply_hf_endpoint()
    hf_token = get_config("HF_TOKEN")
    if not hf_token:
        raise ValueError("需要设置 HF_TOKEN 才能下载说话者分离模型")
    logger.info("Downloading pyannote/speaker-diarization-community-1 model...")
    _snapshot_download_with_retry(
        "pyannote/speaker-diarization-community-1",
        max_retries=3,
        token=hf_token,
        cache_dir=HF_CACHE_DIR,
        local_files_only=False,
        resume_download=True,
    )
    gc.collect()
    logger.info("pyannote/speaker-diarization-community-1 downloaded successfully")


def _download_pyannote_embedding():
    """下载 pyannote 说话者嵌入模型

    从 HuggingFace 下载 pyannote/embedding。
    用于 TTS 模块的说话者音色匹配。
    需要 HF_TOKEN 认证。

    Raises:
        ValueError: 未设置 HF_TOKEN 时抛出
    """
    from huggingface_hub import snapshot_download
    _apply_hf_endpoint()
    hf_token = get_config("HF_TOKEN")
    if not hf_token:
        raise ValueError("需要设置 HF_TOKEN 才能下载说话者嵌入模型")
    logger.info("Downloading pyannote/embedding model...")
    _snapshot_download_with_retry(
        "pyannote/embedding",
        max_retries=3,
        token=hf_token,
        cache_dir=HF_CACHE_DIR,
        local_files_only=False,
        resume_download=True,
    )
    gc.collect()
    logger.info("pyannote/embedding model downloaded successfully")


# ──────────────────────────────────────────────
# 模型注册中心
# ──────────────────────────────────────────────

MODEL_REGISTRY = {
    "demucs_htdemucs_ft": {
        "name": "Demucs (htdemucs_ft)",
        "description": "音频分离模型，用于将人声与伴奏分离",
        "module_id": "audio_separation",          # 关联的处理步骤 ID
        "size_gb": 0.8,                            # 模型大小（GB）
        "requires_hf_token": False,                # 不需要 HF 认证
        "check_fn": _check_demucs_model_cached,    # 检查是否已下载
        "download_fn": _download_demucs,           # 下载函数
    },
    "whisper_large_v3": {
        "name": "Whisper large-v3",
        "description": "语音识别模型，将语音转换为文字",
        "module_id": "speech_recognition",
        "size_gb": 3.0,
        "requires_hf_token": False,
        "check_fn": _check_whisper_model_cached,
        "download_fn": _download_whisper,
    },
    "whisper_align": {
        "name": "WhisperX 对齐模型",
        "description": "语音对齐模型，用于精确时间戳对齐（含多语言）",
        "module_id": "speech_recognition",
        "size_gb": 0.5,
        "requires_hf_token": False,
        "check_fn": _check_whisper_align_cached,
        "download_fn": _download_align,
    },
    "pyannote_segmentation": {
        "name": "pyannote/speaker-diarization-community-1",
        "description": "说话者分离模型（含分割/嵌入/校准），区分不同说话人",
        "module_id": "speech_recognition",
        "size_gb": 0.3,
        "requires_hf_token": True,              # 需要 HF_TOKEN（受限模型）
        "check_fn": _check_pyannote_segmentation_cached,
        "download_fn": _download_pyannote_segmentation,
    },
    "pyannote_embedding": {
        "name": "pyannote/embedding",
        "description": "说话者嵌入模型，为 TTS 匹配音色",
        "module_id": "tts",
        "size_gb": 0.4,
        "requires_hf_token": True,              # 需要 HF_TOKEN（受限模型）
        "check_fn": _check_pyannote_embedding_cached,
        "download_fn": _download_pyannote_embedding,
    },
    "indextts": {
        "name": "IndexTTS-2",
        "description": "工业级零样本 TTS 模型，支持声音克隆（将自动安装 indextts 库）",
        "module_id": "tts",
        "size_gb": 3.5,
        "requires_hf_token": False,
        "check_fn": _check_indextts_model_cached,
        "download_fn": _download_indextts,
        "extra_check_fn": _check_indextts_package_installed,  # 额外检查：Python 库是否安装
    },
}


# ──────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────

def get_all_models():
    """获取所有已注册的模型 ID 列表"""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_id):
    """获取指定模型的注册信息

    Args:
        model_id: 模型 ID（如 "whisper_large_v3"）

    Returns:
        dict 或 None: 模型的注册信息字典，不存在返回 None
    """
    return MODEL_REGISTRY.get(model_id)


def get_models_for_module(module_id):
    """获取属于某处理步骤的所有模型

    根据 MODEL_REGISTRY 中每个模型的 module_id 字段进行过滤。

    Args:
        module_id: 处理步骤 ID（如 "speech_recognition"）

    Returns:
        List[str]: 模型 ID 列表
    """
    return [
        mid for mid, info in MODEL_REGISTRY.items()
        if info["module_id"] == module_id
    ]


def check_model_status(model_id):
    """检查指定模型的下载状态

    调用模型注册信息中的 check_fn 进行检测。
    对于有 extra_check_fn 的模型（如 indextts 需要检查 Python 库），
    也会一并检测并返回额外状态。

    Args:
        model_id: 模型 ID

    Returns:
        dict: 包含模型名称、描述、大小、是否已下载等状态的字典
    """
    info = MODEL_REGISTRY.get(model_id)
    if not info:
        return {"id": model_id, "downloaded": False, "error": "Unknown model"}
    try:
        downloaded = info["check_fn"]()
    except Exception as e:
        logger.debug(f"Error checking model {model_id}: {e}")
        downloaded = False

    extra_status = {}
    if "extra_check_fn" in info:
        try:
            extra_status["extra_ok"] = info["extra_check_fn"]()
            if model_id == "indextts":
                extra_status["extra_label"] = "indextts库"
        except Exception as e:
            logger.debug(f"Error in extra check for {model_id}: {e}")
            extra_status["extra_ok"] = False

    return {
        "id": model_id,
        "name": info["name"],
        "description": info["description"],
        "module_id": info["module_id"],
        "size_gb": info["size_gb"],
        "requires_hf_token": info["requires_hf_token"],
        "downloaded": downloaded,
        **extra_status,
    }


def check_all_models_status():
    """检查所有已注册模型的下载状态

    Returns:
        dict: 模型 ID 到状态的映射字典
    """
    results = {}
    for model_id in MODEL_REGISTRY:
        results[model_id] = check_model_status(model_id)
    return results


def download_model(model_id):
    """下载指定的模型

    根据 MODEL_REGISTRY 中的 download_fn 执行下载。
    如果模型要求 HF_TOKEN 但未配置，则抛出 ValueError。

    Args:
        model_id: 模型 ID

    Raises:
        ValueError: 未知模型或缺少 HF_TOKEN 时抛出
    """
    info = MODEL_REGISTRY.get(model_id)
    if not info:
        raise ValueError(f"Unknown model: {model_id}")
    if info["requires_hf_token"] and not get_config("HF_TOKEN"):
        raise ValueError(f"下载模型 {info['name']} 需要先设置 HF_TOKEN")
    info["download_fn"]()


def download_all_models():
    """下载所有尚未下载的模型

    遍历 MODEL_REGISTRY，自动跳过已下载的模型。
    每个模型的下载结果独立记录，一个模型下载失败不影响其他模型。

    Returns:
        dict: 模型 ID 到状态的映射（"success" / "failed: ..." / "already_downloaded"）
    """
    results = {}
    for model_id in MODEL_REGISTRY:
        status = check_model_status(model_id)
        if not status["downloaded"]:
            try:
                download_model(model_id)
                results[model_id] = "success"
            except Exception as e:
                results[model_id] = f"failed: {str(e)}"
        else:
            results[model_id] = "already_downloaded"
    return results


def format_model_status():
    """将所有模型的状态格式化为可读的中文字符串

    用于在 UI 中展示模型管理状态面板。

    Returns:
        str: 格式化后的模型状态信息（含图标标记和下载进度）
    """
    statuses = check_all_models_status()
    lines = ["模型状态：", ""]
    for model_id, status in statuses.items():
        mark = "✅" if status["downloaded"] else "❌"
        hf_note = " (需HF_TOKEN)" if status.get("requires_hf_token") else ""
        extra_mark = ""
        if "extra_ok" in status and "extra_label" in status:
            extra_mark = f" [{'✅' if status['extra_ok'] else '❌'} {status['extra_label']}]"
        lines.append(
            f"{mark} {status['name']} ({status['size_gb']}GB{hf_note}){extra_mark} "
            f"— {status['description']}"
        )
    downloaded = sum(1 for s in statuses.values() if s["downloaded"])
    total = len(statuses)
    lines.append(f"\n已下载 {downloaded}/{total} 个模型")
    return "\n".join(lines)
