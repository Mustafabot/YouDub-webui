import os
import gc
import sys
import time
import subprocess
from pathlib import Path
from loguru import logger

from .config import get_config, PROJECT_ROOT
from .utils import install_package_with_mirrors


def _install_f5_package():
    """安装 f5-tts Python 库（使用多镜像回退）"""
    if install_package_with_mirrors("f5-tts", timeout=300):
        logger.info("f5-tts 库安装成功")
        return True
    raise RuntimeError(
        "f5-tts 库安装失败。请尝试手动安装:\n"
        "  1. 配置 pip 镜像: pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple\n"
        "  2. 手动安装: pip install f5-tts"
    )


def _check_f5_package_installed() -> bool:
    """检查 f5-tts 库是否已安装"""
    try:
        from f5_tts.api import F5TTS
        return True
    except ImportError:
        return False


_HF_HOME = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
HF_CACHE_DIR = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_HOME, "hub")))

TORCH_HUB_DIR = Path(os.environ.get(
    "TORCH_HOME",
    os.path.join(os.path.expanduser("~"), ".cache", "torch")
)) / "hub"

WHISPER_DOWNLOAD_ROOT = str(PROJECT_ROOT / "models" / "ASR" / "whisper")


def _apply_hf_endpoint():
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


def _check_hf_model_cached(org: str, name: str) -> bool:
    model_dir = HF_CACHE_DIR / f"models--{org}--{name}"
    if not model_dir.exists():
        return False
    refs_dir = model_dir / "refs"
    snapshots_dir = model_dir / "snapshots"
    if not refs_dir.exists() or not snapshots_dir.exists():
        return False
    refs = list(refs_dir.iterdir())
    if not refs:
        return False
    return True


def _check_whisper_model_cached() -> bool:
    model_dir = Path(WHISPER_DOWNLOAD_ROOT) / "models--Systran--faster-whisper-large-v3"
    if not model_dir.exists():
        return False
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return False
    return True


def _check_demucs_model_cached() -> bool:
    import yaml
    try:
        import demucs.pretrained
        remote_dir = demucs.pretrained.REMOTE_ROOT
        yaml_file = remote_dir / "htdemucs_ft.yaml"
        if not yaml_file.exists():
            return False
        bag = yaml.safe_load(open(yaml_file))
        sigs = bag.get("models", bag.get("model", None))
        if sigs is None:
            return False
        if isinstance(sigs, str):
            sigs = [sigs]
        checkpoints_dir = TORCH_HUB_DIR / "checkpoints"
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


def _check_f5_model_cached() -> bool:
    base_dir = HF_CACHE_DIR / "models--SWivid--F5-TTS"
    small_dir = HF_CACHE_DIR / "models--SWivid--F5-TTS-small"
    return base_dir.exists() or small_dir.exists()


def _download_demucs():
    import torch
    import yaml
    import time
    from urllib.parse import urlparse
    from demucs.pretrained import REMOTE_ROOT, ROOT_URL
    _apply_hf_endpoint()

    logger.info("Downloading Demucs (htdemucs_ft) model...")

    yaml_file = REMOTE_ROOT / "htdemucs_ft.yaml"
    bag = yaml.safe_load(open(yaml_file, encoding="utf-8"))
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

    checkpoints_dir = TORCH_HUB_DIR / "checkpoints"

    for sig in signatures:
        if sig not in url_map:
            logger.warning(f"找不到签名 {sig} 的 URL，跳过")
            continue
        url = url_map[sig]
        filename = os.path.basename(urlparse(url).path)
        cached_file = checkpoints_dir / filename

        for attempt in range(3):
            try:
                if cached_file.exists():
                    cached_file.unlink()
                    logger.info(f"已清除旧缓存: {filename}")
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


ALIGN_HF_MODELS = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
}


def _check_whisper_align_cached() -> bool:
    for lang, model_id in ALIGN_HF_MODELS.items():
        org, name = model_id.split("/", 1)
        if not _check_hf_model_cached(org, name):
            logger.debug(f"Align model for {lang} ({model_id}) not cached")
            return False
    return True


def _download_whisper():
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
    from huggingface_hub import snapshot_download
    _apply_hf_endpoint()
    logger.info("Downloading align models...")
    for lang, model_id in ALIGN_HF_MODELS.items():
        logger.info(f"Downloading align model for {lang} ({model_id})...")
        try:
            _snapshot_download_with_retry(
                model_id,
                max_retries=2,
                local_files_only=False,
                resume_download=True,
            )
            logger.info(f"Align model for {lang} downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download align model for {lang}: {e}")
    gc.collect()
    logger.info("Align models download completed")


def _check_pyannote_segmentation_cached() -> bool:
    return _check_hf_model_cached("pyannote", "speaker-diarization-community-1")


def _check_pyannote_embedding_cached() -> bool:
    return _check_hf_model_cached("pyannote", "embedding")


def _download_pyannote_segmentation():
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
        local_files_only=False,
        resume_download=True,
    )
    gc.collect()
    logger.info("pyannote/speaker-diarization-community-1 downloaded successfully")


def _download_pyannote_embedding():
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
        local_files_only=False,
        resume_download=True,
    )
    gc.collect()
    logger.info("pyannote/embedding model downloaded successfully")


def _download_f5():
    from huggingface_hub import snapshot_download
    _apply_hf_endpoint()
    
    if not _check_f5_package_installed():
        logger.info("f5-tts 库未安装，开始安装...")
        _install_f5_package()
    
    logger.info("Downloading F5-TTS models...")
    failures = []
    for model_id in ["SWivid/F5-TTS", "SWivid/F5-TTS-small"]:
        try:
            _snapshot_download_with_retry(
                model_id,
                max_retries=2,
                local_files_only=False,
                resume_download=True,
            )
        except Exception as e:
            logger.warning(f"下载 {model_id} 失败: {e}")
            failures.append(model_id)
    gc.collect()
    if len(failures) == 2:
        raise RuntimeError(
            f"F5-TTS 模型下载失败，请检查网络连接或镜像配置后再试。\n"
            f"下载失败的模型: {', '.join(failures)}"
        )
    if failures:
        logger.warning(f"部分模型下载失败: {failures}，但至少有一个模型可用")
    else:
        logger.info("F5-TTS models downloaded successfully")


MODEL_REGISTRY = {
    "demucs_htdemucs_ft": {
        "name": "Demucs (htdemucs_ft)",
        "description": "音频分离模型，用于将人声与伴奏分离",
        "module_id": "audio_separation",
        "size_gb": 0.8,
        "requires_hf_token": False,
        "check_fn": _check_demucs_model_cached,
        "download_fn": _download_demucs,
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
        "requires_hf_token": True,
        "check_fn": _check_pyannote_segmentation_cached,
        "download_fn": _download_pyannote_segmentation,
    },
    "pyannote_embedding": {
        "name": "pyannote/embedding",
        "description": "说话者嵌入模型，为 TTS 匹配音色",
        "module_id": "tts",
        "size_gb": 0.4,
        "requires_hf_token": True,
        "check_fn": _check_pyannote_embedding_cached,
        "download_fn": _download_pyannote_embedding,
    },
    "f5_tts": {
        "name": "F5-TTS",
        "description": "零样本声音克隆模型，中文质量优秀，支持低显存自动适配（将自动安装 f5-tts 库）",
        "module_id": "tts",
        "size_gb": 1.2,
        "requires_hf_token": False,
        "check_fn": _check_f5_model_cached,
        "download_fn": _download_f5,
        "extra_check_fn": _check_f5_package_installed,
    },
}


def get_all_models():
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_id):
    return MODEL_REGISTRY.get(model_id)


def get_models_for_module(module_id):
    return [
        mid for mid, info in MODEL_REGISTRY.items()
        if info["module_id"] == module_id
    ]


def check_model_status(model_id):
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
            if model_id == "f5_tts":
                extra_status["extra_label"] = "f5-tts库"
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
    results = {}
    for model_id in MODEL_REGISTRY:
        results[model_id] = check_model_status(model_id)
    return results


def download_model(model_id):
    info = MODEL_REGISTRY.get(model_id)
    if not info:
        raise ValueError(f"Unknown model: {model_id}")
    if info["requires_hf_token"] and not get_config("HF_TOKEN"):
        raise ValueError(f"下载模型 {info['name']} 需要先设置 HF_TOKEN")
    info["download_fn"]()


def download_all_models():
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
