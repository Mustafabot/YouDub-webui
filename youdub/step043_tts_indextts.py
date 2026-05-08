import os
import gc
import contextlib
import io
from loguru import logger
import torch
import time
from .config import get_config, PROJECT_ROOT

try:
    from indextts.infer_v2 import IndexTTS2
    INDEXTTS_AVAILABLE = True
except ImportError:
    INDEXTTS_AVAILABLE = False
    logger.warning("indextts library not available. Please install with: pip install indextts")

model = None
model_config = None


def _detect_tts_config(device='auto'):
    model_dir = get_config('INDEXTTS_MODEL_DIR', 'checkpoints')
    if not os.path.isabs(model_dir):
        model_dir = str(PROJECT_ROOT / model_dir)
    use_fp16 = get_config('INDEXTTS_USE_FP16', 'true').lower() in ('true', '1', 'yes')
    use_deepspeed = get_config('INDEXTTS_USE_DEEPSPEED', 'false').lower() in ('true', '1', 'yes')

    if device == 'auto':
        if not torch.cuda.is_available():
            logger.info('未检测到 GPU，IndexTTS 将使用 CPU 推理')
            return {'device': 'cpu', 'use_fp16': False, 'use_deepspeed': False, 'model_dir': model_dir}
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8:
            config = {'device': 'cuda', 'use_fp16': use_fp16, 'use_deepspeed': use_deepspeed, 'model_dir': model_dir}
        elif vram_gb >= 4:
            config = {'device': 'cuda', 'use_fp16': True, 'use_deepspeed': False, 'model_dir': model_dir}
        else:
            config = {'device': 'cuda', 'use_fp16': True, 'use_deepspeed': False, 'model_dir': model_dir}
        logger.info(f'GPU 显存: {vram_gb:.1f}GB，IndexTTS 配置: fp16={config["use_fp16"]}, deepspeed={config["use_deepspeed"]}')
        return config
    elif device == 'cpu':
        return {'device': 'cpu', 'use_fp16': False, 'use_deepspeed': False, 'model_dir': model_dir}
    else:
        return {'device': device, 'use_fp16': use_fp16, 'use_deepspeed': use_deepspeed, 'model_dir': model_dir}


def init_indextts():
    load_model()


def load_model(device='auto', use_fp16=None, use_deepspeed=None):
    global model, model_config
    if model is not None:
        return

    if not INDEXTTS_AVAILABLE:
        raise RuntimeError("IndexTTS 未安装，请执行 pip install indextts 进行安装")

    config = _detect_tts_config(device)
    if use_fp16 is not None:
        config['use_fp16'] = use_fp16
    if use_deepspeed is not None:
        config['use_deepspeed'] = use_deepspeed
    model_config = config

    model_dir = config['model_dir']
    cfg_path = os.path.join(model_dir, 'config.yaml')

    if not os.path.exists(cfg_path):
        raise RuntimeError(
            f'IndexTTS 模型配置文件不存在: {cfg_path}\n'
            f'请先在设置页「模型管理」中下载 IndexTTS-2 模型，或手动下载到 {model_dir} 目录。\n'
            f'下载命令: huggingface-cli download IndexTeam/IndexTTS-2 --local-dir={model_dir}'
        )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    logger.info(f'加载 IndexTTS v2 模型 (fp16={config["use_fp16"]}, deepspeed={config["use_deepspeed"]}) 从 {model_dir}')
    t_start = time.time()

    load_error = None
    try:
        model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=config['use_fp16'],
            use_cuda_kernel=False,
            use_deepspeed=config['use_deepspeed'],
        )
    except Exception as e:
        load_error = e
        logger.warning(f'加载 IndexTTS v2 模型失败: {e}')

    if model is None and load_error:
        raise RuntimeError(
            f'IndexTTS v2 模型加载失败。\n'
            f'错误详情: {load_error}'
        )

    t_end = time.time()
    logger.info(f'IndexTTS v2 模型加载完成，耗时 {t_end - t_start:.2f}s')
    _log_cuda_memory()


def _log_cuda_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(
            f'CUDA 显存: 已分配={allocated:.2f}GB, '
            f'已预留={reserved:.2f}GB, 总计={total:.2f}GB'
        )


def tts(text, output_path, speaker_wav, device='auto', ref_text=""):
    global model
    if not INDEXTTS_AVAILABLE:
        raise RuntimeError("IndexTTS 未安装，请执行 pip install indextts 进行安装")

    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f'参考音频不存在: {speaker_wav}，请确认说话人分离步骤已正确执行')

    if os.path.exists(output_path):
        logger.info(f'IndexTTS 音频已存在，跳过: {output_path}')
        return

    if model is None:
        load_model(device)

    if model is None:
        raise RuntimeError('IndexTTS v2 模型加载失败，无法进行语音合成')

    use_cuda = model_config and model_config.get('device') != 'cpu'

    last_error = None
    for retry in range(3):
        try:
            if use_cuda:
                torch.cuda.empty_cache()
            with contextlib.redirect_stdout(io.StringIO()):
                with torch.no_grad():
                    model.infer(
                        spk_audio_prompt=speaker_wav,
                        text=text,
                        output_path=output_path,
                        verbose=False,
                    )

            if not os.path.exists(output_path):
                raise RuntimeError(f'音频文件保存失败: {output_path}')

            if use_cuda:
                torch.cuda.empty_cache()
            break
        except Exception as e:
            last_error = e
            logger.warning(f'IndexTTS 生成失败 (尝试 {retry+1}/3): {e}')
            if use_cuda:
                torch.cuda.empty_cache()
                gc.collect()
    else:
        raise RuntimeError(
            f'IndexTTS 生成失败，已重试 3 次。最后一次错误: {last_error}\n'
            f'文本: {text[:50]}...' if len(text) > 50 else f'IndexTTS 生成失败，已重试 3 次。最后一次错误: {last_error}\n文本: {text}'
        )


def cleanup_indextts():
    global model, model_config
    if model is not None:
        del model
        model = None
    model_config = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info('IndexTTS v2 模型已清理，显存已释放')


def get_model_info():
    if model_config is None:
        return {'device': 'not_loaded', 'use_fp16': False, 'use_deepspeed': False, 'model_dir': 'unknown'}
    return dict(model_config)


if __name__ == '__main__':
    speaker_wav = r'videos\TED-Ed\20231121 Why did the US try to kill all the bison？ - Andrew C. Isenberg\audio_vocals.wav'
    while True:
        text = input('请输入：')
        tts(text, f'playground/{text}.wav', speaker_wav)
