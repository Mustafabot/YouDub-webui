import os
import gc
import io
import contextlib
from loguru import logger
import numpy as np
import torch
import time
from .utils import save_wav

try:
    from f5_tts.api import F5TTS
    F5_AVAILABLE = True
except ImportError:
    F5_AVAILABLE = False
    logger.warning("f5-tts library not available. Please install with: pip install f5-tts")

model = None
model_config = None


def _detect_tts_config(device='auto'):
    if device == 'auto':
        if not torch.cuda.is_available():
            logger.info('未检测到 GPU，F5-TTS 将使用 CPU 推理')
            return {'device': 'cpu', 'model_size': 'small', 'nfe_step': 16}
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8:
            config = {'device': 'cuda', 'model_size': 'base', 'nfe_step': 32}
        elif vram_gb >= 6:
            config = {'device': 'cuda', 'model_size': 'base', 'nfe_step': 20}
        elif vram_gb >= 4:
            config = {'device': 'cuda', 'model_size': 'small', 'nfe_step': 32}
        else:
            config = {'device': 'cuda', 'model_size': 'small', 'nfe_step': 16}
        logger.info(f'GPU 显存: {vram_gb:.1f}GB，F5-TTS 配置: {config}')
        return config
    elif device == 'cpu':
        return {'device': 'cpu', 'model_size': 'small', 'nfe_step': 16}
    else:
        return {'device': device, 'model_size': 'base', 'nfe_step': 32}


def init_F5TTS():
    load_model()


def load_model(device='auto', model_size='auto', nfe_step=None):
    global model, model_config
    if model is not None:
        return

    if not F5_AVAILABLE:
        raise RuntimeError("F5-TTS 未安装，请执行 pip install f5-tts 进行安装")

    config = _detect_tts_config(device)
    if model_size != 'auto':
        config['model_size'] = model_size
    if nfe_step is not None:
        config['nfe_step'] = nfe_step
    model_config = config

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    logger.info(f'加载 F5-TTS 模型 ({config["model_size"]}) 到 {config["device"]}')
    t_start = time.time()

    load_error = None
    try:
        if config['model_size'] == 'small':
            model = F5TTS(ckpt_file="hf://SWivid/F5-TTS/F5TTS_v1_Small", device=config['device'])
        else:
            model = F5TTS(device=config['device'])
    except Exception as e:
        load_error = e
        logger.warning(f'加载 F5-TTS {config["model_size"]} 模型失败: {e}')

    if model is None and load_error:
        raise RuntimeError(
            f'F5-TTS 模型加载失败，请检查网络连接或 Hugging Face 镜像配置。\n'
            f'错误详情: {load_error}'
        )

    if config['device'] == 'cuda':
        try:
            model.model = model.model.half()
            logger.info('F5-TTS 模型已转换为 FP16 精度')
        except Exception as e:
            logger.warning(f'FP16 转换失败，使用默认精度: {e}')

    t_end = time.time()
    logger.info(f'F5-TTS 模型加载完成，耗时 {t_end - t_start:.2f}s')
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
    if not F5_AVAILABLE:
        raise RuntimeError("F5-TTS 未安装，请执行 pip install f5-tts 进行安装")

    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f'参考音频不存在: {speaker_wav}，请确认说话人分离步骤已正确执行')

    if os.path.exists(output_path):
        logger.info(f'F5-TTS 音频已存在，跳过: {output_path}')
        return

    if model is None:
        load_model(device)

    if model is None:
        raise RuntimeError('F5-TTS 模型加载失败，无法进行语音合成')

    nfe_step = model_config.get('nfe_step', 32) if model_config is not None else 32
    use_cuda = model_config and model_config.get('device') == 'cuda'

    if not ref_text:
        logger.warning('参考文本为空，F5-TTS 将使用 ASR 自动转录，可能需要额外显存。建议提供 ref_text 参数以获得更好的稳定性。')

    last_error = None
    for retry in range(3):
        try:
            if use_cuda:
                torch.cuda.empty_cache()
            with contextlib.redirect_stdout(io.StringIO()):
                with torch.no_grad():
                    wav, sr, _ = model.infer(
                        ref_file=speaker_wav,
                        ref_text=ref_text,
                        gen_text=text,
                        nfe_step=nfe_step,
                    )

            wav = np.array(wav) if not isinstance(wav, np.ndarray) else wav
            save_wav(wav, output_path)

            if not os.path.exists(output_path):
                raise RuntimeError(f'音频文件保存失败: {output_path}')

            if use_cuda:
                torch.cuda.empty_cache()
            break
        except Exception as e:
            last_error = e
            logger.warning(f'F5-TTS 生成失败 (尝试 {retry+1}/3): {e}')
            if use_cuda:
                torch.cuda.empty_cache()
                gc.collect()
    else:
        raise RuntimeError(
            f'F5-TTS 生成失败，已重试 3 次。最后一次错误: {last_error}\n'
            f'文本: {text[:50]}...' if len(text) > 50 else text
        )


def cleanup_f5tts():
    global model, model_config
    if model is not None:
        del model
        model = None
    model_config = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info('F5-TTS 模型已清理，显存已释放')


def get_model_info():
    if model_config is None:
        return {'device': 'not_loaded', 'model_size': 'unknown', 'nfe_step': 0}
    return dict(model_config)


if __name__ == '__main__':
    speaker_wav = r'videos\TED-Ed\20231121 Why did the US try to kill all the bison？ - Andrew C. Isenberg\audio_vocals.wav'
    while True:
        text = input('请输入：')
        tts(text, f'playground/{text}.wav', speaker_wav)
