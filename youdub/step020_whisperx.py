import json
import os
import time
import threading
from contextlib import contextmanager
import librosa
import numpy as np
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from loguru import logger

from .utils import save_wav
from .config import get_config, ensure_ffmpeg_available, get_ffmpeg_path, get_hf_local_files_only, PROJECT_ROOT, MODEL_ROOT


def _log_cuda_memory(logger_func=logger.info):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated
    logger_func(f'CUDA 显存: 已分配={allocated:.2f}GB, 已预留={reserved:.2f}GB, 空闲={free:.2f}GB, 总计={total:.2f}GB')


_MODEL_VRAM_ESTIMATES = {
    'large-v3': 6.0, 'large-v2': 6.0, 'large': 6.0,
    'medium': 4.0, 'small': 2.0, 'base': 1.0, 'tiny': 0.5,
    'align': 1.5, 'diarize': 1.5,
}


def _get_free_vram_gb():
    """获取当前空闲 GPU 显存（GB），无 GPU 返回无穷大"""
    if not torch.cuda.is_available():
        return float('inf')
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total - allocated


def _get_default_batch_size():
    """根据 GPU 显存自动选择保守的批处理大小，预留安全余量避免 OOM"""
    if not torch.cuda.is_available():
        return 1
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # 保守估算：实际峰值内存 = 模型 + 对齐 + 批处理数据，留 30% 余量
    if total_vram_gb >= 24:
        return 16
    elif total_vram_gb >= 16:
        return 8
    elif total_vram_gb >= 8:
        return 4
    else:
        return 1


def _check_vram_before_load(model_name, device='cuda'):
    """加载模型前主动检查显存，不足时返回 (False, 建议消息) 避免进程被系统 OOM-kill"""
    if device == 'cpu' or not torch.cuda.is_available():
        return True, None
    estimated = _MODEL_VRAM_ESTIMATES.get(model_name, 4.0)
    free = _get_free_vram_gb()
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if free < estimated * 0.7:
        return False, (
            f"GPU 显存不足以加载 {model_name} 模型（约需 {estimated:.1f}GB，"
            f"当前空闲 {free:.1f}GB / 总计 {total:.1f}GB）。"
            f"建议：选择更小的模型（如 medium/small/tiny）或切换为 CPU 模式。"
        )
    return True, None


def _is_oom_error(e):
    """判断异常是否由 CUDA/系统 OOM 导致"""
    msg = str(e).lower()
    return 'out of memory' in msg or ('cuda' in msg and ('alloc' in msg or 'oom' in msg))


@contextmanager
def _ffmpeg_in_path():
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        yield
        return
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    current_path = os.environ.get('PATH', '')
    if ffmpeg_dir in current_path:
        yield
        return
    original_path = current_path
    os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
    try:
        yield
    finally:
        os.environ['PATH'] = original_path

whisper_model = None
diarize_model = None

align_model = None
language_code = None
align_metadata = None

_model_lock = threading.Lock()

def init_whisperx():
    pass
    
def load_whisper_model(model_name: str = 'large-v3', download_root = None, device='auto'):
    if model_name == 'large':
        model_name = 'large-v3'
    if download_root is None:
        download_root = str(MODEL_ROOT / 'whisper')
    global whisper_model
    if whisper_model is not None:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_files_only = get_hf_local_files_only()
    if local_files_only:
        logger.info(f'离线模式：从本地缓存加载 WhisperX 模型: {model_name}')
    else:
        logger.info(f'Loading WhisperX model: {model_name}')
    _log_cuda_memory()
    t_start = time.time()
    try:
        whisper_model = whisperx.load_model(
            model_name, 
            download_root=download_root, 
            device=device,
            local_files_only=local_files_only
        )
    except Exception:
        cleanup_whisperx()
        raise
    t_end = time.time()
    logger.info(f'Loaded WhisperX model: {model_name} in {t_end - t_start:.2f}s')
    _log_cuda_memory()

def load_align_model(language='en', device='auto'):
    global align_model, language_code, align_metadata
    if align_model is not None and language_code == language:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    language_code = language
    local_files_only = get_hf_local_files_only()
    if local_files_only:
        logger.info(f'离线模式：从本地缓存加载对齐模型: {language_code}')
    t_start = time.time()
    try:
        _prev_hub_offline = os.environ.get('HF_HUB_OFFLINE', None)
        if local_files_only:
            os.environ['HF_HUB_OFFLINE'] = '1'
        _prev_hf_home = os.environ.get('HF_HOME', None)
        os.environ['HF_HOME'] = str(MODEL_ROOT / "huggingface")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device
        )
    except Exception:
        cleanup_whisperx()
        raise
    finally:
        if _prev_hub_offline is not None:
            os.environ['HF_HUB_OFFLINE'] = _prev_hub_offline
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)
        if _prev_hf_home is not None:
            os.environ['HF_HOME'] = _prev_hf_home
        else:
            os.environ.pop('HF_HOME', None)
    t_end = time.time()
    logger.info(f'Loaded alignment model: {language_code} in {t_end - t_start:.2f}s')
    _log_cuda_memory()
    
def load_diarize_model(device='auto'):
    global diarize_model
    if diarize_model is not None:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_files_only = get_hf_local_files_only()
    if local_files_only:
        logger.info('离线模式：从本地缓存加载说话者分离模型')
    t_start = time.time()
    try:
        _prev_hub_offline = os.environ.get('HF_HUB_OFFLINE', None)
        if local_files_only:
            os.environ['HF_HUB_OFFLINE'] = '1'
        _prev_hf_home = os.environ.get('HF_HOME', None)
        os.environ['HF_HOME'] = str(MODEL_ROOT / "huggingface")
        diarize_model = DiarizationPipeline(
            token=get_config('HF_TOKEN'),
            device=device
        )
    except Exception:
        cleanup_whisperx()
        raise
    finally:
        if _prev_hub_offline is not None:
            os.environ['HF_HUB_OFFLINE'] = _prev_hub_offline
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)
        if _prev_hf_home is not None:
            os.environ['HF_HOME'] = _prev_hf_home
        else:
            os.environ.pop('HF_HOME', None)
    t_end = time.time()
    logger.info(f'Loaded diarization model in {t_end - t_start:.2f}s')
    _log_cuda_memory()


def merge_segments(transcript, ending='!"\').:;?]}~'):
    merged_transcription = []
    buffer_segment = None

    for segment in transcript:
        if buffer_segment is None:
            buffer_segment = segment
        else:
            if buffer_segment['text'][-1] in ending:
                merged_transcription.append(buffer_segment)
                buffer_segment = segment
            else:
                buffer_segment['text'] += ' ' + segment['text']
                buffer_segment['end'] = segment['end']

    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription

def transcribe_audio(folder, model_name: str = 'large', download_root='models/whisper', device='auto', batch_size=None, diarization=True, min_speakers=None, max_speakers=None):
    if batch_size is None:
        batch_size = _get_default_batch_size()
    if os.path.exists(os.path.join(folder, 'transcript.json')):
        logger.info(f'Transcript already exists in {folder}')
        return True

    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f'音频文件不存在: {wav_path}，请确认音频分离步骤已正确执行')

    ffmpeg_available, ffmpeg_msg = ensure_ffmpeg_available(auto_download=True)
    if not ffmpeg_available:
        raise RuntimeError(
            f'FFmpeg 不可用，无法进行语音识别。{ffmpeg_msg}\n'
            f'请按以下方式之一安装 FFmpeg：\n'
            f'1. 运行 python scripts/download_ffmpeg.py 自动下载\n'
            f'2. 从 https://ffmpeg.org/download.html 下载，解压后将 bin 目录添加到系统 PATH\n'
            f'3. 在配置中设置 FFMPEG_PATH'
        )

    logger.info(f'Transcribing {wav_path}')
    _log_cuda_memory()
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载前主动检查显存，避免进程被系统 OOM-kill
    ok, err_msg = _check_vram_before_load(model_name, device)
    if not ok:
        logger.warning(err_msg)

    try:
        load_whisper_model(model_name, download_root, device)

        with _ffmpeg_in_path():
            rec_result = whisper_model.transcribe(wav_path, batch_size=batch_size)

        # 转录完成后立即卸载 Whisper 模型（~6GB），为对齐步骤释放显存
        try:
            _unload_whisper_model()
        except Exception as unload_err:
            logger.warning(f"卸载 Whisper 模型时出错（非致命）: {unload_err}")

        if rec_result['language'] == 'nn':
            logger.warning(f'No language detected in {wav_path}')
            return False

        # 加载对齐模型前检查显存
        ok, err_msg = _check_vram_before_load('align', device)
        if not ok:
            logger.warning(err_msg)

        load_align_model(rec_result['language'])
        with _ffmpeg_in_path():
            rec_result = whisperx.align(rec_result['segments'], align_model, align_metadata,
                                        wav_path, device, return_char_alignments=False)

        # 对齐完成后卸载对齐模型（~1.5GB），为说话者分离释放显存
        try:
            _unload_align_model()
        except Exception as unload_err:
            logger.warning(f"卸载对齐模型时出错（非致命）: {unload_err}")

        if diarization:
            ok, err_msg = _check_vram_before_load('diarize', device)
            if not ok:
                logger.warning(err_msg)

            load_diarize_model(device)
            with _ffmpeg_in_path():
                diarize_segments = diarize_model(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
                rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)

        transcript = [{'start': segement['start'], 'end': segement['end'], 'text': segement['text'].strip(), 'speaker': segement.get('speaker', 'SPEAKER_00')} for segement in rec_result['segments']]
        transcript = merge_segments(transcript)
        with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=4, ensure_ascii=False)
        logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
        generate_speaker_audio(folder, transcript)
        return True
    except Exception as e:
        # 如果因 CUDA OOM 失败且 batch_size > 1，自动减半后重试一次
        if _is_oom_error(e) and batch_size > 1:
            new_batch_size = max(1, batch_size // 2)
            logger.warning(
                f"CUDA 显存不足（batch_size={batch_size}），"
                f"自动降低至 batch_size={new_batch_size} 后重试..."
            )
            cleanup_whisperx()
            return transcribe_audio(
                folder, model_name, download_root, device,
                new_batch_size, diarization, min_speakers, max_speakers
            )
        cleanup_whisperx()
        raise

def generate_speaker_audio(folder, transcript):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_segments = {}
    length = len(audio_data)
    delay = 0.05
    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end'] + delay) * samplerate), length)
        speaker = segment['speaker']
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(audio_data[start:end])

    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)
    
    for speaker, segments in speaker_segments.items():
        speaker_file_path = os.path.join(speaker_folder, f"{speaker}.wav")
        if segments:
            save_wav(np.concatenate(segments), speaker_file_path)
        else:
            logger.warning(f"说话人 {speaker} 无有效音频段，跳过保存")
            

def transcribe_all_audio_under_folder(folder, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=None, diarization=True, min_speakers=None, max_speakers=None):
    if batch_size is None:
        batch_size = _get_default_batch_size()
    if not os.path.isabs(folder):
        folder = str(PROJECT_ROOT / folder)
    if not os.path.isabs(download_root):
        download_root = str(PROJECT_ROOT / download_root)
    found_video_dir = False
    success_list = []
    fail_list = []
    try:
        for root, dirs, files in os.walk(folder):
            if 'download.mp4' not in files and 'audio.wav' not in files and 'audio_vocals.wav' not in files and 'transcript.json' not in files:
                continue
            found_video_dir = True
            if 'audio_vocals.wav' not in files:
                fail_list.append(f"{root}: 缺少 audio_vocals.wav")
                logger.error(f"目录 {root} 转录失败: 缺少 audio_vocals.wav")
                continue
            if 'transcript.json' in files:
                success_list.append(root)
                continue
            try:
                transcribe_audio(root, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers)
                success_list.append(root)
            except Exception as e:
                logger.error(f"目录 {root} 转录失败: {e}")
                fail_list.append(f"{root}: {e}")
        if not found_video_dir:
            raise FileNotFoundError(f'在 {folder} 下未找到任何视频处理目录')
        summary = f'语音识别完成: 成功 {len(success_list)}, 失败 {len(fail_list)}'
        logger.info(summary)
        if fail_list:
            logger.warning(f'失败详情: {fail_list}')
        return summary
    finally:
        cleanup_whisperx()

def _unload_whisper_model():
    """转录完成后卸载 Whisper 模型，为对齐和说话者分离释放显存"""
    global whisper_model
    import gc
    import torch

    with _model_lock:
        if whisper_model is not None:
            del whisper_model
            whisper_model = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    logger.info('Whisper 模型已卸载，显存已释放给后续步骤')
    _log_cuda_memory()


def _unload_align_model():
    """对齐完成后卸载对齐模型，为说话者分离释放显存"""
    global align_model, language_code, align_metadata
    import gc
    import torch

    with _model_lock:
        if align_model is not None:
            del align_model
            align_model = None
            language_code = None
            align_metadata = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    logger.info('对齐模型已卸载，显存已释放给后续步骤')
    _log_cuda_memory()


def cleanup_whisperx():
    """清理 WhisperX 相关模型，释放显存"""
    global whisper_model, align_model, language_code, align_metadata, diarize_model
    import gc
    import torch

    with _model_lock:
        if whisper_model is not None:
            del whisper_model
            whisper_model = None

        if align_model is not None:
            del align_model
            align_model = None
            language_code = None
            align_metadata = None

        if diarize_model is not None:
            del diarize_model
            diarize_model = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info('WhisperX 模型已清理，显存已释放')
    _log_cuda_memory()


def transcribe_audio_in_folders(folder_list, model_name='large', download_root='models/ASR/whisper', device='auto', batch_size=None, diarization=True, min_speakers=None, max_speakers=None):
    """处理指定目录列表中的语音识别

    Args:
        folder_list: 需要处理的目录路径列表
        model_name: Whisper 模型名称
        download_root: 模型下载目录
        device: 计算设备
        batch_size: 批处理大小，None 时根据显存自动选择
        diarization: 是否启用说话者分离
        min_speakers: 最小说话人数
        max_speakers: 最大说话人数
    """
    if batch_size is None:
        batch_size = _get_default_batch_size()
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    if not os.path.isabs(download_root):
        download_root = str(PROJECT_ROOT / download_root)
    success_list = []
    fail_list = []
    try:
        for subdir in folder_list:
            subdir = os.path.abspath(subdir)
            if 'audio_vocals.wav' not in os.listdir(subdir):
                fail_list.append(f"{subdir}: 缺少 audio_vocals.wav")
                continue
            if 'transcript.json' in os.listdir(subdir):
                logger.info(f'Transcript already exists in {subdir}')
                success_list.append(subdir)
                continue
            try:
                transcribe_audio(subdir, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers)
                success_list.append(subdir)
            except Exception as e:
                logger.error(f'Error transcribing audio in {subdir}: {e}')
                fail_list.append(f"{subdir}: {e}")
        logger.info(f'语音识别完成: 成功 {len(success_list)}/{len(folder_list)}, 失败 {len(fail_list)}')
        return f'成功: {len(success_list)}\n失败: {len(fail_list)}'
    finally:
        cleanup_whisperx()


if __name__ == '__main__':
    transcribe_all_audio_under_folder('videos')
    
    
