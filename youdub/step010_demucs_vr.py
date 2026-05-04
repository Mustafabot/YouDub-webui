import shutil
from demucs.api import Separator
import os
import subprocess
from loguru import logger
import time
import gc
import numpy as np
from .utils import save_wav, normalize_wav
from .config import ensure_ffmpeg_available, get_ffmpeg_path, PROJECT_ROOT
import torch
import wave

auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
separator = None

_DEFAULT_MAX_CHUNK_SECONDS = 600
_DEFAULT_SEGMENT = 10
_OVERLAP_SECONDS = 10

def init_demucs(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5, segment: int = _DEFAULT_SEGMENT):
    pass


def cleanup_demucs():
    """清理 Demucs 模型，释放显存"""
    global separator
    if separator is not None:
        del separator
        separator = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info('Demucs 模型已清理，显存已释放')

def load_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int=5, segment: int = _DEFAULT_SEGMENT) -> Separator:
    global separator
    logger.info(f'Loading Demucs model: {model_name} (device={device}, shifts={shifts}, segment={segment}s)')
    t_start = time.time()
    effective_device = auto_device if device=='auto' else device
    separator = Separator(model_name, device=effective_device, progress=progress, shifts=shifts, segment=segment)
    t_end = time.time()
    logger.info(f'Demucs model loaded in {t_end - t_start:.2f} seconds')

def _get_audio_duration(audio_path: str) -> float:
    with wave.open(audio_path, 'rb') as wf:
        return wf.getnframes() / wf.getframerate()

def _extract_audio_chunk(audio_path: str, output_path: str, start_seconds: float, duration_seconds: float) -> str:
    ffmpeg_path = get_ffmpeg_path()
    cmd = [
        str(ffmpeg_path),
        '-loglevel', 'error',
        '-ss', str(start_seconds),
        '-i', str(audio_path),
        '-t', str(duration_seconds),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
        str(output_path)
    ]
    logger.debug(f'提取分块: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else '未知错误'
        raise RuntimeError(f'FFmpeg 分块提取失败: {error_msg}')
    return output_path

def _apply_crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap_samples: int) -> np.ndarray:
    if overlap_samples <= 0 or chunk1 is None:
        if chunk1 is None:
            return chunk2
        if chunk2 is None:
            return chunk1
        return np.concatenate([chunk1, chunk2])
    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
    if chunk1.ndim == 2:
        fade_out = fade_out[:, np.newaxis]
        fade_in = fade_in[:, np.newaxis]
    tail = chunk1[-overlap_samples:]
    head = chunk2[:overlap_samples:]
    crossfaded = tail.astype(np.float32) * fade_out + head.astype(np.float32) * fade_in
    result_shape = list(chunk1.shape)
    result_shape[0] = len(chunk1) - overlap_samples + len(chunk2)
    result = np.empty(result_shape, dtype=np.float32)
    result[:len(chunk1) - overlap_samples] = chunk1[:-overlap_samples]
    result[len(chunk1) - overlap_samples:len(chunk1)] = crossfaded
    result[len(chunk1):] = chunk2[overlap_samples:]
    return result

def _merge_instruments(separated: dict):
    vocals = separated['vocals'].numpy().T
    del separated['vocals']
    instrument_keys = [k for k in separated.keys() if k != 'vocals']
    instruments = None
    for k in instrument_keys:
        v = separated[k]
        if instruments is None:
            instruments = v.clone()
        else:
            instruments.add_(v)
        del separated[k]
    instruments_np = instruments.numpy().T if instruments is not None else np.zeros_like(vocals)
    del instruments
    return vocals, instruments_np

def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5, segment: int = _DEFAULT_SEGMENT, max_chunk_seconds: int = _DEFAULT_MAX_CHUNK_SECONDS) -> None:
    global separator
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'音频文件不存在: {audio_path}，请确认步骤01已正确执行')
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')
    
    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'Audio already separated in {folder}')
        return
    
    logger.info(f'Separating audio from {folder}')
    logger.info(f'注意: {model_name} 模型将分离 4 个音轨 (drums, bass, other, vocals) {shifts} 遍，每个音轨都会显示独立的进度条')
    
    duration = _get_audio_duration(audio_path)
    logger.info(f'音频时长: {duration:.1f}秒, 最大分块: {max_chunk_seconds}秒, 内部段长: {segment}秒')
    
    if duration <= max_chunk_seconds:
        logger.info(f'音频时长 {duration:.1f}秒 不超过最大分块 {max_chunk_seconds}秒，直接处理')
        _separate_audio_direct(folder, audio_path, model_name, device, progress, shifts, segment)
    else:
        logger.info(f'音频时长 {duration:.1f}秒 超过最大分块 {max_chunk_seconds}秒，启动分块处理模式 (每块约{max_chunk_seconds}秒, 重叠{_OVERLAP_SECONDS}秒)')
        _separate_audio_chunked(folder, audio_path, duration, model_name, device, progress, shifts, segment, max_chunk_seconds)

def _separate_audio_direct(folder: str, audio_path: str, model_name: str, device: str, progress: bool, shifts: int, segment: int) -> None:
    load_model(model_name, device, progress, shifts, segment)
    t_start = time.time()
    try:
        origin, separated = separator.separate_audio_file(audio_path)
    except Exception as e:
        logger.error(f'Error separating audio: {e}')
        raise
    t_end = time.time()
    logger.info(f'Audio separated in {t_end - t_start:.2f} seconds, separated tracks: {list(separated.keys())}')
    
    del origin
    gc.collect()
    
    _save_separated_tracks(separated, folder)

def _save_separated_tracks(separated: dict, folder: str) -> None:
    vocals = separated['vocals'].numpy().T
    del separated['vocals']
    gc.collect()
    
    instrument_keys = [k for k in separated.keys() if k != 'vocals']
    instruments = None
    for k in instrument_keys:
        v = separated[k]
        if instruments is None:
            instruments = v.clone()
        else:
            instruments.add_(v)
        del separated[k]
        gc.collect()
    
    instruments_np = instruments.numpy().T
    del instruments
    gc.collect()
    
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')
    
    save_wav(vocals, vocal_output_path, sample_rate=44100)
    logger.info(f'Vocals saved to {vocal_output_path}')
    del vocals
    gc.collect()
    
    save_wav(instruments_np, instruments_output_path, sample_rate=44100)
    logger.info(f'Instruments saved to {instruments_output_path}')
    del instruments_np
    gc.collect()

def _separate_audio_chunked(folder: str, audio_path: str, duration: float,
                            model_name: str, device: str, progress: bool, shifts: int, segment: int,
                            max_chunk_seconds: int) -> None:
    load_model(model_name, device, progress, shifts, segment)
    
    stride_seconds = max_chunk_seconds
    overlap_seconds = _OVERLAP_SECONDS
    sample_rate = 44100
    
    chunk_starts = list(range(0, int(duration), stride_seconds))
    if len(chunk_starts) > 1 and (duration - chunk_starts[-1]) < overlap_seconds:
        chunk_starts.pop()
    
    num_chunks = len(chunk_starts)
    logger.info(f'将音频分为 {num_chunks} 个分块处理 (步长 {stride_seconds}秒, 重叠 {overlap_seconds}秒)')
    
    vocals_result = None
    instruments_result = None
    tmp_files = []
    
    try:
        for i, start_sec in enumerate(chunk_starts):
            if i == 0:
                chunk_start_sec = 0
            else:
                chunk_start_sec = max(0, start_sec - overlap_seconds)
            
            core_end_sec = min(start_sec + stride_seconds, duration)
            if i < num_chunks - 1:
                chunk_end_sec = min(start_sec + stride_seconds + overlap_seconds, duration)
            else:
                chunk_end_sec = duration
            
            chunk_duration_sec = chunk_end_sec - chunk_start_sec
            
            logger.info(f'处理分块 {i+1}/{num_chunks}: 源音频 {chunk_start_sec:.1f}s - {chunk_end_sec:.1f}s ({chunk_duration_sec:.1f}秒)')
            
            tmp_path = os.path.join(folder, f'_tmp_chunk_{i}.wav')
            _extract_audio_chunk(audio_path, tmp_path, chunk_start_sec, chunk_duration_sec)
            tmp_files.append(tmp_path)
            
            t_start = time.time()
            try:
                _, separated = separator.separate_audio_file(tmp_path)
            except Exception as e:
                logger.error(f'Error separating chunk {i+1}/{num_chunks}: {e}')
                raise
            t_end = time.time()
            logger.info(f'分块 {i+1}/{num_chunks} 分离完成, 耗时 {t_end - t_start:.2f}秒, 音轨: {list(separated.keys())}')
            
            chunk_vocals, chunk_instruments = _merge_instruments(separated)
            del separated
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if i == 0:
                vocals_result = chunk_vocals
                instruments_result = chunk_instruments
            else:
                overlap_samples = int(overlap_seconds * sample_rate)
                actual_overlap = min(overlap_samples, len(vocals_result), len(chunk_vocals))
                logger.debug(f'分块 {i+1}: 交叉淡出 {actual_overlap} 采样点 ({actual_overlap/sample_rate:.1f}秒)')
                vocals_result = _apply_crossfade(vocals_result, chunk_vocals, actual_overlap)
                instruments_result = _apply_crossfade(instruments_result, chunk_instruments, actual_overlap)
                del chunk_vocals, chunk_instruments
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info(f'分块处理完成，拼接后 vocals: {vocals_result.shape}, instruments: {instruments_result.shape}')
        
        vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
        instruments_output_path = os.path.join(folder, 'audio_instruments.wav')
        
        save_wav(vocals_result, vocal_output_path, sample_rate=44100)
        logger.info(f'Vocals saved to {vocal_output_path}')
        del vocals_result
        gc.collect()
        
        save_wav(instruments_result, instruments_output_path, sample_rate=44100)
        logger.info(f'Instruments saved to {instruments_output_path}')
        del instruments_result
        gc.collect()
    
    finally:
        for tmp_path in tmp_files:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        if tmp_files:
            logger.info(f'已清理 {len(tmp_files)} 个临时分块文件')

def extract_audio_from_video(folder: str) -> bool:
    ffmpeg_available, ffmpeg_msg = ensure_ffmpeg_available(auto_download=True)
    if not ffmpeg_available:
        raise RuntimeError(
            f'FFmpeg 不可用，无法进行音频提取。{ffmpeg_msg}\n'
            f'请按以下方式之一安装 FFmpeg：\n'
            f'1. 运行 python scripts/download_ffmpeg.py 自动下载\n'
            f'2. Windows: 从 https://ffmpeg.org/download.html 下载，解压后将 bin 目录添加到系统 PATH，或在配置中设置 FFMPEG_PATH\n'
            f'3. macOS: brew install ffmpeg\n'
            f'4. Linux: sudo apt install ffmpeg'
        )
    
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'视频文件不存在: {video_path}，请确认下载步骤已正确执行')
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'Audio already extracted in {folder}')
        return True
    logger.info(f'Extracting audio from {folder}')

    ffmpeg_path = get_ffmpeg_path()
    cmd = [
        str(ffmpeg_path),
        '-loglevel', 'error',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
        str(audio_path)
    ]
    
    logger.info(f'执行 FFmpeg 命令: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else '未知错误'
        raise RuntimeError(
            f'音频提取失败，FFmpeg 返回错误码 {result.returncode}: {error_msg}\n'
            f'命令: {" ".join(cmd)}'
        )
    
    if not os.path.exists(audio_path):
        raise RuntimeError(
            f'FFmpeg 执行成功但未生成音频文件: {audio_path}\n'
            f'请检查视频文件 {video_path} 是否损坏'
        )
    
    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise RuntimeError(f'生成的音频文件为空: {audio_path}')
    
    logger.info(f'Audio extracted from {folder}, size: {file_size} bytes')
    return True
    
def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5, segment: int = _DEFAULT_SEGMENT, max_chunk_seconds: int = _DEFAULT_MAX_CHUNK_SECONDS) -> None:
    if not os.path.isabs(root_folder):
        root_folder = str(PROJECT_ROOT / root_folder)
    global separator
    found_video_dir = False
    processed_dirs = set()
    
    logger.info(f'开始遍历目录: {root_folder}')
    try:
        for subdir, dirs, files in os.walk(root_folder):
            if subdir in processed_dirs:
                logger.debug(f'跳过已处理目录: {subdir}')
                continue
            
            video_path = os.path.join(subdir, 'download.mp4')
            if not os.path.exists(video_path):
                if os.path.exists(os.path.join(subdir, 'download.info.json')):
                    raise FileNotFoundError(
                        f'发现视频目录 {subdir} 但缺少 download.mp4，'
                        f'请确认下载步骤已正确执行。'
                    )
                continue
            
            found_video_dir = True
            logger.info(f'处理视频目录: {subdir}')
            
            audio_path = os.path.join(subdir, 'audio.wav')
            if not os.path.exists(audio_path):
                logger.info(f'提取音频: {subdir}')
                extract_audio_from_video(subdir)
            
            vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
            instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
            if not os.path.exists(vocal_output_path) or not os.path.exists(instruments_output_path):
                logger.info(f'分离人声和伴奏: {subdir}')
                separate_audio(subdir, model_name, device, progress, shifts, segment, max_chunk_seconds)
            else:
                logger.info(f'音频已分离，跳过: {subdir}')
            
            processed_dirs.add(subdir)
            gc.collect()
        
        logger.info(f'遍历完成，共处理 {len(processed_dirs)} 个视频目录')
        
        if not found_video_dir:
            raise FileNotFoundError(f'在 {root_folder} 下未找到任何包含 download.mp4 的视频目录')

        logger.info(f'All audio separated under {root_folder}')
        return f'All audio separated under {root_folder}'
    finally:
        cleanup_demucs()


def separate_audio_in_folders(folder_list, model_name="htdemucs_ft", device='auto', progress=True, shifts=5, segment=_DEFAULT_SEGMENT, max_chunk_seconds=_DEFAULT_MAX_CHUNK_SECONDS):
    """处理指定目录列表中的音频分离

    Args:
        folder_list: 需要处理的目录路径列表
        model_name: Demucs 模型名称
        device: 计算设备
        progress: 是否显示进度条
        shifts: 移位数
        segment: Demucs 内部分段推理的秒数，控制显存占用
        max_chunk_seconds: 外部分块处理的最大分块秒数，控制内存占用
    """
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    global separator
    success_list = []
    fail_list = []
    for subdir in folder_list:
        subdir = os.path.abspath(subdir)
        video_path = os.path.join(subdir, 'download.mp4')
        if not os.path.exists(video_path):
            fail_list.append(f"{subdir}: 缺少 download.mp4")
            continue
        try:
            audio_path = os.path.join(subdir, 'audio.wav')
            if not os.path.exists(audio_path):
                logger.info(f'提取音频: {subdir}')
                extract_audio_from_video(subdir)
            vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
            instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
            if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
                logger.info(f'音频已分离，跳过: {subdir}')
                success_list.append(subdir)
                continue
            separate_audio(subdir, model_name, device, progress, shifts, segment, max_chunk_seconds)
            success_list.append(subdir)
        except Exception as e:
            logger.error(f'Error separating audio in {subdir}: {e}')
            fail_list.append(f"{subdir}: {e}")
        gc.collect()
    cleanup_demucs()
    logger.info(f'音频分离完成: 成功 {len(success_list)}/{len(folder_list)}, 失败 {len(fail_list)}')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'
    
if __name__ == '__main__':
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)
