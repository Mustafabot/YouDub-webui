import shutil
from demucs.api import Separator
import os
from loguru import logger
import time
from .utils import save_wav, normalize_wav
from .config import ensure_ffmpeg_available, get_ffmpeg_path
import torch
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
separator = None

def init_demucs():
    global separator
    separator = load_model()
    
def load_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int=5) -> Separator:
    global separator
    if separator is not None:
        logger.info(f'Demucs model already loaded')
        return
    
    logger.info(f'Loading Demucs model: {model_name}')
    t_start = time.time()
    separator = Separator(model_name, device=auto_device if device=='auto' else device, progress=progress, shifts=shifts)
    t_end = time.time()
    logger.info(f'Demucs model loaded in {t_end - t_start:.2f} seconds')

def reload_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int=5) -> Separator:
    global separator
    logger.info(f'Reloading Demucs model: {model_name}')
    t_start = time.time()
    separator = Separator(model_name, device=auto_device if device=='auto' else device, progress=progress, shifts=shifts)
    t_end = time.time()
    logger.info(f'Demucs model reloaded in {t_end - t_start:.2f} seconds')
    
def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5) -> None:
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
    load_model(model_name, device, progress, shifts)
    t_start = time.time()
    try:
        origin, separated = separator.separate_audio_file(audio_path)
    except:
        # reload_model(model_name, device, progress, shifts)
                # origin, separated = separator.separate_audio_file(audio_path)
        time.sleep(5)
        logger.error(f'Error separating audio from {folder}')
        raise Exception(f'Error separating audio from {folder}')
    t_end = time.time()
    logger.info(f'Audio separated in {t_end - t_start:.2f} seconds')
    
    vocals = separated['vocals'].numpy().T
    instruments = None
    for k, v in separated.items():
        if k == 'vocals':
            continue
        if instruments is None:
            instruments = v
        else:
            instruments += v
    instruments = instruments.numpy().T
    
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')
    
    save_wav(vocals, vocal_output_path, sample_rate=44100)
    logger.info(f'Vocals saved to {vocal_output_path}')
    
    save_wav(instruments, instruments_output_path, sample_rate=44100)
    logger.info(f'Instruments saved to {instruments_output_path}')
    
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
    os.system(
        f'"{ffmpeg_path}" -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"')
    
    time.sleep(1)
    logger.info(f'Audio extracted from {folder}')
    return True
    
def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5) -> None:
    global separator
    found_video_dir = False
    for subdir, dirs, files in os.walk(root_folder):
        if 'download.mp4' not in files:
            if 'download.info.json' in files:
                raise FileNotFoundError(
                    f'发现视频目录 {subdir} 但缺少 download.mp4，'
                    f'请确认下载步骤已正确执行。目录内容: {files}'
                )
            continue
        
        found_video_dir = True
        if 'audio.wav' not in files:
            extract_audio_from_video(subdir)
        if 'audio_vocals.wav' not in files:
            separate_audio(subdir, model_name, device, progress, shifts)

    if not found_video_dir:
        raise FileNotFoundError(f'在 {root_folder} 下未找到任何包含 download.mp4 的视频目录')

    logger.info(f'All audio separated under {root_folder}')
    return f'All audio separated under {root_folder}'
    
if __name__ == '__main__':
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)
    
    
