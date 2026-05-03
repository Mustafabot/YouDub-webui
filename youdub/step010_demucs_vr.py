import shutil
from demucs.api import Separator
import os
import subprocess
from loguru import logger
import time
from .utils import save_wav, normalize_wav
from .config import ensure_ffmpeg_available, get_ffmpeg_path, PROJECT_ROOT
import torch
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
separator = None

def init_demucs(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5):
    pass


def cleanup_demucs():
    """清理 Demucs 模型，释放显存"""
    global separator
    import gc
    import torch
    
    if separator is not None:
        del separator
        separator = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info('Demucs 模型已清理，显存已释放')

def load_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int=5) -> Separator:
    global separator
    logger.info(f'Loading Demucs model: {model_name} (device={device}, shifts={shifts})')
    t_start = time.time()
    separator = Separator(model_name, device=auto_device if device=='auto' else device, progress=progress, shifts=shifts)
    t_end = time.time()
    logger.info(f'Demucs model loaded in {t_end - t_start:.2f} seconds')
    
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
    logger.info(f'注意: {model_name} 模型将分离 4 个音轨 (drums, bass, other, vocals) {shifts} 遍，每个音轨都会显示独立的进度条')
    load_model(model_name, device, progress, shifts)
    t_start = time.time()
    try:
        origin, separated = separator.separate_audio_file(audio_path)
    except Exception as e:
        logger.error(f'Error separating audio from {folder}: {e}')
        raise
    t_end = time.time()
    logger.info(f'Audio separated in {t_end - t_start:.2f} seconds, separated tracks: {list(separated.keys())}')
    
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
    
def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5) -> None:
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
                separate_audio(subdir, model_name, device, progress, shifts)
            else:
                logger.info(f'音频已分离，跳过: {subdir}')
            
            processed_dirs.add(subdir)
        
        logger.info(f'遍历完成，共处理 {len(processed_dirs)} 个视频目录')
        
        if not found_video_dir:
            raise FileNotFoundError(f'在 {root_folder} 下未找到任何包含 download.mp4 的视频目录')

        logger.info(f'All audio separated under {root_folder}')
        return f'All audio separated under {root_folder}'
    finally:
        cleanup_demucs()


def separate_audio_in_folders(folder_list, model_name="htdemucs_ft", device='auto', progress=True, shifts=5):
    """处理指定目录列表中的音频分离

    Args:
        folder_list: 需要处理的目录路径列表
        model_name: Demucs 模型名称
        device: 计算设备
        progress: 是否显示进度条
        shifts: 移位数
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
            separate_audio(subdir, model_name, device, progress, shifts)
            success_list.append(subdir)
        except Exception as e:
            logger.error(f'Error separating audio in {subdir}: {e}')
            fail_list.append(f"{subdir}: {e}")
    cleanup_demucs()
    logger.info(f'音频分离完成: 成功 {len(success_list)}/{len(folder_list)}, 失败 {len(fail_list)}')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'
    
if __name__ == '__main__':
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)
    
    
