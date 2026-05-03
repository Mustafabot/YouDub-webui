# -*- coding: utf-8 -*-
import json
import os
import subprocess
import time

from loguru import logger

from .config import ensure_ffmpeg_available, get_ffmpeg_path, get_ffprobe_path, PROJECT_ROOT

_video_encoder_cache = {"name": None}


def get_available_video_encoder():
    if _video_encoder_cache["name"] is not None:
        return _video_encoder_cache["name"]

    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        return None

    try:
        result = subprocess.run(
            [ffmpeg_path, '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
    except Exception:
        return None

    preference_order = [
        'libx264',
        'h264_nvenc',
        'h264_amf',
        'h264_qsv',
        'libopenh264',
    ]

    for encoder in preference_order:
        if f' {encoder} ' in output or f'\t{encoder}\t' in output or f' {encoder}\t' in output:
            _video_encoder_cache["name"] = encoder
            return encoder

    return None


def split_text(input_data,
               punctuations=['，', '；', '：', '。', '？', '！', '\n', '”']):
    # Chinese punctuation marks for sentence ending

    # Function to check if a character is a Chinese ending punctuation
    def is_punctuation(char):
        return char in punctuations

    # Process each item in the input data
    output_data = []
    for item in input_data:
        start = item["start"]
        text = item["translation"]
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = item["text"]
        sentence_start = 0

        if not text:
            continue
        duration_per_char = (item["end"] - item["start"]) / len(text)
        for i, char in enumerate(text):
            # If the character is a punctuation, split the sentence
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i+1]):
                continue
            sentence = text[sentence_start:i+1]
            sentence_end = start + duration_per_char * len(sentence)

            # Append the new item
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": original_text,
                "translation": sentence,
                "speaker": speaker
            })

            # Update the start for the next sentence
            start = sentence_end
            sentence_start = i + 1

    return output_data
    
def format_timestamp(seconds):
    """Converts seconds to the SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"

def generate_srt(translation, srt_path, speed_up=1, max_line_char=30):
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(translation):
            start = format_timestamp(line['start']/speed_up)
            end = format_timestamp(line['end']/speed_up)
            text = line['translation']
            line = len(text)//(max_line_char+1) + 1
            avg = min(round(len(text)/line), max_line_char)
            text = '\n'.join([text[i*avg:(i+1)*avg]
                             for i in range(line)])
            f.write(f'{i+1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{text}\n\n')


def get_aspect_ratio(video_path):
    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path:
        raise RuntimeError("FFprobe 未找到，无法获取视频信息。请运行自动下载或在配置中设置 FFMPEG_PATH")
    command = [ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'json', video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    dimensions = json.loads(result.stdout)['streams'][0]
    return dimensions['width'] / dimensions['height']


def convert_resolution(aspect_ratio, resolution='1080p'):
    if aspect_ratio < 1:
        width = int(resolution[:-1])
        height = int(width / aspect_ratio)
    else:
        height = int(resolution[:-1])
        width = int(height * aspect_ratio)
    # make sure width and height are divisibal by 2
    width = width - width % 2
    height = height - height % 2
    
    # return f'{width}x{height}'
    return width, height
    
def synthesize_video(folder, subtitles=True, use_original_audio=False, speed_up=1.05, fps=30, resolution='1080p'):
    ffmpeg_available, ffmpeg_msg = ensure_ffmpeg_available(auto_download=True)
    if not ffmpeg_available:
        raise RuntimeError(
            f'FFmpeg 不可用，无法进行视频合成。{ffmpeg_msg}\n'
            f'请按以下方式之一安装 FFmpeg：\n'
            f'1. 运行 python scripts/download_ffmpeg.py 自动下载\n'
            f'2. Windows: 从 https://ffmpeg.org/download.html 下载，解压后将 bin 目录添加到系统 PATH，或在配置中设置 FFMPEG_PATH\n'
            f'3. macOS: brew install ffmpeg\n'
            f'4. Linux: sudo apt install ffmpeg'
        )
    
    if os.path.exists(os.path.join(folder, 'video.mp4')):
        logger.info(f'Video already synthesized in {folder}')
        return
    
    translation_path = os.path.join(folder, 'translation.json')
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f'视频文件不存在: {input_video}，请确认下载步骤已正确执行')
    if not os.path.exists(translation_path):
        raise FileNotFoundError(f'翻译文件不存在: {translation_path}，请确认翻译步骤已正确执行')
    if not use_original_audio and not os.path.exists(input_audio):
        raise FileNotFoundError(f'合成音频不存在: {input_audio}，请确认TTS步骤已正确执行')
    
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)
        
    srt_path = os.path.join(folder, 'subtitles.srt')
    output_video = os.path.join(folder, 'video.mp4')
    generate_srt(translation, srt_path, speed_up)
    srt_path = srt_path.replace('\\', '/')
    srt_path = srt_path.replace(':', '\\:')
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    resolution = f'{width}x{height}'
    font_size = int(width/128)
    outline = int(round(font_size/8))
    video_speed_filter = f"setpts=PTS/{speed_up}"
    audio_speed_filter = f"atempo={speed_up}"
    subtitle_filter = f"subtitles=filename='{srt_path}':force_style='FontName=Arial,FontSize={font_size},PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline={outline},WrapStyle=2'"
    
    audio_stream = '0:a' if use_original_audio else '1:a'
    if subtitles:
        filter_complex = f"[0:v]{video_speed_filter},{subtitle_filter}[v];[{audio_stream}]{audio_speed_filter}[a]"
    else:
        filter_complex = f"[0:v]{video_speed_filter}[v];[{audio_stream}]{audio_speed_filter}[a]"
    
    ffmpeg_path = get_ffmpeg_path()
    video_encoder = get_available_video_encoder()
    if not video_encoder:
        raise RuntimeError(
            '未找到可用的 H.264 视频编码器。当前 FFmpeg 版本可能不支持 libx264/libopenh264/NVENC/AMF/QSV。\n'
            '请安装包含 H.264 编码支持的 FFmpeg 版本。'
        )
    if video_encoder != 'libx264':
        logger.info(f'使用视频编码器: {video_encoder} (libx264 不可用)')
    ffmpeg_command = [
        ffmpeg_path,
        '-i', input_video,
    ]
    if not use_original_audio:
        ffmpeg_command.extend(['-i', input_audio])
    ffmpeg_command.extend([
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '[a]',
        '-r', str(fps),
        '-s', resolution,
        '-c:v', video_encoder,
        '-c:a', 'aac',
        output_video,
        '-y'
    ])
    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f'FFmpeg 视频合成失败 (返回码 {result.returncode}):')
        for line in result.stderr.strip().split('\n'):
            if line.strip():
                logger.error(f'  {line.strip()}')
        raise RuntimeError(f'FFmpeg 视频合成失败，返回码: {result.returncode}')
    time.sleep(1)
    

def synthesize_all_video_under_folder(folder, subtitles=True, use_original_audio=False, speed_up=1.05, fps=30, resolution='1080p'):
    if not os.path.isabs(folder):
        folder = str(PROJECT_ROOT / folder)
    found_video_dir = False
    for root, dirs, files in os.walk(folder):
        if 'download.mp4' not in files and 'video.mp4' not in files:
            continue
        found_video_dir = True
        if 'download.mp4' not in files:
            raise FileNotFoundError(
                f'发现视频目录 {root} 但缺少 download.mp4，请确认下载步骤已正确执行。目录内容: {files}'
            )
        if 'video.mp4' in files:
            continue
        synthesize_video(root, subtitles=subtitles, use_original_audio=use_original_audio,
                         speed_up=speed_up, fps=fps, resolution=resolution)
    if not found_video_dir:
        raise FileNotFoundError(f'在 {folder} 下未找到任何视频处理目录')
    return f'Synthesized all videos under {folder}'
def synthesize_video_in_folders(folder_list, subtitles=True, use_original_audio=False, speed_up=1.05, fps=30, resolution='1080p'):
    """处理指定目录列表中的视频合成

    Args:
        folder_list: 需要处理的目录路径列表
        subtitles: 是否添加字幕
        use_original_audio: 是否使用原视频音轨
        speed_up: 加速倍率
        fps: 帧率
        resolution: 分辨率
    """
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    success_list = []
    fail_list = []
    for subdir in folder_list:
        subdir = os.path.abspath(subdir)
        files = os.listdir(subdir) if os.path.exists(subdir) else []
        if 'download.mp4' not in files:
            fail_list.append(f"{subdir}: 缺少 download.mp4")
            continue
        if 'video.mp4' in files:
            logger.info(f'Video already synthesized in {subdir}')
            success_list.append(subdir)
            continue
        try:
            synthesize_video(subdir, subtitles=subtitles, use_original_audio=use_original_audio,
                             speed_up=speed_up, fps=fps, resolution=resolution)
            success_list.append(subdir)
        except Exception as e:
            logger.error(f'Error synthesizing video in {subdir}: {e}')
            fail_list.append(f"{subdir}: {e}")
    logger.info(f'视频合成完成: 成功 {len(success_list)}/{len(folder_list)}, 失败 {len(fail_list)}')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'


if __name__ == '__main__':
    folder = r'videos\3Blue1Brown\20231207 Im still astounded this is true'
    synthesize_all_video_under_folder(folder, subtitles=True)
