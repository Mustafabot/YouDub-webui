import json
import os
import re
import librosa

from loguru import logger
import numpy as np

from .utils import save_wav, save_wav_norm
from .config import get_config, PROJECT_ROOT
from .step041_tts_bytedance import tts as bytedance_tts
from .step043_tts_f5 import tts as f5_tts, F5_AVAILABLE
from .step020_whisperx import generate_speaker_audio
from .cn_tx import TextNorm
from audiostretchy.stretch import stretch_audio
normalizer = TextNorm()
def preprocess_text(text):
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    # 使用正则表达式在字母和数字之间插入空格
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text
    
    
def adjust_audio_length(wav_path, desired_length, sample_rate = 24000, min_speed_factor = 0.6, max_speed_factor = 1.1):
    wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    current_length = len(wav)/sample_rate
    speed_factor = max(
        min(desired_length / current_length, max_speed_factor), min_speed_factor)
    desired_length = current_length * speed_factor
    target_path = wav_path.replace('.wav', f'_adjusted.wav')
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    return wav[:int(desired_length*sample_rate)], desired_length

def generate_wavs(folder, force_bytedance=False):
    transcript_path = os.path.join(folder, 'translation.json')
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f'翻译文件不存在: {transcript_path}，请确认翻译步骤已正确执行')
    
    audio_vocals_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(audio_vocals_path):
        raise FileNotFoundError(f'人声音频不存在: {audio_vocals_path}，请确认音频分离步骤已正确执行')
    
    audio_instruments_path = os.path.join(folder, 'audio_instruments.wav')
    if not os.path.exists(audio_instruments_path):
        raise FileNotFoundError(f'伴奏音频不存在: {audio_instruments_path}，请确认音频分离步骤已正确执行')
    
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    speakers = set()
    
    for line in transcript:
        speakers.add(line['speaker'])
    num_speakers = len(speakers)
    num_segments = len(transcript)
    logger.info(f'共 {num_speakers} 个说话人, {num_segments} 个片段需要合成')
    
    has_bytedance_config = bool(get_config('BYTEDANCE_APPID') and get_config('BYTEDANCE_ACCESS_TOKEN'))
    
    if force_bytedance:
        if not has_bytedance_config:
            raise RuntimeError('强制使用火山引擎 TTS 但未配置。请在设置中配置 BYTEDANCE_APPID 和 BYTEDANCE_ACCESS_TOKEN')
        use_bytedance = True
    elif num_speakers == 1 and has_bytedance_config:
        use_bytedance = True
        logger.info('单说话人场景，使用火山引擎 TTS')
    elif F5_AVAILABLE:
        use_bytedance = False
        logger.info('使用 F5-TTS 声音克隆')
    else:
        if has_bytedance_config:
            use_bytedance = True
            logger.warning('F5-TTS 未安装 (pip install f5-tts)，将使用火山引擎 TTS')
        else:
            raise RuntimeError(
                '所有 TTS 引擎均不可用。请至少配置一种 TTS 方式：\n'
                '1. 配置火山引擎 TTS (BYTEDANCE_APPID + BYTEDANCE_ACCESS_TOKEN)\n'
                '2. 安装 F5-TTS (pip install f5-tts)'
            )

    ref_text_map = {}
    speaker_folder = os.path.join(folder, 'SPEAKER')
    
    speakers = set()
    for line in transcript:
        speakers.add(line['speaker'])
    
    speaker_files_exist = True
    if not os.path.exists(speaker_folder):
        speaker_files_exist = False
    else:
        for speaker in speakers:
            speaker_wav = os.path.join(speaker_folder, f'{speaker}.wav')
            if not os.path.exists(speaker_wav):
                speaker_files_exist = False
                break
    
    if not speaker_files_exist:
        logger.info(f'说话人音频文件不完整，重新生成: {speaker_folder}')
        generate_speaker_audio(folder, transcript)
    
    if not use_bytedance:
        for line in transcript:
            speaker = line['speaker']
            original_text = line.get('original', line.get('text', ''))
            if speaker not in ref_text_map:
                ref_text_map[speaker] = []
            ref_text_map[speaker].append(original_text)
        for speaker in ref_text_map:
            ref_text_map[speaker] = ' '.join(ref_text_map[speaker][:3])

    full_wav = np.zeros((0, ))
    total_segments = len(transcript)
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        logger.info(f"[{i+1}/{total_segments}] 正在合成 ({speaker}): {text[:50]}")
        if use_bytedance:
            voice_type = 'BV701_streaming' if num_speakers == 1 else None
            bytedance_tts(text, output_path, speaker_wav, voice_type=voice_type)
        else:
            f5_tts(text, output_path, speaker_wav, ref_text=ref_text_map.get(speaker, ''))
        if not os.path.exists(output_path):
            raise RuntimeError(f'TTS 生成失败: {output_path}，请检查 TTS 配置')
        start = line['start']
        end = line['end']
        length = end-start
        last_end = len(full_wav)/24000
        if start > last_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000), ))))
        start = len(full_wav)/24000
        line['start'] = start
        if i < len(transcript) - 1:
            next_line = transcript[i+1]
            next_end = next_line['end']
            end = min(start + length, next_end)
        wav, length = adjust_audio_length(output_path, end-start)

        full_wav = np.concatenate((full_wav, wav))
        line['end'] = start + length
        
    vocal_wav, sr = librosa.load(os.path.join(folder, 'audio_vocals.wav'), sr=24000)
    full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav))
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    
    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)
    
    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')
    combined_wav = full_wav + instruments_wav
    # combined_wav /= np.max(np.abs(combined_wav))
    save_wav_norm(combined_wav, os.path.join(folder, 'audio_combined.wav'))
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')
        

def generate_all_wavs_under_folder(root_folder, force_bytedance=False):
    if not os.path.isabs(root_folder):
        root_folder = str(PROJECT_ROOT / root_folder)
    logger.info(f'开始语音合成扫描: {root_folder}')
    found_video_dir = False
    for root, dirs, files in os.walk(root_folder):
        dirs[:] = [d for d in dirs if d not in ('wavs', 'SPEAKER')]
        if 'translation.json' not in files and 'audio_combined.wav' not in files:
            continue
        found_video_dir = True
        if 'translation.json' not in files:
            raise FileNotFoundError(
                f'发现视频目录 {root} 但缺少 translation.json，请确认翻译步骤已正确执行。目录内容: {files}'
            )
        if 'audio_combined.wav' in files:
            logger.info(f'跳过已合成目录: {root}')
            continue
        logger.info(f'处理目录: {root}')
        generate_wavs(root, force_bytedance)
    if not found_video_dir:
        raise FileNotFoundError(f'在 {root_folder} 下未找到任何视频处理目录')
    logger.info(f'语音合成完成: {root_folder}')
    return f'Generated all wavs under {root_folder}'

if __name__ == '__main__':
    folder = r'videos\TED-Ed\20211214 Would you raise the bird that murdered your children？ - Steve Rothstein'
    generate_wavs(folder, force_bytedance=False)
