# -*- coding: utf-8 -*-
"""
step040_tts.py — TTS 调度层（步骤4）

所属流水线位置：
  步骤4/7：TTS 语音合成调度模块
  上游依赖：步骤2（WhisperX 语音识别）生成的 translation.json、步骤1（Demucs 人声分离）生成的 audio_vocals.wav
  下游产出：audio_tts.wav（纯 TTS 音频）、audio_combined.wav（TTS + 伴奏混合音频）

功能说明：
  1. 根据说话人数量和可用引擎自动选择 TTS 后端（火山引擎 / IndexTTS）
  2. 遍历 translation.json 中的每个片段，逐个合成语音
  3. 对合成的音频进行时长对齐（拉伸/填充静音），使其与原片段时间线匹配
  4. 将所有片段拼接为完整音频，并与伴奏（audio_instruments.wav）混合输出

输入文件：
  - {folder}/translation.json          : 翻译结果，每段含 start/end/speaker/translation
  - {folder}/SPEAKER/{speaker}.wav     : 各说话人的参考音频（用于音色克隆或声纹匹配）
  - {folder}/audio_vocals.wav          : 分离出的人声（用于音量归一化参考）
  - {folder}/audio_instruments.wav     : 分离出的伴奏（用于最终混合）

输出文件：
  - {folder}/wavs/*.wav                : 每段翻译对应的合成音频文件
  - {folder}/audio_tts.wav             : 纯 TTS 拼接音频
  - {folder}/audio_combined.wav        : TTS + 伴奏的混合音频
  - {folder}/translation.json          : 更新后的时间戳（已对齐）
"""

import json       # 用于读写 translation.json 和 summary.json
import os         # 文件和路径操作：目录创建、路径拼接、文件存在性检查
import re         # 正则表达式，用于文本预处理中的格式规范化
import librosa    # 音频处理库：加载音频文件、获取采样率、计算时长

from loguru import logger            # 结构化日志记录，带色彩输出和文件轮转
import numpy as np                   # 数值计算库，用于音频信号数组操作（拼接、填充、归一化）

from .utils import save_wav          # 工具函数：将 numpy 数组保存为 WAV 文件
from .config import get_config, PROJECT_ROOT  # 配置管理：读取配置项、项目根路径
from .step041_tts_bytedance import tts as bytedance_tts  # 火山引擎 TTS 函数
from .step043_tts_indextts import tts as indextts_tts, INDEXTTS_AVAILABLE  # IndexTTS 函数及可用性标志
from .step020_whisperx import generate_speaker_audio      # 说话人音频生成函数
from .cn_tx import TextNorm            # 中文文本规范化器（繁体转简体、数字格式化等）
from audiostretchy.stretch import stretch_audio  # 音频拉伸库，用于改变音频播放速度而不改变音调


normalizer = TextNorm()  # 全局中文文本规范化器实例


def preprocess_text(text):
    """
    文本预处理函数：在 TTS 合成前对文本进行规范化处理。

    处理流程：
      1. 将 "AI" 替换为中文 "人工智能"（避免字母发音问题）
      2. 在连续大写字母之间插入空格（改善英文缩写的朗读效果）
      3. 调用 TextNorm 进行中文规范化（繁体转简体、数字格式化等）
      4. 在字母和数字之间插入空格（改善混合内容的发音）

    Args:
        text (str): 原始文本（可能包含中英文混合内容）

    Returns:
        str: 预处理后的文本，适合 TTS 引擎合成
    """
    # 将英文缩写 "AI" 替换为中文全称，提升中文 TTS 的自然度
    text = text.replace('AI', '人工智能')
    # 在连续大写字母之间插入空格，防止 TTS 将多个字母当作一个单词读
    # 例如 "ABC" -> "A B C"，使用负向前瞻排除行首的单个字母
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    # 调用中文文本规范化器（处理繁体字、全角符号、数字格式等）
    text = normalizer(text)
    # 在字母和数字之间插入空格，避免 "abc123" 被错误连读
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text


def adjust_audio_length(wav_path, desired_length, sample_rate=24000):
    """
    调整音频长度以匹配目标时长（通过速度拉伸实现）。

    使用 audiostretchy 库进行时域拉伸，保持音调不变。
    拉伸速度受配置项 TTS_STRETCH_MIN_SPEED 和 TTS_STRETCH_MAX_SPEED 限制。

    Args:
        wav_path (str):         原始 WAV 文件路径
        desired_length (float): 目标时长（秒）
        sample_rate (int):      采样率，默认 24000 Hz

    Returns:
        tuple: (调整后的音频 numpy 数组, 实际拉伸后的时长（秒）)
    """
    # 从配置中读取最小和最大速度因子（防止过度拉伸导致音质劣化）
    min_speed_factor = float(get_config('TTS_STRETCH_MIN_SPEED', 0.6))
    max_speed_factor = float(get_config('TTS_STRETCH_MAX_SPEED', 1.3))
    # 加载原始音频
    wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    # 计算原始时长（秒）
    current_length = len(wav) / sample_rate
    # 计算所需速度因子并限制在允许范围内
    speed_factor = max(
        min(desired_length / current_length, max_speed_factor), min_speed_factor)
    # 按限制后的速度因子重新计算目标时长
    desired_length = current_length * speed_factor
    # 生成调整后的音频文件路径（在原文件名后加 _adjusted 后缀）
    target_path = wav_path.replace('.wav', f'_adjusted.wav')
    # 调用 audiostretchy 进行速度拉伸（保持音调不变）
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    # 重新加载拉伸后的音频
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    # 返回前 desired_length 秒的音频
    return wav[:int(desired_length * sample_rate)], desired_length


def distribute_extra_silence(wav, sample_rate, desired_length,
                             silence_threshold=0.02,
                             min_silence_ms=50):
    """
    在音频的静音片段中均匀分配额外时长，以达到目标总时长。

    用于时长补齐：当拉伸后的音频仍短于目标时长时，不在末尾简单补零，
    而是将缺少的时长按比例分配到已有的静音间隙中，使听感更自然。

    Args:
        wav (np.ndarray):       输入音频数组
        sample_rate (int):      采样率
        desired_length (float): 目标时长（秒）
        silence_threshold (float): 静音判定阈值（相对于最大幅值的比例），默认 0.02
        min_silence_ms (int):   最小静音片段长度（毫秒），小于此值的不作为静音间隙

    Returns:
        np.ndarray: 补齐时长后的音频数组（如果已经够长则直接返回原数组）
    """
    # 计算当前音频时长
    current_len = len(wav) / sample_rate
    # 如果当前时长已达到或超过目标时长，直接返回
    if current_len >= desired_length:
        return wav
    # 计算需要补充的采样点数
    extra_samples = int((desired_length - current_len) * sample_rate)
    # 获取音频的绝对值数组，用于判断静音
    abs_wav = np.abs(wav)
    # 计算静音阈值（基于最大幅值的百分比）
    threshold = silence_threshold * np.max(abs_wav)
    # 标记每个采样点是否为静音
    is_silent = abs_wav < threshold

    # 扫描并记录所有的静音片段（连续静音区域）
    silence_runs = []        # 每个元素为 [start_index, end_index]
    in_silence = False       # 当前是否在静音片段中
    run_start = 0            # 当前静音片段的起始位置
    # 将最小静音片段的毫秒数转换为采样点数
    min_silence_samples = int(min_silence_ms / 1000 * sample_rate)

    # 逐采样点扫描音频，标记静音区间
    for i in range(len(is_silent)):
        if is_silent[i] and not in_silence:
            # 进入静音区域
            in_silence = True
            run_start = i
        elif not is_silent[i] and in_silence:
            # 离开静音区域
            in_silence = False
            # 仅保留长度超过最小阈值的静音片段
            if i - run_start >= min_silence_samples:
                silence_runs.append([run_start, i])

    # 处理音频末尾正在进行的静音片段
    if in_silence and len(is_silent) - run_start >= min_silence_samples:
        silence_runs.append([run_start, len(is_silent)])

    # 如果没有找到符合条件的静音片段，在末尾补零
    if not silence_runs:
        return np.concatenate((wav, np.zeros(extra_samples)))

    # 计算所有静音片段的总长度（采样点数）
    total_silence = float(sum(end - start for start, end in silence_runs))

    # 在静音片段中按比例插入额外的静音采样点
    result_parts = []  # 存储构建结果音频的各片段
    last_end = 0       # 上一个处理到的位置
    for start, end in silence_runs:
        # 添加静音片段前的非静音部分
        result_parts.append(wav[last_end:start])
        # 添加原始静音片段
        result_parts.append(wav[start:end])
        # 计算该静音片段在总静音中的占比，插入对应比例的额外静音
        proportion = (end - start) / total_silence
        extra_len = int(extra_samples * proportion)
        if extra_len > 0:
            result_parts.append(np.zeros(extra_len))
        last_end = end
    # 添加最后一个静音片段之后的非静音部分
    result_parts.append(wav[last_end:])

    # 拼接所有片段
    result = np.concatenate(result_parts)
    # 确保不超过目标长度
    return result[:int(desired_length * sample_rate)]


def generate_wavs(folder, force_bytedance=False):
    """
    核心函数：为指定文件夹中的翻译文本逐段合成语音。

    执行流程：
      1. 检查输入文件（translation.json, audio_vocals.wav, audio_instruments.wav）是否存在
      2. 根据说话人数量和可用引擎决策使用哪个 TTS 后端
      3. 确保说话人参考音频完整，缺失时调用 generate_speaker_audio 重新生成
      4. 逐段调用 TTS 合成，进行时长对齐
      5. 拼接所有音频片段，与伴奏混合后输出

    Args:
        folder (str):           视频处理目录的路径
        force_bytedance (bool): 是否强制使用火山引擎 TTS（默认 False，自动选择）

    Raises:
        FileNotFoundError: 翻译文件或人声/伴奏音频文件不存在
        RuntimeError:      所有 TTS 引擎均不可用时抛出

    Returns:
        None（结果直接写入文件夹内的文件）
    """
    # ---- 输入文件存在性检查 ----
    # 翻译结果文件路径
    transcript_path = os.path.join(folder, 'translation.json')
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f'翻译文件不存在: {transcript_path}，请确认翻译步骤已正确执行')

    # 分离出的人声音频路径
    audio_vocals_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(audio_vocals_path):
        raise FileNotFoundError(f'人声音频不存在: {audio_vocals_path}，请确认音频分离步骤已正确执行')

    # 分离出的伴奏音频路径
    audio_instruments_path = os.path.join(folder, 'audio_instruments.wav')
    if not os.path.exists(audio_instruments_path):
        raise FileNotFoundError(f'伴奏音频不存在: {audio_instruments_path}，请确认音频分离步骤已正确执行')

    # ---- 创建输出目录 ----
    # TTS 生成的每段音频存放在 wavs/ 子目录
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- 加载翻译数据 ----
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    # ---- 统计说话人和片段数量 ----
    speakers = set()
    for line in transcript:
        speakers.add(line['speaker'])
    num_speakers = len(speakers)
    num_segments = len(transcript)
    logger.info(f'共 {num_speakers} 个说话人, {num_segments} 个片段需要合成')

    # ---- TTS 引擎选择逻辑 ----
    # 检查是否配置了火山引擎（需要 APPID 和 ACCESS_TOKEN）
    has_bytedance_config = bool(get_config('BYTEDANCE_APPID') and get_config('BYTEDANCE_ACCESS_TOKEN'))

    # 决策树：
    #   强制火山引擎 → 必须已配置，否则报错
    #   单说话人 + 已配置火山引擎 → 使用火山引擎（质量更高）
    #   IndexTTS 可用 → 使用 IndexTTS（支持声音克隆）
    #   两者皆不可用 → 报错
    if force_bytedance:
        if not has_bytedance_config:
            raise RuntimeError('强制使用火山引擎 TTS 但未配置。请在设置中配置 BYTEDANCE_APPID 和 BYTEDANCE_ACCESS_TOKEN')
        use_bytedance = True
    elif num_speakers == 1 and has_bytedance_config:
        use_bytedance = True
        logger.info('单说话人场景，使用火山引擎 TTS')
    elif INDEXTTS_AVAILABLE:
        use_bytedance = False
        logger.info('使用 IndexTTS 声音克隆')
    else:
        if has_bytedance_config:
            # IndexTTS 不可用时回退到火山引擎
            use_bytedance = True
            logger.warning('IndexTTS 未安装，将使用火山引擎 TTS。可在 WebUI「模型管理」中一键安装')
        else:
            # 两种引擎都不可用
            raise RuntimeError(
                '所有 TTS 引擎均不可用。请至少配置一种 TTS 方式：\n'
                '1. 配置火山引擎 TTS (BYTEDANCE_APPID + BYTEDANCE_ACCESS_TOKEN)\n'
                '2. 在 WebUI 设置页「模型管理」中下载 IndexTTS 模型（自动安装库+模型）'
            )

    # ---- 检查并生成说话人参考音频 ----
    speaker_folder = os.path.join(folder, 'SPEAKER')

    speakers = set()
    for line in transcript:
        speakers.add(line['speaker'])

    # 检查每个说话人是否都有对应的参考音频文件
    speaker_files_exist = True
    if not os.path.exists(speaker_folder):
        speaker_files_exist = False
    else:
        for speaker in speakers:
            speaker_wav = os.path.join(speaker_folder, f'{speaker}.wav')
            if not os.path.exists(speaker_wav):
                speaker_files_exist = False
                break

    # 如果有任何说话人缺失参考音频，重新生成
    if not speaker_files_exist:
        logger.info(f'说话人音频文件不完整，重新生成: {speaker_folder}')
        generate_speaker_audio(folder, transcript)

    # IndexTTS 不使用 ref_text 参数，无需构建参考文本映射

    # ---- 逐段合成语音 ----
    wav_segments = []         # 存储各片段合成的音频数组
    cumulative_samples = 0    # 累积采样点数，用于计算当前时间位置
    total_segments = len(transcript)

    for i, line in enumerate(transcript):
        speaker = line['speaker']
        # 对翻译文本进行预处理
        text = preprocess_text(line['translation'])
        # TTS 输出路径：wavs/0000.wav, wavs/0001.wav, ...
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        # 该说话人的参考音频路径
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        logger.info(f"[{i+1}/{total_segments}] 正在合成 ({speaker}): {text[:50]}")

        # 根据选择的 TTS 引擎调用对应接口
        if use_bytedance:
            # 单说话人场景使用 BV701_streaming 音色（高质量通用音色），多说话人场景由声纹匹配自动选择
            voice_type = 'BV701_streaming' if num_speakers == 1 else None
            bytedance_tts(text, output_path, speaker_wav, voice_type=voice_type)
        else:
            indextts_tts(text, output_path, speaker_wav)

        # 检查 TTS 是否成功生成文件
        if not os.path.exists(output_path):
            raise RuntimeError(f'TTS 生成失败: {output_path}，请检查 TTS 配置')

        # ---- 时长对齐处理 ----
        # 原始片段的起止时间（基于原视频时间线）
        start = line['start']
        end = line['end']
        length = end - start  # 片段原始时长

        # 计算当前累积位置（已合成的音频总时长）
        last_end = cumulative_samples / 24000

        # 如果当前片段起始时间晚于累积位置，插入静音补齐间隙
        if start > last_end:
            silence = np.zeros((int((start - last_end) * 24000), ))
            wav_segments.append(silence)
            cumulative_samples += len(silence)

        # 更新当前片段的起始时间为实际时间线位置
        start = cumulative_samples / 24000
        line['start'] = start

        # 如果不是最后一段，限制结束时间不超过下一段的开始时间（避免重叠）
        if i < len(transcript) - 1:
            next_line = transcript[i + 1]
            next_end = next_line['end']
            end = min(start + length, next_end)

        # 用速度拉伸调整音频长度至目标时长
        wav, length = adjust_audio_length(output_path, end - start)
        # 如果拉伸后仍不够长，在静音间隙中分配额外时长
        wav = distribute_extra_silence(wav, 24000, end - start)
        # 计算实际调整后的时长
        length = len(wav) / 24000

        # 添加到拼接列表
        wav_segments.append(wav)
        cumulative_samples += len(wav)
        # 更新片段的结束时间
        line['end'] = start + length

    # ---- 生成完整 TTS 音频 ----
    full_wav = np.concatenate(wav_segments) if wav_segments else np.zeros((0,))

    # ---- 音量归一化（以原始人声为参考） ----
    vocal_wav, sr = librosa.load(os.path.join(folder, 'audio_vocals.wav'), sr=24000)
    max_full = np.max(np.abs(full_wav)) if len(full_wav) > 0 else 0
    max_vocal = np.max(np.abs(vocal_wav)) if len(vocal_wav) > 0 else 0
    # 仅在两端都有有效音频时进行归一化，防止除以零
    if max_full > 1e-8 and max_vocal > 1e-8:
        full_wav = full_wav / max_full * max_vocal
    elif max_full < 1e-8:
        logger.warning("TTS 输出为静音，跳过归一化")

    # 保存纯 TTS 音频
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))

    # ---- 更新 translation.json（加入对齐后的时间戳） ----
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    # ---- 混合 TTS 音频与伴奏 ----
    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)

    # 将较短的一个补零至与较长的等长
    if len_full_wav > len_instruments_wav:
        # TTS 音频更长：在伴奏末尾补零
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 伴奏更长：在 TTS 音频末尾补零
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')

    # 混合（直接相加，音量已在上一步对齐）
    combined_wav = full_wav + instruments_wav
    # 注：不进行整体归一化，保留原始动态范围
    # combined_wav /= np.max(np.abs(combined_wav))

    # 保存混合音频
    save_wav(combined_wav, os.path.join(folder, 'audio_combined.wav'))
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')


def generate_all_wavs_under_folder(root_folder, force_bytedance=False):
    """
    扫描指定目录下所有视频处理目录，逐个执行 TTS 合成。

    通过 os.walk 递归查找包含 translation.json 的子目录，跳过已合成
    （audio_combined.wav 已存在）的目录。

    Args:
        root_folder (str):         根目录路径（相对或绝对路径）
        force_bytedance (bool):    是否强制使用火山引擎 TTS

    Raises:
        FileNotFoundError: 未找到任何视频处理目录，或目录缺少 translation.json

    Returns:
        str: 提示信息，表示所有目录合成完成
    """
    # 如果是相对路径，则基于项目根目录拼接
    if not os.path.isabs(root_folder):
        root_folder = str(PROJECT_ROOT / root_folder)
    logger.info(f'开始语音合成扫描: {root_folder}')
    found_video_dir = False
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(root_folder):
        # 跳过 wavs 和 SPEAKER 子目录（避免重复处理已合成的片段目录）
        dirs[:] = [d for d in dirs if d not in ('wavs', 'SPEAKER')]
        # 只处理包含 translation.json 且尚未生成 audio_combined.wav 的目录
        if 'translation.json' not in files and 'audio_combined.wav' not in files:
            continue
        found_video_dir = True
        # 如果目录有 audio_combined.wav 但没有 translation.json，说明翻译步骤可能缺失
        if 'translation.json' not in files:
            raise FileNotFoundError(
                f'发现视频目录 {root} 但缺少 translation.json，请确认翻译步骤已正确执行。目录内容: {files}'
            )
        # 跳过已经合成完成的目录
        if 'audio_combined.wav' in files:
            logger.info(f'跳过已合成目录: {root}')
            continue
        logger.info(f'处理目录: {root}')
        # 执行 TTS 合成
        generate_wavs(root, force_bytedance)
    if not found_video_dir:
        raise FileNotFoundError(f'在 {root_folder} 下未找到任何视频处理目录')
    logger.info(f'语音合成完成: {root_folder}')
    return f'Generated all wavs under {root_folder}'


def generate_wavs_in_folders(folder_list, force_bytedance=False):
    """处理指定目录列表中的语音合成（批量入口）

    支持传入单个目录路径字符串或多个路径的列表。逐一调用 generate_wavs，
    分别统计成功和失败的个数。

    Args:
        folder_list (str|list): 需要处理的目录路径列表（或单个路径字符串）
        force_bytedance (bool):  是否强制使用火山引擎 TTS

    Returns:
        str: 包含成功/失败数量的提示信息
    """
    # 如果传入的是单个字符串，包装为列表
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    success_list = []
    fail_list = []
    for subdir in folder_list:
        subdir = os.path.abspath(subdir)
        # 检查目录中必要的文件
        files = os.listdir(subdir) if os.path.exists(subdir) else []
        if 'translation.json' not in files:
            fail_list.append(f"{subdir}: 缺少 translation.json")
            continue
        if 'audio_combined.wav' in files:
            logger.info(f'跳过已合成目录: {subdir}')
            success_list.append(subdir)
            continue
        try:
            generate_wavs(subdir, force_bytedance)
            success_list.append(subdir)
        except Exception as e:
            logger.error(f'Error generating wavs in {subdir}: {e}')
            fail_list.append(f"{subdir}: {e}")
    logger.info(f'语音合成完成: 成功 {len(success_list)}/{len(folder_list)}, 失败 {len(fail_list)}')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'


if __name__ == '__main__':
    # 命令行直接运行时的测试入口
    folder = r'videos\TED-Ed\20211214 Would you raise the bird that murdered your children？ - Steve Rothstein'
    generate_wavs(folder, force_bytedance=False)
