# -*- coding: utf-8 -*-
"""
step020_whisperx.py — 步骤2：语音识别与说话者分离模块

本模块是 YouDub 流水线的第三个步骤（流水线编号 02），负责：
1. 加载 WhisperX 模型对分离后的人声音频进行语音识别（ASR）
2. 对识别结果进行词级时间戳对齐（使用 Wav2Vec2 对齐模型）
3. 可选：使用说话者分离（Diarization）识别不同说话人
4. 合并短句为完整句子（基于标点符号）
5. 按说话人切割音频片段，保存为独立的说话人音频文件
6. 提供主动的 GPU 显存管理：加载/卸载模型、OOM 降级重试

输入：
    {folder}/audio_vocals.wav              # 步骤1分离的人声音频
    {folder}/download.info.json（间接引用）

输出：
    {folder}/transcript.json               # 带时间戳和说话人标记的转录结果
    {folder}/SPEAKER/{SPEAKER_XX}.wav      # 各说话人的独立音频片段

转录结果格式：
    [{"start": float, "end": float, "text": str, "speaker": str}, ...]

输出供步骤03（LLM 翻译）使用。
"""

import json          # JSON 读写，用于保存和加载转录结果
import os            # 文件和路径操作
import time          # 时间测量，用于日志记录
import threading     # 线程锁，保证模型加载的线程安全
from contextlib import contextmanager  # 上下文管理器装饰器，用于临时修改环境变量
import librosa       # 音频处理库，用于加载 WAV 文件
import numpy as np   # 数值计算，用于音频数据处理
import torch         # PyTorch，用于 CUDA 显存管理
import whisperx      # WhisperX 语音识别库（核心依赖）
from whisperx.diarize import DiarizationPipeline  # WhisperX 说话者分离管线
from loguru import logger                          # 结构化日志记录

from .utils import save_wav                         # 项目工具：保存 WAV 文件
from .config import (get_config,                    # 获取配置项
                     ensure_ffmpeg_available,        # 确保 FFmpeg 可用
                     get_ffmpeg_path,               # 获取 FFmpeg 路径
                     get_hf_local_files_only,       # 检查是否离线模式
                     PROJECT_ROOT,                  # 项目根目录
                     MODEL_ROOT)                    # 模型存储根目录


def _log_cuda_memory(logger_func=logger.info):
    """
    记录当前 CUDA 显存使用情况到日志。

    用于显存跟踪和调试，帮助定位 OOM 问题。
    在模型加载前后调用，对比显存变化。

    Args:
        logger_func: 日志函数，默认 logger.info，可用 logger.debug 减少输出
    """
    if not torch.cuda.is_available():
        return
    # 获取已分配的显存（GB）
    allocated = torch.cuda.memory_allocated() / 1024**3
    # 获取已预留的显存（GB）
    reserved = torch.cuda.memory_reserved() / 1024**3
    # 获取 GPU 总显存
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated
    logger_func(
        f'CUDA 显存: 已分配={allocated:.2f}GB, '
        f'已预留={reserved:.2f}GB, '
        f'空闲={free:.2f}GB, '
        f'总计={total:.2f}GB')


# 各模型的显存估算值（GB），用于加载前预检
_MODEL_VRAM_ESTIMATES = {
    'large-v3': 6.0, 'large-v2': 6.0, 'large': 6.0,
    'medium': 4.0, 'small': 2.0, 'base': 1.0, 'tiny': 0.5,
    'align': 1.5,   # 对齐模型（Wav2Vec2）估算
    'diarize': 1.5, # 说话者分离模型估算
}


def _get_free_vram_gb():
    """
    获取当前空闲 GPU 显存（GB）。

    无 GPU 时返回无穷大，确保不阻止 CPU 模式的流程。

    Returns:
        空闲显存大小（GB），无 GPU 时为 float('inf')
    """
    if not torch.cuda.is_available():
        return float('inf')
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total - allocated


def _get_default_batch_size():
    """
    根据 GPU 显存自动选择保守的批处理大小，预留安全余量避免 OOM。

    显存越大，可用的批处理大小越大，GPU 利用率越高。
    估算公式：实际峰值显存 = 模型 + 对齐 + 批处理数据，留 30% 余量。

    Returns:
        推荐的 batch_size 整数值
    """
    if not torch.cuda.is_available():
        return 1
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # 根据显存分档选择安全的批处理大小
    if total_vram_gb >= 24:
        return 16     # 24GB+ (如 RTX 3090/4090)
    elif total_vram_gb >= 16:
        return 8      # 16GB (如 RTX 4060 Ti)
    elif total_vram_gb >= 8:
        return 4      # 8GB (如 RTX 3070/4060)
    else:
        return 1      # 低显存或未知


def _check_vram_before_load(model_name, device='cuda'):
    """
    加载模型前主动检查显存是否充足。

    避免因显存不足导致进程被系统 OOM-kill。
    比较模型预估需求与当前空闲显存。

    Args:
        model_name: 模型名称（键名在 _MODEL_VRAM_ESTIMATES 中查找）
        device: 计算设备

    Returns:
        (ok: bool, message: str) 元组
        ok=True 表示显存足够继续加载
        message 在不足时包含详细建议消息
    """
    if device == 'cpu' or not torch.cuda.is_available():
        return True, None               # CPU 模式不需要检查
    estimated = _MODEL_VRAM_ESTIMATES.get(model_name, 4.0)  # 默认预估 4GB
    free = _get_free_vram_gb()
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # 如果空闲显存 < 预估需求的 70%，判定为不足
    if free < estimated * 0.7:
        return False, (
            f"GPU 显存不足以加载 {model_name} 模型（约需 {estimated:.1f}GB，"
            f"当前空闲 {free:.1f}GB / 总计 {total:.1f}GB）。"
            f"建议：选择更小的模型（如 medium/small/tiny）或切换为 CPU 模式。"
        )
    return True, None


def _is_oom_error(e):
    """
    判断异常是否由 CUDA/系统 OOM（显存不足）导致。

    通过检查异常消息中的关键词来识别 OOM 错误。

    Args:
        e: 捕获的异常对象

    Returns:
        True 表示 OOM 导致的异常
    """
    msg = str(e).lower()
    return ('out of memory' in msg or
            ('cuda' in msg and ('alloc' in msg or 'oom' in msg)))


@contextmanager
def _ffmpeg_in_path():
    """
    上下文管理器：临时将 FFmpeg 所在目录添加到系统 PATH 环境变量。

    WhisperX 在转录时会自动调用 FFmpeg 解码音频，
    需要确保 FFmpeg 在 PATH 中可找到。
    使用完后恢复原始 PATH。

    Yields:
        None（仅在 with 块中使用）
    """
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        yield
        return
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    current_path = os.environ.get('PATH', '')
    # 如果 FFmpeg 已在 PATH 中，无需操作
    if ffmpeg_dir in current_path:
        yield
        return
    # 备份并修改 PATH
    original_path = current_path
    os.environ['PATH'] = ffmpeg_dir + os.pathsep + original_path
    try:
        yield
    finally:
        os.environ['PATH'] = original_path


# 全局模型实例和锁（单例模式，避免重复加载）
whisper_model = None      # WhisperX 语音识别模型
diarize_model = None      # 说话者分离模型

align_model = None        # 词级对齐模型
language_code = None      # 当前对齐模型对应的语言代码
align_metadata = None     # 对齐模型的元数据

_model_lock = threading.Lock()  # 全局模型锁，保护多线程环境下的模型加载/卸载


def init_whisperx():
    """
    初始化 WhisperX（占位函数）。

    保留此函数供外部兼容调用，实际加载延迟到第一次转录时。
    """
    pass


def load_whisper_model(model_name: str = 'large-v3', download_root=None, device='auto'):
    """
    加载 WhisperX 语音识别模型。

    模型会缓存到本地，第二次加载更快。
    线程安全：使用全局锁保护模型加载过程。

    Args:
        model_name: 模型名称，如 'large-v3'（推荐）、'medium'、'small'、'base'、'tiny'
        download_root: 模型下载/缓存目录，None 则使用默认 {MODEL_ROOT}/whisper
        device: 计算设备，'auto' 自动选择
    """
    # 'large' 是 'large-v3' 的别名
    if model_name == 'large':
        model_name = 'large-v3'
    if download_root is None:
        download_root = str(MODEL_ROOT / 'whisper')

    global whisper_model
    if whisper_model is not None:
        return  # 模型已加载，跳过

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 检查是否离线模式（仅从本地缓存加载）
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
        # 加载失败时清理可能已部分加载的资源
        cleanup_whisperx()
        raise
    t_end = time.time()
    logger.info(f'Loaded WhisperX model: {model_name} in {t_end - t_start:.2f}s')
    _log_cuda_memory()


def load_align_model(language='en', device='auto'):
    """
    加载词级时间戳对齐模型（Wav2Vec2）。

    对 Whisper 的识别结果进行更精确的词级时间对齐。
    按语言加载不同的对齐模型。

    Args:
        language: 语言代码，如 'en'、'zh'、'ja' 等
        device: 计算设备
    """
    global align_model, language_code, align_metadata

    # 如果已加载同语言的模型，跳过
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
        # 临时设置环境变量以控制 HuggingFace Hub 的行为
        _prev_hub_offline = os.environ.get('HF_HUB_OFFLINE', None)
        if local_files_only:
            os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线模式

        _prev_hf_home = os.environ.get('HF_HOME', None)
        os.environ['HF_HOME'] = str(MODEL_ROOT / "huggingface")  # 指定 HuggingFace 缓存目录

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device
        )
    except Exception:
        cleanup_whisperx()
        raise
    finally:
        # 恢复环境变量
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
    """
    加载说话者分离模型（Diarization）。

    使用 PyAnnote Audio 进行说话者日志分离，
    识别"谁在什么时候说话"。

    Args:
        device: 计算设备
    """
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
        # 临时设置环境变量
        _prev_hub_offline = os.environ.get('HF_HUB_OFFLINE', None)
        if local_files_only:
            os.environ['HF_HUB_OFFLINE'] = '1'

        _prev_hf_home = os.environ.get('HF_HOME', None)
        os.environ['HF_HOME'] = str(MODEL_ROOT / "huggingface")

        # 创建说话者分离管线，需要 HuggingFace Token 以访问模型
        diarize_model = DiarizationPipeline(
            token=get_config('HF_TOKEN'),
            device=device
        )
    except Exception:
        cleanup_whisperx()
        raise
    finally:
        # 恢复环境变量
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
    """
    合并相邻的短句段为完整的句子。

    WhisperX 可能输出过短的片段，本函数根据结尾标点判断
    是否将当前段与下一段合并，使句子更完整。

    Args:
        transcript: 转录片段列表，每个片段有 'text'、'start'、'end' 字段
        ending: 表示句子结束的标点符号集合

    Returns:
        合并后的转录列表
    """
    merged_transcription = []   # 合并后的结果列表
    buffer_segment = None       # 缓存当前正在累积的句子

    for segment in transcript:
        if buffer_segment is None:
            # 开始新句子
            buffer_segment = segment
        else:
            # 如果缓存的段以句尾标点结束，则提交并开始新段
            if buffer_segment['text'][-1] in ending:
                merged_transcription.append(buffer_segment)
                buffer_segment = segment
            else:
                # 否则将当前段合并到缓存段中
                buffer_segment['text'] += ' ' + segment['text']
                buffer_segment['end'] = segment['end']

    # 提交最后一个缓存段
    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription


def transcribe_audio(folder, model_name: str = 'large', download_root='models/whisper',
                     device='auto', batch_size=None, diarization=True,
                     min_speakers=None, max_speakers=None):
    """
    主入口：对指定目录中的人声音频执行完整的语音识别流程。

    完整流程：
    1. 加载 Whisper 模型进行初始转录
    2. 立即卸载 Whisper 模型释放显存
    3. 加载对齐模型进行词级时间戳对齐
    4. 卸载对齐模型释放显存
    5. 可选：加载说话者分离模型识别说话人
    6. 保存转录结果为 transcript.json
    7. 生成各说话人的独立音频文件

    Args:
        folder: 视频处理目录（包含 audio_vocals.wav）
        model_name: Whisper 模型名称
        download_root: 模型下载目录
        device: 计算设备
        batch_size: 批处理大小，None 时根据显存自动选择
        diarization: 是否启用说话者分离
        min_speakers: 最小说话人数（可选，提高分离精度）
        max_speakers: 最大说话人数（可选，提高分离精度）

    Returns:
        True 表示转录成功

    Raises:
        FileNotFoundError: 人声音频文件不存在时抛出
        RuntimeError: 转录过程中出现异常（OOM 时会自动降级重试）
    """
    if batch_size is None:
        batch_size = _get_default_batch_size()

    # 检查转录结果是否已存在（跳过已处理目录）
    if os.path.exists(os.path.join(folder, 'transcript.json')):
        logger.info(f'Transcript already exists in {folder}')
        return True

    # 检查人声音频文件
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f'音频文件不存在: {wav_path}，请确认音频分离步骤已正确执行')

    # 检查 FFmpeg 可用性（WhisperX 需要 FFmpeg 解码音频）
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
        # 第一步：加载 Whisper 模型并执行转录
        load_whisper_model(model_name, download_root, device)

        with _ffmpeg_in_path():
            rec_result = whisper_model.transcribe(wav_path, batch_size=batch_size)

        # 转录完成后立即卸载 Whisper 模型（~6GB），为对齐步骤释放显存
        try:
            _unload_whisper_model()
        except Exception as unload_err:
            logger.warning(f"卸载 Whisper 模型时出错（非致命）: {unload_err}")

        # 检查是否检测到语言（'nn' 表示未检测到）
        if rec_result['language'] == 'nn':
            logger.warning(f'No language detected in {wav_path}')
            return False

        # 第二步：加载对齐模型进行词级时间戳对齐
        ok, err_msg = _check_vram_before_load('align', device)
        if not ok:
            logger.warning(err_msg)

        load_align_model(rec_result['language'])
        with _ffmpeg_in_path():
            rec_result = whisperx.align(
                rec_result['segments'], align_model, align_metadata,
                wav_path, device, return_char_alignments=False
            )

        # 对齐完成后卸载对齐模型（~1.5GB），为说话者分离释放显存
        try:
            _unload_align_model()
        except Exception as unload_err:
            logger.warning(f"卸载对齐模型时出错（非致命）: {unload_err}")

        # 第三步（可选）：加载说话者分离模型
        if diarization:
            ok, err_msg = _check_vram_before_load('diarize', device)
            if not ok:
                logger.warning(err_msg)

            load_diarize_model(device)
            with _ffmpeg_in_path():
                # 执行说话者分离，获取每个时间段的说话人标签
                diarize_segments = diarize_model(
                    wav_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                # 将说话人标签分配到转录结果的每个词/段上
                rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)

        # 格式化转录结果：提取所需字段，为每段添加说话人标记
        transcript = [
            {
                'start': segement['start'],
                'end': segement['end'],
                'text': segement['text'].strip(),
                'speaker': segement.get('speaker', 'SPEAKER_00')  # 默认说话人
            }
            for segement in rec_result['segments']
        ]

        # 合并短句为完整句子
        transcript = merge_segments(transcript)

        # 保存转录结果到 JSON 文件
        with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=4, ensure_ascii=False)
        logger.info(
            f'Transcribed {wav_path} successfully, '
            f'and saved to {os.path.join(folder, "transcript.json")}')

        # 生成各说话人的独立音频文件
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
            cleanup_whisperx()  # 清理现有模型再重试
            return transcribe_audio(
                folder, model_name, download_root, device,
                new_batch_size, diarization, min_speakers, max_speakers
            )
        cleanup_whisperx()
        raise


def generate_speaker_audio(folder, transcript):
    """
    根据转录结果中的说话人标记，将人声音频按说话人切分为独立文件。

    每个说话人的所有音频段被拼接为一个 WAV 文件，
    存储在 {folder}/SPEAKER/{speaker}.wav。
    这些文件可用于说话人相关的 TTS 处理（保留声纹特征）。

    Args:
        folder: 视频处理目录
        transcript: 转录结果列表，每项包含 start、end、speaker 字段
    """
    # 加载人声音频（librosa 加载时自动重采样到 24kHz）
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    audio_data, samplerate = librosa.load(wav_path, sr=24000)

    # 按说话人分组存储音频段
    speaker_segments = {}
    length = len(audio_data)
    delay = 0.05  # 每段前后各加 50ms 的缓冲，避免切音突变

    for segment in transcript:
        # 将时间戳转换为采样点索引，加上缓冲
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end'] + delay) * samplerate), length)
        speaker = segment['speaker']

        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(audio_data[start:end])

    # 创建说话人音频输出目录
    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)

    # 保存每个说话人的音频文件
    for speaker, segments in speaker_segments.items():
        speaker_file_path = os.path.join(speaker_folder, f"{speaker}.wav")
        if segments:
            # 拼接所有音频段并保存
            save_wav(np.concatenate(segments), speaker_file_path)
        else:
            logger.warning(f"说话人 {speaker} 无有效音频段，跳过保存")


def transcribe_all_audio_under_folder(folder, model_name: str = 'large',
                                       download_root='models/ASR/whisper',
                                       device='auto', batch_size=None,
                                       diarization=True, min_speakers=None,
                                       max_speakers=None):
    """
    递归扫描目录，对所有包含人声音频的目录执行语音识别。

    用于批量处理场景：自动遍历目录树，找到所有需要转录的视频目录。

    Args:
        folder: 根目录路径（相对或绝对）
        model_name: Whisper 模型名称
        download_root: 模型下载目录
        device: 计算设备
        batch_size: 批处理大小
        diarization: 是否启用说话者分离
        min_speakers: 最小说话人数
        max_speakers: 最大说话人数

    Returns:
        执行结果摘要字符串

    Raises:
        FileNotFoundError: 未找到任何视频处理目录时抛出
    """
    if batch_size is None:
        batch_size = _get_default_batch_size()

    # 相对路径转为绝对路径
    if not os.path.isabs(folder):
        folder = str(PROJECT_ROOT / folder)
    if not os.path.isabs(download_root):
        download_root = str(PROJECT_ROOT / download_root)

    found_video_dir = False
    success_list = []
    fail_list = []

    try:
        # 递归遍历所有子目录
        for root, dirs, files in os.walk(folder):
            # 跳过不相关的目录（需要至少有视频或音频文件）
            if ('download.mp4' not in files and 'audio.wav' not in files
                    and 'audio_vocals.wav' not in files and 'transcript.json' not in files):
                continue

            found_video_dir = True

            # 检查人声音频是否存在
            if 'audio_vocals.wav' not in files:
                fail_list.append(f"{root}: 缺少 audio_vocals.wav")
                logger.error(f"目录 {root} 转录失败: 缺少 audio_vocals.wav")
                continue

            # 跳过已转录的目录
            if 'transcript.json' in files:
                success_list.append(root)
                continue

            # 执行转录
            try:
                transcribe_audio(root, model_name, download_root, device,
                                 batch_size, diarization, min_speakers, max_speakers)
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
        cleanup_whisperx()  # 无论成功还是失败，最后清理模型


def _unload_whisper_model():
    """
    转录完成后卸载 Whisper 模型，为后续步骤释放 GPU 显存。

    Whisper large-v3 约占用 6GB 显存，转录完成后立即卸载
    可以为对齐模型和说话者分离模型腾出空间。
    """
    global whisper_model
    import gc
    import torch

    with _model_lock:
        if whisper_model is not None:
            del whisper_model
            whisper_model = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()    # 确保所有 CUDA 操作完成
        torch.cuda.empty_cache()    # 清空 CUDA 缓存
    logger.info('Whisper 模型已卸载，显存已释放给后续步骤')
    _log_cuda_memory()


def _unload_align_model():
    """
    对齐完成后卸载对齐模型，为说话者分离释放显存。

    对齐模型（Wav2Vec2）约占用 1.5GB 显存。
    """
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
    """
    清理所有 WhisperX 相关模型（Whisper、对齐、说话者分离），释放显存。

    在批量处理结束后或发生异常时调用，确保 GPU 显存被完全释放。
    """
    global whisper_model, align_model, language_code, align_metadata, diarize_model
    import gc
    import torch

    with _model_lock:
        # 卸载 Whisper 模型
        if whisper_model is not None:
            del whisper_model
            whisper_model = None

        # 卸载对齐模型
        if align_model is not None:
            del align_model
            align_model = None
            language_code = None
            align_metadata = None

        # 卸载说话者分离模型
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


def transcribe_audio_in_folders(folder_list, model_name='large',
                                 download_root='models/ASR/whisper',
                                 device='auto', batch_size=None,
                                 diarization=True, min_speakers=None,
                                 max_speakers=None):
    """
    处理指定目录列表中的语音识别。

    与 transcribe_all_audio_under_folder 的区别：
    接收明确的目录列表而非根目录，不递归遍历。

    Args:
        folder_list: 需要处理的目录路径列表（单个字符串或列表）
        model_name: Whisper 模型名称
        download_root: 模型下载目录
        device: 计算设备
        batch_size: 批处理大小，None 时根据显存自动选择
        diarization: 是否启用说话者分离
        min_speakers: 最小说话人数
        max_speakers: 最大说话人数

    Returns:
        格式化的执行结果字符串
    """
    if batch_size is None:
        batch_size = _get_default_batch_size()

    # 统一为列表格式
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    if not os.path.isabs(download_root):
        download_root = str(PROJECT_ROOT / download_root)

    success_list = []
    fail_list = []

    try:
        for subdir in folder_list:
            subdir = os.path.abspath(subdir)

            # 检查人声音频是否存在
            if 'audio_vocals.wav' not in os.listdir(subdir):
                fail_list.append(f"{subdir}: 缺少 audio_vocals.wav")
                continue

            # 跳过已转录的目录
            if 'transcript.json' in os.listdir(subdir):
                logger.info(f'Transcript already exists in {subdir}')
                success_list.append(subdir)
                continue

            # 执行转录
            try:
                transcribe_audio(subdir, model_name, download_root, device,
                                 batch_size, diarization, min_speakers, max_speakers)
                success_list.append(subdir)
            except Exception as e:
                logger.error(f'Error transcribing audio in {subdir}: {e}')
                fail_list.append(f"{subdir}: {e}")

        logger.info(
            f'语音识别完成: 成功 {len(success_list)}/{len(folder_list)}, '
            f'失败 {len(fail_list)}')
        return f'成功: {len(success_list)}\n失败: {len(fail_list)}'
    finally:
        cleanup_whisperx()


if __name__ == '__main__':
    # 模块直接运行时的示例用法
    transcribe_all_audio_under_folder('videos')
