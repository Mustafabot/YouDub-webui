# -*- coding: utf-8 -*-
"""
step010_demucs_vr.py — 步骤1：人声与伴奏分离模块

本模块是 YouDub 流水线的第二个步骤（流水线编号 01），负责：
1. 从步骤0输出的视频中提取音频（WAV 格式，使用 FFmpeg）
2. 使用 Demucs（HTDemucs）模型将音频分离为人声（vocals）和伴奏（instruments）
3. 支持超长音频的分块处理（含交叉淡入淡出拼接），避免内存溢出
4. 提供批量扫描目录进行音频提取和分离的功能

输入：
    {folder}/download.mp4              # 步骤0下载的视频文件
    {folder}/audio.wav                  # 从视频中提取的音频（步骤1生成）

输出：
    {folder}/audio_vocals.wav           # 分离后的人声音频（44.1kHz, 16bit PCM）
    {folder}/audio_instruments.wav      # 分离后的伴奏音频（44.1kHz, 16bit PCM）

输出供步骤02（WhisperX 语音识别）使用。
"""

import shutil          # 文件和目录操作（当前模块未直接使用，保留导入）
import threading       # 线程锁，保证 Demucs 模型的线程安全访问
from demucs.api import Separator   # Demucs 音频分离 API（核心依赖）
import os              # 文件和路径操作
import subprocess      # 执行外部命令（FFmpeg 音频提取和截取）
from loguru import logger          # 结构化日志记录
import time            # 时间测量（模型加载耗时、分离耗时）
import gc              # Python 垃圾回收，及时释放内存
import numpy as np     # 数值计算库，用于音频数据拼接和交叉淡入淡出
from .utils import save_wav, normalize_wav  # 项目工具：保存 WAV、归一化音频
# 配置模块：FFmpeg 可用性检查、获取 FFmpeg 路径、项目根目录、模型根目录
from .config import ensure_ffmpeg_available, get_ffmpeg_path, PROJECT_ROOT, MODEL_ROOT
import torch           # PyTorch，用于 CUDA 显存管理和设备选择
import wave            # 标准库 wave 模块，用于获取 WAV 文件时长

# 自动选择计算设备：优先使用 CUDA GPU，无 GPU 则使用 CPU
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
separator = None              # Demucs Separator 全局单例，避免重复加载模型
_separator_lock = threading.RLock()  # 可重入锁，保护 separator 的线程安全访问

# 默认配置常量
_DEFAULT_MAX_CHUNK_SECONDS = 600   # 外部分块最大时长：600 秒（10 分钟），超过则分块处理
_DEFAULT_SEGMENT = 10              # Demucs 模型内部分段推理秒数（控制显存占用）
_OVERLAP_SECONDS = 10              # 分块间的重叠秒数，用于交叉淡入淡出


def init_demucs(model_name: str = "htdemucs_ft", device: str = 'auto',
                progress: bool = True, shifts: int = 5, segment: int = _DEFAULT_SEGMENT):
    """
    初始化 Demucs 模型（占位函数，实际加载在 load_model 中完成）。

    保留此函数用于兼容旧代码的初始化调用。

    Args:
        model_name: Demucs 预训练模型名称，默认 "htdemucs_ft"（微调版）
        device: 计算设备，'auto' 自动选择 CPU/CUDA
        progress: 是否显示分离进度条
        shifts: 输入偏移预测次数，越多精度越高但越慢
        segment: 模型内部分段推理秒数
    """
    pass  # 实际加载延迟到第一次调用 separate_audio 时


def cleanup_demucs():
    """
    清理 Demucs 模型，释放 GPU 显存。

    删除全局 separator 对象，触发垃圾回收，并清空 CUDA 缓存。
    在批量处理结束后调用，防止显存泄漏。
    """
    global separator
    with _separator_lock:
        if separator is not None:
            del separator       # 删除模型对象
            separator = None
            gc.collect()        # 强制 Python 垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清空 CUDA 缓存
    logger.info('Demucs 模型已清理，显存已释放')


def load_model(model_name: str = "htdemucs_ft", device: str = 'auto',
               progress: bool = True, shifts: int = 5, segment: int = _DEFAULT_SEGMENT) -> Separator:
    """
    加载 Demucs 分离模型。

    使用 torch.hub 从本地或网络加载 Demucs 模型。
    模型缓存目录为 {MODEL_ROOT}/demucs。
    线程安全：使用全局锁保护模型加载过程。

    Args:
        model_name: 模型名称，如 "htdemucs_ft"（推荐）、"htdemucs"
        device: 计算设备，'auto' 时自动选择
        progress: 是否显示下载和加载进度条
        shifts: 输入偏移次数，增加可减少伪影
        segment: 内部分段长度（秒），小值减少显存占用

    Returns:
        加载好的 Separator 实例
    """
    global separator
    with _separator_lock:
        logger.info(f'Loading Demucs model: {model_name} (device={device}, shifts={shifts}, segment={segment}s)')
        t_start = time.time()
        # 确定实际使用的设备
        effective_device = auto_device if device == 'auto' else device
        # 设置模型下载/缓存目录
        torch.hub.set_dir(str(MODEL_ROOT / "demucs"))
        # 创建 Demucs Separator 实例（首次加载会自动下载模型权重）
        separator = Separator(model_name, device=effective_device,
                              progress=progress, shifts=shifts, segment=segment)
        # 禁用所有子模型的 train_segment 标志，确保推理模式
        for sub in separator.model.models:
            sub.use_train_segment = False
        t_end = time.time()
        logger.info(f'Demucs model loaded in {t_end - t_start:.2f} seconds')


def _get_audio_duration(audio_path: str) -> float:
    """
    使用标准库 wave 模块获取 WAV 音频文件的时长（秒）。

    比调用 FFmpeg 更轻量，适合快速获取时长信息。

    Args:
        audio_path: WAV 文件路径

    Returns:
        音频时长（秒）
    """
    with wave.open(audio_path, 'rb') as wf:
        # 帧数 / 帧率 = 时长（秒）
        return wf.getnframes() / wf.getframerate()


def _extract_audio_chunk(audio_path: str, output_path: str,
                         start_seconds: float, duration_seconds: float) -> str:
    """
    使用 FFmpeg 从音频文件中提取指定时间段的分块。

    用于超长音频的分块处理：从完整音频中截取一段区间。

    Args:
        audio_path: 原始音频文件路径
        output_path: 输出分块文件路径
        start_seconds: 起始时间（秒）
        duration_seconds: 截取时长（秒）

    Returns:
        输出分块文件路径

    Raises:
        RuntimeError: FFmpeg 执行失败时抛出
    """
    ffmpeg_path = get_ffmpeg_path()
    # 构建 FFmpeg 命令：无损截取 PCM 16bit 44.1kHz 双声道 WAV
    cmd = [
        str(ffmpeg_path),
        '-loglevel', 'error',             # 仅输出错误信息
        '-ss', str(start_seconds),         # 起始时间偏移
        '-i', str(audio_path),             # 输入文件
        '-t', str(duration_seconds),       # 截取时长
        '-vn',                             # 不处理视频流
        '-acodec', 'pcm_s16le',            # PCM 16bit 小端编码
        '-ar', '44100',                    # 采样率 44.1kHz
        '-ac', '2',                        # 双声道
        str(output_path)
    ]
    logger.debug(f'提取分块: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else '未知错误'
        raise RuntimeError(f'FFmpeg 分块提取失败: {error_msg}')
    return output_path


def _apply_crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap_samples: int) -> np.ndarray:
    """
    对两个音频块应用交叉淡入淡出（crossfade）拼接。

    解决分块处理时块边界处的音频不连续问题：
    前一块的尾部渐弱（fade-out），后一块的头部渐强（fade-in），
    两者在重叠区域叠加。

    Args:
        chunk1: 前一个音频块的数据（numpy 数组）
        chunk2: 后一个音频块的数据（numpy 数组）
        overlap_samples: 重叠区域的采样点数（决定了淡入淡出的长度）

    Returns:
        拼接后的完整音频数据数组
    """
    # 处理边界情况：无重叠或空块
    if overlap_samples <= 0 or chunk1 is None:
        if chunk1 is None:
            return chunk2
        if chunk2 is None:
            return chunk1
        return np.concatenate([chunk1, chunk2])

    # 生成淡出曲线（从 1.0 线性下降到 0.0）
    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    # 生成淡入曲线（从 0.0 线性上升到 1.0）
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

    # 如果是多声道音频（如立体声），将淡入淡出曲线扩展为二维
    if chunk1.ndim == 2:
        fade_out = fade_out[:, np.newaxis]  # shape: (overlap_samples, 1)
        fade_in = fade_in[:, np.newaxis]

    # 前一块的尾部重叠区域
    tail = chunk1[-overlap_samples:]
    # 后一块的头部重叠区域
    head = chunk2[:overlap_samples:]
    # 重叠区域叠加（淡出 + 淡入）
    crossfaded = tail.astype(np.float32) * fade_out + head.astype(np.float32) * fade_in

    # 构建最终结果数组
    result_shape = list(chunk1.shape)
    result_shape[0] = len(chunk1) - overlap_samples + len(chunk2)
    result = np.empty(result_shape, dtype=np.float32)

    # 拼接三部分：前块的非重叠部分 + 交叉淡入淡出部分 + 后块的非重叠部分
    result[:len(chunk1) - overlap_samples] = chunk1[:-overlap_samples]
    result[len(chunk1) - overlap_samples:len(chunk1)] = crossfaded
    result[len(chunk1):] = chunk2[overlap_samples:]

    return result


def _merge_instruments(separated: dict):
    """
    合并 Demucs 分离结果中除人声外的所有乐器音轨。

    Demucs htdemucs_ft 输出 4 个音轨: drums, bass, other, vocals。
    本函数将 drums + bass + other 合并为单一的伴奏音轨。

    Args:
        separated: Demucs 分离结果字典，键为音轨名，值为 torch Tensor

    Returns:
        (vocals_np, instruments_np) 元组
        vocals_np: 人声数据（numpy 数组，shape=[采样数, 声道数]）
        instruments_np: 合并后的伴奏数据
    """
    # 提取人声并转置为 (采样数, 声道数) 格式
    vocals = separated['vocals'].numpy().T
    del separated['vocals']  # 及时删除引用

    # 获取非人声音轨的键名列表
    instrument_keys = [k for k in separated.keys() if k != 'vocals']
    # 将所有乐器音轨叠加为单个伴奏音轨
    instruments = None
    for k in instrument_keys:
        v = separated[k]
        if instruments is None:
            instruments = v.clone()  # 第一个音轨直接克隆
        else:
            instruments.add_(v)      # 后续音轨累加
        del separated[k]             # 释放原始引用

    # 将伴奏 Tensor 转为 numpy 并转置
    instruments_np = instruments.numpy().T if instruments is not None else np.zeros_like(vocals)
    del instruments  # 释放临时变量

    return vocals, instruments_np


def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto',
                   progress: bool = True, shifts: int = 5, segment: int = _DEFAULT_SEGMENT,
                   max_chunk_seconds: int = _DEFAULT_MAX_CHUNK_SECONDS) -> None:
    """
    主入口：对指定目录中的音频执行人声/伴奏分离。

    流程：
    1. 检查输入音频文件和输出文件状态
    2. 获取音频时长
    3. 根据时长选择直接分离或分块分离

    Args:
        folder: 视频处理目录（包含 audio.wav）
        model_name: Demucs 模型名称
        device: 计算设备
        progress: 显示进度条
        shifts: 输入偏移次数
        segment: 内部分段长度
        max_chunk_seconds: 外部分块最大秒数，超过此值启用分块模式

    Raises:
        FileNotFoundError: audio.wav 不存在时抛出
    """
    global separator
    # 线程安全地检查/加载模型
    with _separator_lock:
        if separator is None:
            load_model(model_name, device, progress, shifts, segment)

    # 检查输入音频文件是否存在
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f'音频文件不存在: {audio_path}，请确认步骤01已正确执行')

    # 检查输出文件是否已存在（跳过已处理的目录）
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'Audio already separated in {folder}')
        return

    logger.info(f'Separating audio from {folder}')
    logger.info(
        f'注意: {model_name} 模型将分离 4 个音轨 (drums, bass, other, vocals) '
        f'{shifts} 遍，每个音轨都会显示独立的进度条')

    # 获取音频时长并决定处理策略
    duration = _get_audio_duration(audio_path)
    logger.info(
        f'音频时长: {duration:.1f}秒, 最大分块: {max_chunk_seconds}秒, '
        f'内部段长: {segment}秒')

    if duration <= max_chunk_seconds:
        # 音频在最大分块限制内，直接整体处理
        logger.info(
            f'音频时长 {duration:.1f}秒 不超过最大分块 {max_chunk_seconds}秒，直接处理')
        _separate_audio_direct(folder, audio_path, model_name, device, progress, shifts, segment)
    else:
        # 超长音频，启用分块处理模式
        logger.info(
            f'音频时长 {duration:.1f}秒 超过最大分块 {max_chunk_seconds}秒，'
            f'启动分块处理模式 (每块约{max_chunk_seconds}秒, 重叠{_OVERLAP_SECONDS}秒)')
        _separate_audio_chunked(folder, audio_path, duration,
                                model_name, device, progress, shifts, segment,
                                max_chunk_seconds)


def _separate_audio_direct(folder: str, audio_path: str, model_name: str,
                           device: str, progress: bool, shifts: int, segment: int) -> None:
    """
    直接对完整音频执行 Demucs 分离（不分块）。

    适用于时长在 max_chunk_seconds 以内的音频。

    Args:
        folder: 输出目录
        audio_path: 音频文件路径
        model_name: 模型名称
        device: 计算设备
        progress: 显示进度条
        shifts: 输入偏移次数
        segment: 内部分段长度
    """
    t_start = time.time()
    try:
        # 调用 Demucs 的 API 执行分离
        origin, separated = separator.separate_audio_file(audio_path)
    except Exception as e:
        logger.error(f'Error separating audio: {e}')
        raise
    t_end = time.time()
    logger.info(
        f'Audio separated in {t_end - t_start:.2f} seconds, '
        f'separated tracks: {list(separated.keys())}')

    # 释放原始音频数据（不再需要）
    del origin
    gc.collect()

    # 保存分离结果到文件
    _save_separated_tracks(separated, folder)


def _save_separated_tracks(separated: dict, folder: str) -> None:
    """
    保存 Demucs 分离结果为人声和伴奏两个 WAV 文件。

    将人声和伴奏分别存为独立文件，伴奏为所有乐器音轨的叠加。

    Args:
        separated: Demucs 分离结果字典（音轨名 -> Tensor）
        folder: 输出目录
    """
    # 提取人声并转置为 NumPy 数组
    vocals = separated['vocals'].numpy().T
    del separated['vocals']
    gc.collect()

    # 将非人声音轨合并为伴奏
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

    # 保存人声文件
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    save_wav(vocals, vocal_output_path, sample_rate=44100)
    logger.info(f'Vocals saved to {vocal_output_path}')
    del vocals
    gc.collect()

    # 保存伴奏文件
    save_wav(instruments_np, instruments_output_path, sample_rate=44100)
    logger.info(f'Instruments saved to {instruments_output_path}')
    del instruments_np
    gc.collect()


def _separate_audio_chunked(folder: str, audio_path: str, duration: float,
                            model_name: str, device: str, progress: bool,
                            shifts: int, segment: int,
                            max_chunk_seconds: int) -> None:
    """
    对超长音频执行分块分离。

    将长音频分割为多个重叠的块分别用 Demucs 处理，
    然后用交叉淡入淡出拼接还原为完整的分离结果。

    Args:
        folder: 输出目录
        audio_path: 原始音频文件路径
        duration: 音频总时长（秒）
        model_name: Demucs 模型名称
        device: 计算设备
        progress: 显示进度条
        shifts: 输入偏移次数
        segment: 内部分段长度
        max_chunk_seconds: 每个分块的最大秒数
    """
    stride_seconds = max_chunk_seconds       # 步长 = 分块大小
    overlap_seconds = _OVERLAP_SECONDS       # 重叠秒数
    sample_rate = 44100                      # 采样率

    # 计算分块的起始时间点（均匀间隔）
    chunk_starts = list(range(0, int(duration), stride_seconds))
    # 如果最后一块太短（小于重叠长度），去掉它
    if len(chunk_starts) > 1 and (duration - chunk_starts[-1]) < overlap_seconds:
        chunk_starts.pop()

    num_chunks = len(chunk_starts)
    logger.info(
        f'将音频分为 {num_chunks} 个分块处理 '
        f'(步长 {stride_seconds}秒, 重叠 {overlap_seconds}秒)')

    # 初始化结果累积器
    vocals_result = None
    instruments_result = None
    tmp_files = []  # 临时分块文件列表，用于 finally 清理

    try:
        # 逐个处理每个分块
        for i, start_sec in enumerate(chunk_starts):
            # 计算当前块的起止时间（含重叠）
            if i == 0:
                chunk_start_sec = 0  # 第一块从头开始，无前向重叠
            else:
                chunk_start_sec = max(0, start_sec - overlap_seconds)  # 前向扩展

            # 核心区域结束时间（不含重叠）
            core_end_sec = min(start_sec + stride_seconds, duration)
            if i < num_chunks - 1:
                # 非最后一块：后向扩展
                chunk_end_sec = min(start_sec + stride_seconds + overlap_seconds, duration)
            else:
                chunk_end_sec = duration  # 最后一块直接到结尾

            chunk_duration_sec = chunk_end_sec - chunk_start_sec

            logger.info(
                f'处理分块 {i+1}/{num_chunks}: '
                f'源音频 {chunk_start_sec:.1f}s - {chunk_end_sec:.1f}s '
                f'({chunk_duration_sec:.1f}秒)')

            # 从源音频中提取当前分块
            tmp_path = os.path.join(folder, f'_tmp_chunk_{i}.wav')
            _extract_audio_chunk(audio_path, tmp_path, chunk_start_sec, chunk_duration_sec)
            tmp_files.append(tmp_path)

            # 对当前分块执行 Demucs 分离
            t_start = time.time()
            try:
                _, separated = separator.separate_audio_file(tmp_path)
            except Exception as e:
                logger.error(f'Error separating chunk {i+1}/{num_chunks}: {e}')
                raise
            t_end = time.time()
            logger.info(
                f'分块 {i+1}/{num_chunks} 分离完成, 耗时 {t_end - t_start:.2f}秒, '
                f'音轨: {list(separated.keys())}')

            # 合并当前分块的人声和伴奏
            chunk_vocals, chunk_instruments = _merge_instruments(separated)
            del separated
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 每块处理完后清理显存

            # 拼接到总结果（除第一块外，需要交叉淡入淡出）
            if i == 0:
                vocals_result = chunk_vocals
                instruments_result = chunk_instruments
            else:
                overlap_samples = int(overlap_seconds * sample_rate)
                actual_overlap = min(overlap_samples, len(vocals_result), len(chunk_vocals))
                logger.debug(
                    f'分块 {i+1}: 交叉淡出 {actual_overlap} 采样点 '
                    f'({actual_overlap/sample_rate:.1f}秒)')
                vocals_result = _apply_crossfade(vocals_result, chunk_vocals, actual_overlap)
                instruments_result = _apply_crossfade(instruments_result, chunk_instruments, actual_overlap)
                del chunk_vocals, chunk_instruments
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(
            f'分块处理完成，拼接后 vocals: {vocals_result.shape}, '
            f'instruments: {instruments_result.shape}')

        # 保存最终分离结果
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
        # 清理所有临时分块文件
        for tmp_path in tmp_files:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass  # 删除临时文件失败不阻塞主流程
        if tmp_files:
            logger.info(f'已清理 {len(tmp_files)} 个临时分块文件')


def extract_audio_from_video(folder: str) -> bool:
    """
    从视频文件中提取音频流为 WAV 格式。

    使用 FFmpeg 将 download.mp4 中的音频流提取为
    44.1kHz、16bit 双声道 PCM WAV 文件。

    Args:
        folder: 视频所在目录（应包含 download.mp4）

    Returns:
        True 表示提取成功（或已存在）

    Raises:
        RuntimeError: FFmpeg 不可用或提取失败时抛出
        FileNotFoundError: 视频文件不存在时抛出
    """
    # 检查 FFmpeg 可用性
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

    # 检查输入视频是否存在
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f'视频文件不存在: {video_path}，请确认下载步骤已正确执行')

    # 如果音频已存在则跳过
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'Audio already extracted in {folder}')
        return True

    logger.info(f'Extracting audio from {folder}')

    # 构建 FFmpeg 命令：提取音频流为 PCM WAV
    ffmpeg_path = get_ffmpeg_path()
    cmd = [
        str(ffmpeg_path),
        '-loglevel', 'error',    # 仅显示错误
        '-i', str(video_path),   # 输入视频文件
        '-vn',                   # 不处理视频流
        '-acodec', 'pcm_s16le',  # PCM 16bit 小端编码
        '-ar', '44100',          # 44.1kHz 采样率
        '-ac', '2',              # 双声道
        str(audio_path)
    ]

    logger.info(f'执行 FFmpeg 命令: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

    # 检查 FFmpeg 返回值
    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else '未知错误'
        raise RuntimeError(
            f'音频提取失败，FFmpeg 返回错误码 {result.returncode}: {error_msg}\n'
            f'命令: {" ".join(cmd)}'
        )

    # 验证输出文件
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


def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft",
                                     device: str = 'auto', progress: bool = True,
                                     shifts: int = 5, segment: int = _DEFAULT_SEGMENT,
                                     max_chunk_seconds: int = _DEFAULT_MAX_CHUNK_SECONDS) -> None:
    """
    递归扫描根目录，对所有包含 download.mp4 的视频目录执行音频提取和分离。

    用于批量处理场景：自动遍历目录树，找到所有视频目录并处理。

    Args:
        root_folder: 根目录路径（相对或绝对）
        model_name: Demucs 模型名称
        device: 计算设备
        progress: 显示进度条
        shifts: 输入偏移次数
        segment: 内部分段长度
        max_chunk_seconds: 外部分块最大秒数

    Returns:
        格式化的处理结果字符串

    Raises:
        FileNotFoundError: 未找到任何视频目录时抛出
    """
    # 相对路径转为绝对路径
    if not os.path.isabs(root_folder):
        root_folder = str(PROJECT_ROOT / root_folder)
    global separator
    found_video_dir = False
    processed_dirs = set()   # 记录已处理的目录，避免重复
    success_list = []        # 成功列表
    fail_list = []           # 失败列表

    logger.info(f'开始遍历目录: {root_folder}')
    try:
        # 递归遍历所有子目录
        for subdir, dirs, files in os.walk(root_folder):
            if subdir in processed_dirs:
                logger.debug(f'跳过已处理目录: {subdir}')
                continue

            video_path = os.path.join(subdir, 'download.mp4')
            if not os.path.exists(video_path):
                # 没有视频但有可能相关，记录失败信息
                if os.path.exists(os.path.join(subdir, 'download.info.json')):
                    fail_list.append(f"{subdir}: 缺少 download.mp4")
                continue

            found_video_dir = True
            logger.info(f'处理视频目录: {subdir}')

            try:
                # 第一步：从视频提取音频
                audio_path = os.path.join(subdir, 'audio.wav')
                if not os.path.exists(audio_path):
                    logger.info(f'提取音频: {subdir}')
                    extract_audio_from_video(subdir)

                # 第二步：分离人声和伴奏
                vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
                instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
                if not os.path.exists(vocal_output_path) or not os.path.exists(instruments_output_path):
                    logger.info(f'分离人声和伴奏: {subdir}')
                    separate_audio(subdir, model_name, device, progress,
                                   shifts, segment, max_chunk_seconds)
                else:
                    logger.info(f'音频已分离，跳过: {subdir}')

                success_list.append(subdir)
            except Exception as e:
                logger.error(f'Error separating audio in {subdir}: {e}')
                fail_list.append(f"{subdir}: {e}")

            processed_dirs.add(subdir)
            gc.collect()  # 每处理完一个目录主动回收内存

        logger.info(f'遍历完成，共处理 {len(processed_dirs)} 个视频目录')

        if not found_video_dir:
            raise FileNotFoundError(
                f'在 {root_folder} 下未找到任何包含 download.mp4 的视频目录')

        logger.info(
            f'音频分离完成: 成功 {len(success_list)}/{len(processed_dirs)}, '
            f'失败 {len(fail_list)}')
        return f'成功: {len(success_list)}\n失败: {len(fail_list)}'
    finally:
        # 无论成功与否，最后都要清理模型释放显存
        cleanup_demucs()


def separate_audio_in_folders(folder_list, model_name="htdemucs_ft", device='auto',
                               progress=True, shifts=5, segment=_DEFAULT_SEGMENT,
                               max_chunk_seconds=_DEFAULT_MAX_CHUNK_SECONDS):
    """
    处理指定目录列表中的音频分离。

    与 separate_all_audio_under_folder 的区别：
    接收的是明确的目录列表而非根目录，不递归遍历。

    Args:
        folder_list: 需要处理的目录路径列表（单个字符串或列表）
        model_name: Demucs 模型名称
        device: 计算设备
        progress: 是否显示进度条
        shifts: 输入偏移次数
        segment: Demucs 内部分段推理的秒数，控制显存占用
        max_chunk_seconds: 外部分块处理的最大分块秒数，控制内存占用

    Returns:
        格式化的执行结果字符串
    """
    # 统一为列表格式
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    global separator
    success_list = []
    fail_list = []

    for subdir in folder_list:
        subdir = os.path.abspath(subdir)
        video_path = os.path.join(subdir, 'download.mp4')

        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            fail_list.append(f"{subdir}: 缺少 download.mp4")
            continue

        try:
            # 第一步：提取音频
            audio_path = os.path.join(subdir, 'audio.wav')
            if not os.path.exists(audio_path):
                logger.info(f'提取音频: {subdir}')
                extract_audio_from_video(subdir)

            # 第二步：分离人声和伴奏
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

    # 处理完成后清理模型
    cleanup_demucs()
    logger.info(
        f'音频分离完成: 成功 {len(success_list)}/{len(folder_list)}, '
        f'失败 {len(fail_list)}')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'


if __name__ == '__main__':
    # 模块直接运行时的示例用法
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)
