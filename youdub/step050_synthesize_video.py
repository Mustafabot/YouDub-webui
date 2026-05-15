# -*- coding: utf-8 -*-
"""
step050_synthesize_video.py — 视频合成（步骤5）

所属流水线位置：
  步骤5/7：视频合成模块
  上游依赖：步骤4（TTS 合成）生成的 audio_combined.wav、翻译文件 translation.json
  下游产出：video.mp4（合成字幕 + 配音 + 伴奏的最终视频）

功能说明：
  1. 检测 FFmpeg 可用性，支持自动下载（Windows 环境）
  2. 自动检测可用的 H.264 视频编码器（libx264 > NVENC > AMF > QSV > libopenh264）
  3. 将翻译文本拆分为字幕段落，生成 SRT 字幕文件
  4. 使用 FFmpeg 滤镜进行视频加速（setpts）、音频加速（atempo）、字幕嵌入（subtitles）
  5. 支持自定义分辨率（宽高比保持，宽高值对齐为偶数）
  6. 支持保留原始音频（跳过 TTS 合成音频）

输入文件：
  - {folder}/download.mp4            : 原始下载视频
  - {folder}/audio_combined.wav      : TTS 合成 + 伴奏的混合音频（use_original_audio=False 时需要）
  - {folder}/translation.json        : 翻译结果（含时间戳和说话人信息，用于生成 SRT）

输出文件：
  - {folder}/video.mp4               : 合成完成的中文配音视频
  - {folder}/subtitles.srt           : 生成的中文字幕文件

配置依赖：
  - FFMPEG_PATH（可选，自动检测或通过 ensure_ffmpeg_available 下载）
"""

import json         # 读写 translation.json 和 ffprobe 输出的 JSON
import os           # 文件和路径操作
import subprocess   # 子进程管理（调用 FFmpeg/FFprope 命令行）
import sys          # 系统信息（检测操作系统类型，选择字体）
import time         # 延时（等待文件系统挂载完成）

from loguru import logger  # 结构化日志

from .config import ensure_ffmpeg_available, get_ffmpeg_path, get_ffprobe_path, PROJECT_ROOT


# ---- 视频编码器缓存 ----
# 缓存检测结果，避免每次合成都重新扫描编码器
_video_encoder_cache = {"name": None}


def get_available_video_encoder():
    """
    检测 FFmpeg 中可用的 H.264 视频编码器。

    按优先级顺序检测（从硬件加速到软件编码）：
      libx264（软件，最兼容）> h264_nvenc（NVIDIA GPU）> h264_amf（AMD GPU）
      > h264_qsv（Intel 核显）> libopenh264（Cisco 开源实现）

    检测结果缓存到全局字典 _video_encoder_cache 中。

    Returns:
        str 或 None: 可用的编码器名称，如果没有找到任何 H.264 编码器则返回 None
    """
    # 如果已缓存，直接返回
    if _video_encoder_cache["name"] is not None:
        return _video_encoder_cache["name"]

    # 获取 FFmpeg 路径
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        return None

    # 运行 ffmpeg -encoders 获取所有可用编码器
    try:
        result = subprocess.run(
            [ffmpeg_path, '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
    except Exception:
        return None

    # 按照优先级顺序匹配编码器名称
    preference_order = [
        'libx264',         # 软件编码，通用性最好
        'h264_nvenc',      # NVIDIA GPU 硬件编码
        'h264_amf',        # AMD GPU 硬件编码
        'h264_qsv',        # Intel 核芯显卡硬件编码
        'libopenh264',     # Cisco 开源 H.264 实现
    ]

    for encoder in preference_order:
        # 匹配编码器名称是否在输出中
        if f' {encoder} ' in output or f'\t{encoder}\t' in output or f' {encoder}\t' in output:
            _video_encoder_cache["name"] = encoder
            return encoder

    # 没有找到任何 H.264 编码器
    return None


def split_text(input_data,
               punctuations=['，', '；', '：', '。', '？', '！', '\n', '”']):
    """
    将翻译文本按中文标点符号拆分为字幕句段。

    规则：
      - 检测中文标点符号（逗号、分号、冒号、句号、问号、感叹号、换行、右引号）作为断句点
      - 若句段长度 < 5 字符且不是最后一个字符，则延后拆分（避免过短的孤立字幕）
      - 若下一个字符是标点，则延后拆分（将标点归入当前句段）

    Args:
        input_data (list): 从 translation.json 加载的数据列表，每项含 start/end/text/translation/speaker
        punctuations (list): 用于断句的中文标点符号列表

    Returns:
        list: 拆分后的字幕块列表，每项含 start/end/text/translation/speaker
    """
    # 判断字符是否为断句标点的内部函数
    def is_punctuation(char):
        return char in punctuations

    output_data = []  # 存储拆分后的结果

    for item in input_data:
        # 获取片段的起始时间、翻译文本等信息
        start = item["start"]
        text = item["translation"]
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = item["text"]
        sentence_start = 0  # 当前句段在 text 中的起始字符位置

        # 跳过空文本
        if not text:
            continue

        # 计算每个字符的平均时长（用于估算拆分后各句段的时间位置）
        duration_per_char = (item["end"] - item["start"]) / len(text)

        # 逐字符扫描，在标点处拆分
        for i, char in enumerate(text):
            # 如果不是标点且不是最后一个字符，继续向前扫描
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            # 句段太短（<5 字符）且不是最后一段时，继续合并
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            # 如果下一个字符也是标点，将标点归入当前句段（避免字幕以标点开头）
            if i < len(text) - 1 and is_punctuation(text[i + 1]):
                continue

            # 提取从 sentence_start 到当前位置的句段
            sentence = text[sentence_start:i + 1]
            # 按字符比例估算该句段的结束时间
            sentence_end = start + duration_per_char * len(sentence)

            # 添加到输出列表
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": original_text,
                "translation": sentence,
                "speaker": speaker
            })

            # 更新起始时间和位置指针
            start = sentence_end
            sentence_start = i + 1

    return output_data


def format_timestamp(seconds):
    """
    将秒数转换为 SRT 字幕格式的时间戳字符串。

    SRT 时间戳格式：HH:MM:SS,mmm
    例如 3661.5 秒 → 01:01:01,500

    Args:
        seconds (float): 以秒为单位的时间值

    Returns:
        str: SRT 格式的时间戳字符串
    """
    millisec = int((seconds - int(seconds)) * 1000)  # 提取毫秒部分
    hours, seconds = divmod(int(seconds), 3600)       # 计算小时和剩余秒数
    minutes, seconds = divmod(seconds, 60)            # 计算分钟和剩余秒数
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"


def generate_srt(translation, srt_path, speed_up=1, max_line_char=30):
    """
    从翻译数据生成 SRT 字幕文件。

    先将翻译数据按标点拆分为字幕句段，然后按 SRT 格式写入文件。
    每行字幕不超过 max_line_char 个字符，超过时自动换行。

    Args:
        translation (list): 翻译数据列表
        srt_path (str):     要写入的 SRT 文件路径
        speed_up (float):   视频加速倍率（影响时间戳缩放）
        max_line_char (int): 每行最大字符数，默认 30

    Returns:
        None（结果写入 srt_path 指定的文件）
    """
    # 先拆分字幕
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(translation):
            # 时间戳按加速倍率缩放
            start = format_timestamp(line['start'] / speed_up)
            end = format_timestamp(line['end'] / speed_up)
            text = line['translation']

            # 计算行数（按 max_line_char 分行的近似值）
            line_count = len(text) // (max_line_char + 1) + 1
            avg = min(round(len(text) / line_count), max_line_char)

            # 将文本按平均长度分行
            text = '\n'.join([text[i * avg:(i + 1) * avg] for i in range(line_count)])

            # 写入 SRT 条目
            f.write(f'{i + 1}\n')                            # 序号
            f.write(f'{start} --> {end}\n')                  # 时间轴
            f.write(f'{text}\n\n')                           # 字幕文本


def get_aspect_ratio(video_path):
    """
    使用 FFprobe 获取视频文件的宽高比。

    Args:
        video_path (str): 视频文件路径

    Returns:
        float: 视频的宽高比（width / height）

    Raises:
        RuntimeError: FFprobe 未找到时抛出
    """
    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path:
        raise RuntimeError("FFprobe 未找到，无法获取视频信息。请运行自动下载或在配置中设置 FFMPEG_PATH")

    # 使用 ffprobe 读取视频流的宽度和高度
    command = [
        ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    dimensions = json.loads(result.stdout)['streams'][0]
    return dimensions['width'] / dimensions['height']


def convert_resolution(aspect_ratio, resolution='1080p'):
    """
    根据宽高比和目标分辨率计算输出视频的宽高（像素）。

    根据宽高比判断视频方向：
      - aspect_ratio < 1：竖屏，宽度固定为目标值
      - aspect_ratio >= 1：横屏，高度固定为目标值
    最终宽高值调整为偶数（FFmpeg 的编码器要求）。

    Args:
        aspect_ratio (float): 视频宽高比
        resolution (str):     目标分辨率标识，如 '1080p'、'720p'

    Returns:
        tuple: (width, height) 像素值，均为偶数
    """
    # 从分辨率字符串中提取数值，如 "1080p" → 1080
    width_val = int(str(resolution).rstrip('p'))
    if width_val <= 0:
        width_val = 1080
        logger.warning(f"无效分辨率 '{resolution}'，使用默认 1080p")

    if aspect_ratio < 1:
        # 竖屏视频：以宽度为基准
        width = width_val
        height = int(width / aspect_ratio)
    else:
        # 横屏视频：以高度为基准
        height = width_val
        width = int(height * aspect_ratio)

    # FFmpeg 要求宽高为偶数（某些编码器的限制）
    width = width - width % 2
    height = height - height % 2

    return width, height


def synthesize_video(folder, subtitles=True, use_original_audio=False,
                     speed_up=1.05, fps=30, resolution='1080p'):
    """
    核心函数：合成最终视频（结合原视频画面、TTS 配音/伴奏、中文字幕）。

    处理流程：
      1. 检查 FFmpeg 可用性（必要时自动下载）
      2. 验证输入文件是否存在（download.mp4、translation.json、audio_combined.wav）
      3. 生成 SRT 字幕文件
      4. 检测视频宽高比并计算输出分辨率
      5. 构建 FFmpeg 滤镜链（视频加速 + 字幕 + 音频加速）
      6. 执行 FFmpeg 命令合成视频

    Args:
        folder (str):             视频处理目录路径
        subtitles (bool):         是否嵌入字幕（默认 True）
        use_original_audio (bool): 是否使用原视频音轨而非 TTS 音频（默认 False）
        speed_up (float):         视频/音频加速倍率（默认 1.05，补偿音视频速度差异）
        fps (int):                输出视频帧率（默认 30）
        resolution (str):         输出分辨率（默认 '1080p'）

    Raises:
        RuntimeError: FFmpeg 不可用、缺少必要文件或 FFmpeg 合成失败
        FileNotFoundError: 输入文件不存在
    """
    # ---- 检查 FFmpeg ----
    ffmpeg_available, ffmpeg_msg = ensure_ffmpeg_available(auto_download=True)
    if not ffmpeg_available:
        raise RuntimeError(
            f'FFmpeg 不可用，无法进行视频合成。{ffmpeg_msg}\n'
            f'请按以下方式之一安装 FFmpeg：\n'
            f'1. 运行 python scripts/download_ffmpeg.py 自动下载\n'
            f'2. Windows: 从 https://ffmpeg.org/download.html 下载，解压后将 bin 目录添加到系统 PATH，'
            f'或在配置中设置 FFMPEG_PATH\n'
            f'3. macOS: brew install ffmpeg\n'
            f'4. Linux: sudo apt install ffmpeg'
        )

    # ---- 检查是否已合成完成 ----
    if os.path.exists(os.path.join(folder, 'video.mp4')):
        logger.info(f'Video already synthesized in {folder}')
        return

    # ---- 输入文件路径 ----
    translation_path = os.path.join(folder, 'translation.json')     # 翻译数据
    input_audio = os.path.join(folder, 'audio_combined.wav')        # TTS + 伴奏音频
    input_video = os.path.join(folder, 'download.mp4')              # 原始视频

    # ---- 输入文件存在性检查 ----
    if not os.path.exists(input_video):
        raise FileNotFoundError(f'视频文件不存在: {input_video}，请确认下载步骤已正确执行')
    if not os.path.exists(translation_path):
        raise FileNotFoundError(f'翻译文件不存在: {translation_path}，请确认翻译步骤已正确执行')
    if not use_original_audio and not os.path.exists(input_audio):
        raise FileNotFoundError(f'合成音频不存在: {input_audio}，请确认TTS步骤已正确执行')

    # ---- 加载翻译数据 ----
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    # ---- 文件路径定义 ----
    srt_path = os.path.join(folder, 'subtitles.srt')   # SRT 字幕输出路径
    output_video = os.path.join(folder, 'video.mp4')    # 最终视频输出路径

    # ---- 生成 SRT 字幕 ----
    generate_srt(translation, srt_path, speed_up)

    # ---- 处理字幕路径中的特殊字符（FFmpeg 需要的转义） ----
    srt_path = srt_path.replace('\\', '/')   # 反斜杠转正斜杠
    srt_path = srt_path.replace(':', '\\:')   # Windows 路径盘符冒号转义

    # ---- 计算输出分辨率 ----
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    resolution_str = f'{width}x{height}'

    # ---- 计算字幕样式参数 ----
    font_size = int(width / 128)          # 字体大小基于视频宽度等比缩放
    outline = int(round(font_size / 8))   # 描边宽度（增强字幕可读性）

    # ---- FFmpeg 滤镜参数 ----
    video_speed_filter = f"setpts=PTS/{speed_up}"     # 视频加速：调整 PTS
    audio_speed_filter = f"atempo={speed_up}"          # 音频加速：调整播放速度（保持音调）

    # 根据操作系统选择字体（Windows 用微软雅黑，其他用 Arial）
    font_name = 'Microsoft YaHei' if sys.platform.startswith('win') else 'Arial'

    # 字幕滤镜：嵌入 SRT 字幕并设置样式
    subtitle_filter = (
        f"subtitles=filename='{srt_path}':"
        f"force_style='FontName={font_name},FontSize={font_size},"
        f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
        f"Outline={outline},WrapStyle=2'"
    )

    # 音频流选择：原始音频使用 0:a（第一个输入的第一个音频流），TTS 音频使用 1:a
    audio_stream = '0:a' if use_original_audio else '1:a'

    # 构建滤镜复合图
    if subtitles:
        # 有字幕：视频加速 → 嵌入字幕 → 输出 v 流；音频加速 → 输出 a 流
        filter_complex = (
            f"[0:v]{video_speed_filter},{subtitle_filter}[v];"
            f"[{audio_stream}]{audio_speed_filter}[a]"
        )
    else:
        # 无字幕：视频加速 → 输出 v 流；音频加速 → 输出 a 流
        filter_complex = (
            f"[0:v]{video_speed_filter}[v];"
            f"[{audio_stream}]{audio_speed_filter}[a]"
        )

    # ---- 检测可用视频编码器 ----
    ffmpeg_path = get_ffmpeg_path()
    video_encoder = get_available_video_encoder()
    if not video_encoder:
        raise RuntimeError(
            '未找到可用的 H.264 视频编码器。当前 FFmpeg 版本可能不支持 '
            'libx264/libopenh264/NVENC/AMF/QSV。\n'
            '请安装包含 H.264 编码支持的 FFmpeg 版本。'
        )
    if video_encoder != 'libx264':
        logger.info(f'使用视频编码器: {video_encoder} (libx264 不可用)')

    # ---- 构建 FFmpeg 命令行 ----
    ffmpeg_command = [
        ffmpeg_path,
        '-i', input_video,                     # 输入：原始视频
    ]
    if not use_original_audio:
        ffmpeg_command.extend(['-i', input_audio])   # 输入：TTS 音频（仅在使用时添加）
    ffmpeg_command.extend([
        '-filter_complex', filter_complex,     # 滤镜复合图
        '-map', '[v]',                         # 视频流映射
        '-map', '[a]',                         # 音频流映射
        '-r', str(fps),                        # 帧率设置
        '-s', resolution_str,                  # 分辨率设置
        '-c:v', video_encoder,                 # 视频编码器
        '-c:a', 'aac',                         # 音频编码器（AAC）
        output_video,                          # 输出文件路径
        '-y'                                   # 覆盖已有文件
    ])

    # ---- 执行 FFmpeg 命令 ----
    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f'FFmpeg 视频合成失败 (返回码 {result.returncode}):')
        for line in result.stderr.strip().split('\n'):
            if line.strip():
                logger.error(f'  {line.strip()}')
        raise RuntimeError(f'FFmpeg 视频合成失败，返回码: {result.returncode}')

    # 短暂延时，确保文件写入完成
    time.sleep(1)


def synthesize_all_video_under_folder(folder, subtitles=True, use_original_audio=False,
                                      speed_up=1.05, fps=30, resolution='1080p'):
    """
    扫描指定目录下所有视频处理目录，逐个执行视频合成。

    通过 os.walk 递归查找包含 download.mp4 的子目录，跳过已合成
    （video.mp4 已存在）的目录。

    Args:
        folder (str):             根目录路径（相对或绝对路径）
        subtitles (bool):         是否嵌入字幕
        use_original_audio (bool): 是否使用原视频音轨
        speed_up (float):         加速倍率
        fps (int):                帧率
        resolution (str):         分辨率

    Returns:
        str: 提示信息，表示所有目录合成完成

    Raises:
        FileNotFoundError: 未找到任何视频处理目录，或目录缺少 download.mp4
    """
    if not os.path.isabs(folder):
        folder = str(PROJECT_ROOT / folder)
    found_video_dir = False
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(folder):
        # 只处理包含 download.mp4 且尚未生成 video.mp4 的目录
        if 'download.mp4' not in files and 'video.mp4' not in files:
            continue
        found_video_dir = True
        # 有 video.mp4 但无 download.mp4 说明步骤顺序异常
        if 'download.mp4' not in files:
            raise FileNotFoundError(
                f'发现视频目录 {root} 但缺少 download.mp4，'
                f'请确认下载步骤已正确执行。目录内容: {files}'
            )
        # 跳过已合成目录
        if 'video.mp4' in files:
            continue
        # 执行视频合成
        synthesize_video(root, subtitles=subtitles, use_original_audio=use_original_audio,
                         speed_up=speed_up, fps=fps, resolution=resolution)
    if not found_video_dir:
        raise FileNotFoundError(f'在 {folder} 下未找到任何视频处理目录')
    return f'Synthesized all videos under {folder}'


def synthesize_video_in_folders(folder_list, subtitles=True, use_original_audio=False,
                                speed_up=1.05, fps=30, resolution='1080p'):
    """处理指定目录列表中的视频合成（批量入口）

    支持传入单个目录路径字符串或多个路径的列表。逐一调用 synthesize_video，
    分别统计成功和失败的个数。

    Args:
        folder_list (str|list):  需要处理的目录路径列表（或单个路径字符串）
        subtitles (bool):        是否嵌入字幕
        use_original_audio (bool): 是否使用原视频音轨
        speed_up (float):        加速倍率
        fps (int):               帧率
        resolution (str):        分辨率

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
    # 命令行直接运行时的测试入口
    folder = r'videos\3Blue1Brown\20231207 Im still astounded this is true'
    synthesize_all_video_under_folder(folder, subtitles=True)
