# -*- coding: utf-8 -*-
"""
模块注册系统 —— 声明所有处理模块的元数据。

本模块是整个 YouDub 处理流水线的"中心注册表"，负责：
1. 定义每个处理步骤的元数据（依赖关系、输入/输出文件、必填配置、必填模型、所需环境）
2. 提供基于依赖关系的拓扑排序（确定模块执行顺序）
3. 提供模块查询、可用性检查、输入/输出文件溯源等辅助功能
4. 动态调整 TTS 模块的依赖（根据 IndexTTS 是否可用，切换本地或云端方案）

流水线各步骤的关系如下（箭头表示依赖方向）：
  video_download ──→ audio_separation ──→ speech_recognition ──→ translation ──→ tts ──→ video_synthesis ──→ generate_info ──→ upload_bilibili
                                                                                                            ↑
                                                                                                     (video_synthesis 也可不依赖 tts，使用原音轨)
"""

import os  # 用于文件路径操作和 os.path.exists 等文件存在性检查
from loguru import logger  # 结构化日志记录器，用于输出模块状态、警告和错误信息
from .config import get_config, check_network, get_config_status  # 配置读取、网络连通性检查、配置项状态查询


# ============================================================
# MODULES —— 全局模块注册字典
# ============================================================
# 每个键值对定义一个处理模块，字典结构如下：
#   id:             模块唯一标识符，用于引用和排序
#   name:           模块中文名称，用于 UI 显示和日志
#   description:    模块功能简短描述
#   dependencies:   前置依赖模块 ID 列表，执行前必须完成
#   input_files:    本模块需要的输入文件（相对于处理文件夹）
#   required_config: 本模块必需的配置项键名列表（在 .env 或 config.json 中）
#   required_models: 本模块需要下载的 AI 模型 ID 列表
#   requires_ffmpeg: 是否需要 FFmpeg 工具
#   output_files:   本模块产出的文件列表（用于判断模块是否完成）
#   function:       单视频处理函数名（接收 folder 和其他参数）
#   folder_function: 批量文件夹处理函数名（用于文件夹模式）
#   module:         对应的 Python 模块名（不含 .py 后缀）
#   online_only:    是否需要网络连接（用于在线 API 或下载）
# ============================================================
MODULES = {
    # ---- 步骤 0: 视频下载 ----
    "video_download": {
        "id": "video_download",
        "name": "视频下载",
        "description": "从URL下载视频或导入本地文件",  # 支持 YouTube URL 和本地文件导入
        "dependencies": [],                          # 首步骤，无前置依赖
        "input_files": [],                           # 从零开始，无需输入文件
        "required_config": [],                       # 无需额外配置
        "required_models": [],                       # 无需 AI 模型
        "requires_ffmpeg": True,                     # yt-dlp 下载后可能需要 FFmpeg 合并分段
        "output_files": ["download.mp4", "download.info.json"],  # 产出视频文件和元数据
        "function": "download_all_videos_under_folder",
        "folder_function": "download_videos_in_folders",
        "module": "step000_video_downloader",
        "online_only": True,  # 需要网络下载视频
    },
    # ---- 步骤 1: 音频分离 ----
    "audio_separation": {
        "id": "audio_separation",
        "name": "音频分离",
        "description": "分离人声和伴奏",  # 使用 Demucs 模型将音频分离为人声轨和乐器轨
        "dependencies": ["video_download"],          # 必须等视频下载完成后才能分离
        "input_files": ["download.mp4"],             # 输入：下载的视频文件
        "required_config": [],                       # 无需额外配置
        "required_models": ["demucs_htdemucs_ft"],   # 需要 Demucs 混合频域分离模型
        "requires_ffmpeg": True,                     # 需要 FFmpeg 进行音频格式转换
        "output_files": ["audio_vocals.wav", "audio_instruments.wav"],  # 产出人声和伴奏 WAV
        "function": "separate_all_audio_under_folder",
        "folder_function": "separate_audio_in_folders",
        "module": "step010_demucs_vr",
        "online_only": False,  # 本地模型推理，无需网络
    },
    # ---- 步骤 2: 语音识别 ----
    "speech_recognition": {
        "id": "speech_recognition",
        "name": "语音识别",
        "description": "将语音转换为文字",  # 使用 WhisperX + 声纹分割，生成带时间戳的转录
        "dependencies": ["audio_separation"],        # 需要音频分离后才能转录
        "input_files": ["audio_vocals.wav"],         # 输入：分离后的人声
        "required_config": ["HF_TOKEN"],             # 需要 HuggingFace Token 下载模型
        "required_models": ["whisper_large_v3", "whisper_align", "pyannote_segmentation"],
        # whisper_large_v3: 主 ASR 模型
        # whisper_align:    音素对齐模型（逐词时间戳）
        # pyannote_segmentation: 说话人分离模型
        "requires_ffmpeg": True,                     # WhisperX 内部使用 FFmpeg 解码音频
        "output_files": ["transcript.json"],         # 产出带时间戳的转录 JSON
        "function": "transcribe_all_audio_under_folder",
        "folder_function": "transcribe_audio_in_folders",
        "module": "step020_whisperx",
        "online_only": False,  # 模型下载后本地推理
    },
    # ---- 步骤 3: 字幕翻译 ----
    "translation": {
        "id": "translation",
        "name": "字幕翻译",
        "description": "将字幕翻译为目标语言",  # 使用 OpenAI 兼容 API 将英文翻译为中文等
        "dependencies": ["speech_recognition"],      # 需要转录结果作为翻译源文本
        "input_files": ["transcript.json", "download.info.json"],  # 转录 + 视频标题等元数据
        "required_config": ["OPENAI_API_KEY"],       # 需要 OpenAI API Key
        "required_models": [],                       # 调用远程 API，无需本地模型
        "requires_ffmpeg": False,                    # 纯文本处理，无需 FFmpeg
        "output_files": ["translation.json", "summary.json"],  # 产出翻译结果和视频摘要
        "function": "translate_all_transcript_under_folder",
        "folder_function": "translate_transcripts_in_folders",
        "module": "step030_translation",
        "online_only": True,  # 需要调用 OpenAI API
    },
    # ---- 步骤 4: 语音合成 ----
    "tts": {
        "id": "tts",
        "name": "语音合成",
        "description": "将翻译后的文字生成语音",  # 使用 IndexTTS（本地）或火山引擎（云端）生成配音
        "dependencies": ["translation"],             # 需要翻译结果作为 TTS 输入文本
        "input_files": ["translation.json", "audio_vocals.wav", "audio_instruments.wav"],
        # translation.json: 待朗读的翻译文本
        # audio_vocals.wav: 原声用于声线克隆参考（IndexTTS 模式）
        # audio_instruments.wav: 背景音乐，用于与 TTS 合成
        "required_config": ["BYTEDANCE_APPID", "BYTEDANCE_ACCESS_TOKEN"],  # 火山引擎配置（IndexTTS 不可用时）
        "required_models": ["pyannote_embedding", "indextts"],
        # pyannote_embedding: 说话人嵌入提取
        # indextts: IndexTTS 语音合成模型
        "requires_ffmpeg": False,
        "output_files": ["audio_combined.wav", "audio_tts.wav"],
        # audio_tts.wav: 纯 TTS 合成的配音音频
        # audio_combined.wav: 配音 + 背景乐混合
        "function": "generate_all_wavs_under_folder",
        "folder_function": "generate_wavs_in_folders",
        "module": "step040_tts",
        "online_only": True,  # 火山引擎 API 需要网络；IndexTTS 模式无需网络（由 _apply_tts_overrides 动态调整）
    },
    # ---- 步骤 5: 视频合成 ----
    "video_synthesis": {
        "id": "video_synthesis",
        "name": "视频合成",
        "description": "合成最终视频（含配音/原声+字幕）",  # 将原视频、配音、字幕合成为最终输出
        "dependencies": [],  # 注意：此处空列表不表示无依赖，而是在拓扑排序中通过软依赖机制与 tts 关联
        "input_files": ["download.mp4", "translation.json", "audio_combined.wav"],
        # download.mp4: 原始视频画面
        # translation.json: 字幕文本（用于生成硬编码字幕）
        # audio_combined.wav: TTS 合成并混合后的音频（可选，缺失时用原音轨）
        "required_config": [],      # 无需额外配置
        "required_models": [],      # 无需 AI 模型
        "requires_ffmpeg": True,    # 使用 FFmpeg 进行视频/音频轨道合成和字幕叠加
        "output_files": ["video.mp4"],  # 产出最终合成视频
        "function": "synthesize_all_video_under_folder",
        "folder_function": "synthesize_video_in_folders",
        "module": "step050_synthesize_video",
        "online_only": False,
    },
    # ---- 步骤 6: 信息生成 ----
    "generate_info": {
        "id": "generate_info",
        "name": "信息生成",
        "description": "生成视频上传所需信息",  # 生成视频标题、描述、缩略图等 B 站上传所需信息
        "dependencies": ["video_synthesis"],    # 需要视频合成完成后，基于最终视频生成信息
        "input_files": ["summary.json", "download.info.json", "download.jpg"],
        # summary.json: 视频摘要（由翻译步骤产出）
        # download.info.json: 原始视频元数据（标题、描述等）
        # download.jpg: 原始视频封面图
        "required_config": [],
        "required_models": [],
        "requires_ffmpeg": False,
        "output_files": ["video.txt", "video.png"],  # 产出上传用标题/描述文件 + 缩略图
        "function": "generate_all_info_under_folder",
        "folder_function": "generate_info_in_folders",
        "module": "step060_generate_info",
        "online_only": False,
    },
    # ---- 步骤 7: B 站上传 ----
    "upload_bilibili": {
        "id": "upload_bilibili",
        "name": "B站上传",
        "description": "将视频上传到Bilibili",  # 使用 bilibili-api 将视频自动上传到 B 站
        "dependencies": ["generate_info"],       # 需要上传信息就绪后才能上传
        "input_files": ["video.mp4", "summary.json", "video.png", "download.info.json"],
        # 上传需要：视频文件、标题/描述、缩略图、原始元数据
        "required_config": ["BILI_SESSDATA", "BILI_BILI_JCT"],  # B 站登录凭证（Cookie）
        "required_models": [],
        "requires_ffmpeg": False,
        "output_files": ["bilibili.json"],  # 产出上传结果 JSON（包含 BV 号等）
        "function": "upload_all_videos_under_folder",
        "folder_function": "upload_videos_in_folders",
        "module": "step070_upload_bilibili",
        "online_only": True,  # 上传需要网络
    },
}


def get_module(module_id):
    """
    获取指定模块的元数据字典（返回副本，防止意外修改全局定义）。

    Args:
        module_id: 模块 ID（如 "video_download", "tts"）

    Returns:
        dict | None: 模块元数据副本，如果模块不存在则返回 None

    注意：
        - TTS 模块会经过 _apply_tts_overrides 动态调整
        - 返回副本是为了防止调用方意外修改全局 MODULES 字典
    """
    module = MODULES.get(module_id)
    if module is None:
        return None
    # TTS 模块需要动态调整配置依赖（根据 IndexTTS 是否可用）
    result = _apply_tts_overrides(module) if module_id == "tts" else dict(module)
    return result


def get_all_modules():
    """
    获取所有模块的元数据列表（每个都是字典副本）。

    Returns:
        list[dict]: 所有模块按原定义顺序排列的元数据列表
    """
    result = []
    for module in MODULES.values():
        if module["id"] == "tts":
            # TTS 模块需要动态调整（检查 IndexTTS 可用性）
            result.append(_apply_tts_overrides(module))
        else:
            result.append(dict(module))  # 返回副本，防止意外修改
    return result


def _apply_tts_overrides(module):
    """
    根据 IndexTTS（本地语音合成引擎）是否可用来动态调整 TTS 模块的配置依赖和网络需求。

    逻辑说明：
        1. 尝试导入 step043_tts_indextts 模块
        2. 如果 INDEXTTS_AVAILABLE 为 True：
           - 不需要火山引擎 API Key（required_config 置空）
           - 不需要网络连接（online_only = False）
        3. 如果导入失败或 IndexTTS 不可用：
           - 回退到火山引擎 API 方案
           - 需要 BYTEDANCE_APPID 和 BYTEDANCE_ACCESS_TOKEN
           - 需要网络连接

    Args:
        module: TTS 模块的原始元数据字典

    Returns:
        dict: 调整后的 TTS 模块元数据副本
    """
    module = dict(module)  # 创建副本，避免修改全局
    try:
        # 尝试导入 IndexTTS 模块并检查其可用性状态
        from .step043_tts_indextts import INDEXTTS_AVAILABLE
        if INDEXTTS_AVAILABLE:
            # IndexTTS 可用：使用本地 TTS，无需 API Key 和网络
            module["required_config"] = []
            module["online_only"] = False
        else:
            # IndexTTS 不可用：回退到火山引擎 API
            module["required_config"] = ["BYTEDANCE_APPID", "BYTEDANCE_ACCESS_TOKEN"]
            module["online_only"] = True
    except ImportError:
        # 模块文件不存在或导入失败：同样回退到火山引擎
        module["required_config"] = ["BYTEDANCE_APPID", "BYTEDANCE_ACCESS_TOKEN"]
        module["online_only"] = True
    return module


def get_module_ids():
    """
    获取所有模块 ID 的列表。

    Returns:
        list[str]: 模块 ID 列表，顺序与 MODULES 字典定义顺序一致
    """
    return list(MODULES.keys())


def get_available_modules():
    """
    获取当前环境下可用的模块列表。

    可用性判断条件：
        1. 如果模块标记为 online_only，但当前无网络连接，则不可用
        2. 模块所需的配置项必须全部已设置
        3. 模型可用性不在此处检查（模型可在设置页单独下载）

    Returns:
        list[dict]: 可用的模块元数据列表
    """
    online = check_network()       # 检查网络连通性
    status = get_config_status()   # 获取所有配置项的状态（是否已设置）
    available = []

    for module in get_all_modules():
        # 需要网络的模块在网络不可用时跳过
        if module["online_only"] and not online:
            continue

        # 检查该模块所需的所有配置项是否都已设置
        config_ok = True
        for cfg_key in module["required_config"]:
            if not status.get(cfg_key, {}).get("set", False):
                config_ok = False
                break

        if config_ok:
            available.append(module)

    return available


def get_module_input_files(module_id):
    """
    获取指定模块所需的输入文件列表。

    Args:
        module_id: 模块 ID

    Returns:
        list[str]: 输入文件名列表，模块不存在时返回空列表
    """
    module = get_module(module_id)
    if not module:
        return []
    return module.get("input_files", [])


def get_input_file_producer(filename):
    """
    根据输入文件名反查产出该文件的模块 ID。

    例如：
        get_input_file_producer("download.mp4") -> "video_download"
        get_input_file_producer("transcript.json") -> "speech_recognition"

    这在依赖验证中非常有用：当模块需要某个输入文件时，可以找到
    生产该文件的模块，从而检查该模块是否在执行计划中。

    Args:
        filename: 输入文件名称（不含路径）

    Returns:
        str | None: 产出该文件的模块 ID，未找到则返回 None
    """
    for module in get_all_modules():
        if filename in module.get("output_files", []):
            return module["id"]
    return None


def get_module_dependencies(module_id):
    """
    获取指定模块的直接依赖列表。

    Args:
        module_id: 模块 ID

    Returns:
        list[str]: 直接依赖的模块 ID 列表，模块不存在时返回空列表
    """
    module = get_module(module_id)
    if not module:
        return []
    return list(module["dependencies"])


def get_module_reverse_dependencies(module_id):
    """
    获取依赖指定模块的所有模块（即反向依赖列表）。

    例如：
        get_module_reverse_dependencies("video_download")
        -> ["audio_separation"]  （因为 audio_separation 依赖 video_download）

    这在评估某模块修改影响范围时很有用。

    Args:
        module_id: 模块 ID

    Returns:
        list[str]: 依赖该模块的所有模块 ID 列表
    """
    reverse_deps = []
    for module in get_all_modules():
        if module_id in module["dependencies"]:
            reverse_deps.append(module["id"])
    return reverse_deps


def is_module_completed(folder, module_id):
    """
    检查指定模块在给定文件夹中是否已完成（所有输出文件都存在）。

    判断逻辑：
        - 遍历模块定义的 output_files 列表
        - 检查每个文件在处理文件夹中是否存在
        - 所有文件都存在则认为模块完成

    Args:
        folder: 处理文件夹的绝对路径
        module_id: 模块 ID

    Returns:
        bool: 模块是否已完成
    """
    module = get_module(module_id)
    if not module:
        return False

    # 逐一检查每个输出文件是否存在
    for output_file in module["output_files"]:
        file_path = os.path.join(folder, output_file)
        if not os.path.exists(file_path):
            return False  # 任意一个输出文件缺失即视为未完成

    return True


def check_module_config_available(module_id):
    """
    检查指定模块所需的配置项是否全部可用已设置。

    只检查必要配置，不涉及网络或模型检查。

    Args:
        module_id: 模块 ID

    Returns:
        bool: 配置是否完整
    """
    module = get_module(module_id)
    if not module:
        return False

    for cfg_key in module["required_config"]:
        if not get_config(cfg_key):  # get_config 返回 None 表示未设置
            return False

    return True


def get_module_missing_config(module_id):
    """
    获取指定模块缺失的配置项列表（即尚未设置的必填配置）。

    用于给用户提供精确的配置提示信息。

    Args:
        module_id: 模块 ID

    Returns:
        list[str]: 缺失的配置项键名列表
    """
    module = get_module(module_id)
    if not module:
        return []

    missing = []
    for cfg_key in module["required_config"]:
        if not get_config(cfg_key):
            missing.append(cfg_key)

    return missing


def resolve_dependencies(selected_modules):
    """
    解析用户选择的模块，自动包含所有上游依赖模块，并返回拓扑排序后的执行列表。

    例如：
        用户选择 ["video_synthesis"]
        系统自动补全其上游依赖：
        video_download → audio_separation → speech_recognition → translation → tts → video_synthesis

    实现步骤：
        1. 从选中模块开始，BFS（广度优先）遍历所有上游依赖
        2. 将选中的和自动补全的模块合并为完整集合
        3. 调用 get_execution_order 进行拓扑排序

    Args:
        selected_modules: 用户选择的模块 ID 列表

    Returns:
        list[str]: 拓扑排序后的完整执行列表（包含自动补全的上游依赖）
    """
    if not selected_modules:
        return []

    # BFS 遍历，收集所有上游依赖
    expanded = set(selected_modules)  # 使用集合去重
    queue = list(expanded)            # 队列用于 BFS 遍历
    while queue:
        mid = queue.pop()
        module = get_module(mid)
        if module:
            for dep in module["dependencies"]:
                if dep not in expanded:
                    expanded.add(dep)      # 发现新的上游依赖
                    queue.append(dep)      # 继续向上游遍历

    # 拓扑排序后返回
    return get_execution_order(list(expanded))


def get_execution_order(module_ids):
    """
    对给定的模块 ID 列表进行拓扑排序，确定执行顺序。

    实现算法：Kahn 算法（基于入度的拓扑排序）

    特殊规则：
        - 软依赖（soft dependency）：
          当 tts 和 video_synthesis 都在执行计划中时，
          tts 必须先于 video_synthesis 执行（因为 video_synthesis 可以消费 tts 的输出）。
          但 video_synthesis 不强制依赖 tts（可用原音轨兜底），
          所以这种依赖在模块定义中不直接声明，而是通过此处的软依赖机制处理。

    Args:
        module_ids: 待排序的模块 ID 列表

    Returns:
        list[str]: 拓扑排序后的模块 ID 列表

    Raises:
        ValueError: 如果检测到循环依赖，抛出异常并列出无法排序的模块
    """
    # 初始化入度字典和邻接表
    in_degree = {mid: 0 for mid in module_ids}   # 每个模块的入度（前置依赖数）
    graph = {mid: [] for mid in module_ids}       # 有向图邻接表

    # 根据模块定义的 dependencies 构建有向图
    for mid in module_ids:
        module = get_module(mid)
        if module:
            for dep in module["dependencies"]:
                if dep in module_ids:             # 只关注在计划中的依赖
                    graph[dep].append(mid)        # dep → mid 的有向边
                    in_degree[mid] += 1            # mid 的入度 +1

    # ---- 软依赖处理 ----
    # 当 tts 和 video_synthesis 都在执行计划中时，
    # 强制添加 tts → video_synthesis 的边，
    # 确保 TTS 在视频合成之前执行。
    if "tts" in module_ids and "video_synthesis" in module_ids:
        if "video_synthesis" not in graph.get("tts", []):
            graph["tts"].append("video_synthesis")
            in_degree["video_synthesis"] += 1

    # Kahn 算法：从入度为 0 的节点开始拓扑排序
    queue = [mid for mid in module_ids if in_degree[mid] == 0]
    result = []

    while queue:
        node = queue.pop(0)      # 取出一个入度为 0 的节点
        result.append(node)

        # 遍历该节点的所有后继（邻接节点）
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)  # 入度变为 0，可以执行

    # 检查是否所有节点都被排序（如果 result 长度不足，说明存在环）
    if len(result) != len(module_ids):
        missing = set(module_ids) - set(result)
        logger.error(f"检测到模块依赖循环，以下模块无法排序: {missing}")
        raise ValueError(
            f"模块依赖循环: {', '.join(missing)}。请检查 MODULES 定义中的 depends 字段。"
        )

    return result
