# -*- coding: utf-8 -*-
"""
全流程编排入口 —— 串联整个视频处理流水线。

本模块是 YouDub 处理系统的顶层协调器，负责：
1. 从 URL 或本地路径获取视频信息
2. 下载视频并创建处理文件夹
3. 按用户选择的模块选择性执行处理流水线
4. 支持两种处理模式：
   - 单视频处理模式（process_video_with_modules / process_video）
   - 文件夹批量处理模式（process_folder_with_modules / selected_folders）
5. 收集执行结果并汇总为最终报告

流向说明：
  do_everything() → 解析 URL/本地路径 → 逐个视频循环 →
    ├─ process_video_with_modules() → 选择模块模式
    └─ process_video() → 旧版全流程模式
         └─ 都调至 process_folder_with_modules() → ModuleExecutor.execute()
"""

import json  # 读写 JSON 格式的配置和元数据文件
import os    # 文件路径操作和环境变量
import time  # 上传前等待
import shutil  # 文件复制操作
from loguru import logger  # 结构化日志输出
# 视频下载模块：URL 解析、单视频下载、本地视频处理
from .step000_video_downloader import (
    get_info_list_from_url,       # 从 URL 列表获取视频信息
    download_single_video,         # 下载单个视频
    get_target_folder,             # 获取目标处理文件夹路径
    process_local_video,           # 处理本地导入的视频文件
    get_info_list_from_local       # 从本地路径列表获取视频信息
)
# 模块注册表：模块元数据查询、依赖解析、完成状态检查
from .module_registry import (
    get_module,                   # 获取模块元数据
    get_module_ids,               # 获取所有模块 ID
    is_module_completed,          # 检查模块是否已完成
    get_available_modules,        # 获取当前环境下可用的模块列表
    check_module_config_available # 检查模块所需配置是否完整
)
# 模块执行器：执行计划构建、计划验证、默认执行器创建
from .module_executor import (
    create_default_executor,       # 创建预注册的执行器
    build_execution_plan,          # 构建执行计划
    validate_execution_plan        # 验证执行计划可行性
)


def copy_user_selected_files(folder, selected_files):
    """
    将用户手动选择的输入文件复制到处理文件夹中。

    用户可能在 UI 上手动选择一些文件（如本地音频、字幕等），
    这些文件需要先复制到处理目录，后续模块才能访问它们。

    Args:
        folder: 目标处理文件夹路径
        selected_files: 用户选择的文件字典，格式 {目标文件名: 源文件路径}
            例如：{"audio_vocals.wav": "D:/my_audio.wav"}
    """
    if selected_files is None:
        return  # 没有选择文件，无需操作

    for filename, filepath in selected_files.items():
        if filepath and os.path.exists(filepath):
            dest_path = os.path.join(folder, filename)  # 目标路径
            try:
                # shutil.copy2 保留文件的元数据（修改时间等）
                shutil.copy2(filepath, dest_path)
                logger.info(f"已复制用户选择的文件：{filename} -> {dest_path}")
            except Exception as e:
                logger.warning(f"复制文件失败 {filename}: {e}")


def process_video_with_modules(info, root_folder, params, selected_modules=None, skip_completed=True, selected_files=None):
    """
    使用选择性模块处理单个视频（新版流程入口）。

    与旧的 process_video 函数不同，此函数利用模块化执行系统，
    只运行用户选择的模块，而非始终执行所有模块。

    处理流程：
        1. 获取或创建视频的处理文件夹
        2. 检查是否已上传到 B 站（已上传则跳过）
        3. （非本地视频）下载视频文件
        4. 调用 process_folder_with_modules 执行选中的模块

    Args:
        info: 视频信息字典（包含 title、url 等字段）
        root_folder: 根输出文件夹路径
        params: 全局参数字典
        selected_modules: 选择的模块 ID 列表，None 表示所有模块
        skip_completed: 是否跳过已完成的步骤
        selected_files: 用户手动选择的输入文件字典

    Returns:
        bool: 处理是否成功
    """
    is_local = info.get("is_local", False)  # 是否为本地文件

    # 步骤 1: 获取/创建处理文件夹
    if is_local:
        # 本地视频：将其复制到工作目录并获取文件夹路径
        folder = process_local_video(info, root_folder)
    else:
        # 在线视频：基于视频信息创建标准命名的文件夹
        folder = get_target_folder(info, root_folder)

    if folder is None:
        # 文件夹创建失败，可能是视频信息不完整
        logger.warning(f"Failed to get target folder for video {info['title']}")
        return False

    # 步骤 2: 检查是否已完成上传（幂等性检查）
    # 如果 bilibili.json 存在且上传成功（code == 0），则跳过该视频
    if os.path.exists(os.path.join(folder, "bilibili.json")):
        with open(os.path.join(folder, "bilibili.json"), "r", encoding="utf-8") as f:
            bilibili_info = json.load(f)
        if bilibili_info.get("results", [{}])[0].get("code") == 0:
            logger.info(f"Video already uploaded in {folder}")
            return True

    # 步骤 3: 下载视频（仅非本地视频需要）
    if not is_local:
        folder = download_single_video(info, root_folder, params.get("resolution", "1080p"))
        if folder is None:
            logger.warning(f"Failed to download video {info['title']}")
            return False

    # 步骤 4: 执行选中的模块
    return process_folder_with_modules(folder, params, selected_modules, skip_completed, selected_files)


def process_folder_with_modules(folder, params, selected_modules=None, skip_completed=True, selected_files=None):
    """
    在指定文件夹上执行选中的模块（通用文件夹处理入口）。

    此函数是选择性和全流程模式的共同核心，负责：
        1. 复制用户选择的自定义文件到处理文件夹
        2. 构建执行计划（拓扑排序）
        3. 验证执行计划的可行性
        4. 创建执行器并初始化模型
        5. 逐模块执行流水线
        6. 汇总并记录执行结果

    Args:
        folder: 要处理的文件夹路径（应包含 input_files 所需的所有文件）
        params: 全局参数字典
        selected_modules: 选择的模块 ID 列表，None 表示执行所有模块
        skip_completed: 是否跳过已完成的模块
        selected_files: 用户手动选择的输入文件字典

    Returns:
        bool: 是否全部执行成功
    """
    # 复制用户选择的自定义文件到处理目录
    copy_user_selected_files(folder, selected_files)

    logger.info(f"Process video in {folder}")

    # 如果未指定模块，默认执行所有模块
    if selected_modules is None:
        selected_modules = get_module_ids()

    # 构建执行计划：拓扑排序
    execution_plan = build_execution_plan(selected_modules)

    # 过滤掉未正确配置的模块（如 TTS 未配 BYTEDANCE 或 IndexTTS 不可用）
    available_ids = [m["id"] for m in get_available_modules()]
    unavailable = [m for m in execution_plan if m not in available_ids]
    if unavailable:
        unavailable_names = [get_module(m)["name"] for m in unavailable if get_module(m)]
        logger.warning(
            f"以下模块因配置不完整被跳过: {', '.join(unavailable_names)}，"
            f"请在设置页面补充相关配置"
        )
        execution_plan = [m for m in execution_plan if m in available_ids]

    if not execution_plan:
        logger.warning("No valid modules to execute (all modules skipped due to missing configuration)")
        return True  # 空计划视为成功（没有需要执行的）

    # 验证执行计划：收集警告但不阻止执行
    warnings = validate_execution_plan(execution_plan)
    for warning in warnings:
        logger.warning(warning)

    logger.info(f"Execution plan: {[get_module(mid)['name'] for mid in execution_plan]}")

    # 创建执行器并运行初始化（加载模型到内存/GPU）
    executor = create_default_executor()
    executor.execute_init_functions(execution_plan)

    # 执行模块流水线
    max_retries = params.get("max_retries", 3)  # 默认最大重试 3 次
    results, success = executor.execute(folder, execution_plan, params, skip_completed, max_retries)

    # 构建执行结果摘要日志
    status_lines = []
    for result in results:
        mid = result["module"]
        name = get_module(mid)["name"] if get_module(mid) else mid
        status = result["status"]
        if status == "success":
            status_lines.append(f"  ✅ {name}")
        elif status == "skipped":
            status_lines.append(f"  ⏭️ {name}（已完成）")
        elif status == "error":
            status_lines.append(f"  ❌ {name}: {result.get('error', '未知错误')}")
    logger.info(f"模块执行结果:\n" + "\n".join(status_lines))

    # 如果有任何模块执行失败，记录详细的错误日志
    if not success:
        for result in results:
            if result.get("status") == "error":
                logger.error(f"Module {result['module']} failed: {result.get('error')}")
        return False

    return True


def process_video(
    info, root_folder, resolution, demucs_model, device, shifts,
    whisper_model, whisper_download_root, whisper_batch_size,
    whisper_diarization, whisper_min_speakers, whisper_max_speakers,
    translation_target_language, force_bytedance, subtitles,
    use_original_audio, speed_up, fps, target_resolution,
    max_retries, auto_upload_video
):
    """
    原有全流程处理函数（保持向后兼容）。

    这是旧版的全流程模式入口，始终执行所有流水线步骤：
    下载 → 音频分离 → 语音识别 → 翻译 → TTS → 视频合成 → 信息生成 → B站上传

    与新版的 process_video_with_modules 相比，此函数：
        - 参数扁平化（无 params 字典）
        - 固定执行所有模块（不支持选择性执行）
        - 内部手动导入和调用模块（而非通过 ModuleExecutor）

    注意：此函数保持只为兼容旧调用方，新代码推荐使用 process_video_with_modules。

    Args:
        info: 视频信息字典
        root_folder: 根输出文件夹
        resolution: 视频分辨率
        demucs_model: Demucs 模型名
        device: 推理设备（cpu/cuda）
        shifts: Demucs 随机平移次数
        whisper_model: Whisper 模型大小
        whisper_download_root: Whisper 模型下载根目录
        whisper_batch_size: 批处理大小
        whisper_diarization: 是否启用说话人分离
        whisper_min_speakers: 最小说话人数
        whisper_max_speakers: 最大说话人数
        translation_target_language: 翻译目标语言
        force_bytedance: 是否强制使用火山引擎 TTS
        subtitles: 是否添加字幕
        use_original_audio: 是否使用原音轨（替代 TTS 配音）
        speed_up: 速度调整系数
        fps: 输出帧率
        target_resolution: 输出分辨率
        max_retries: 最大重试次数
        auto_upload_video: 是否自动上传到 B 站

    Returns:
        bool: 处理是否成功
    """
    local_time = time.localtime()  # 记录开始时间（虽然未使用，保留兼容）

    for retry in range(max_retries):
        try:
            is_local = info.get("is_local", False)

            # ---- 获取/创建处理文件夹 ----
            if is_local:
                folder = process_local_video(info, root_folder)
            else:
                folder = get_target_folder(info, root_folder)

            if folder is None:
                logger.warning(f"Failed to get target folder for video {info['title']}")
                return False

            # ---- 检查是否已上传（幂等性检查） ----
            if os.path.exists(os.path.join(folder, "bilibili.json")):
                with open(os.path.join(folder, "bilibili.json"), "r", encoding="utf-8") as f:
                    bilibili_info = json.load(f)
                if bilibili_info.get("results", [{}])[0].get("code") == 0:
                    logger.info(f"Video already uploaded in {folder}")
                    return True

            # ---- 下载视频（非本地视频） ----
            if not is_local:
                folder = download_single_video(info, root_folder, resolution)
                if folder is None:
                    logger.warning(f"Failed to download video {info['title']}")
                    return False

            logger.info(f"Process video in {folder}")

            # ---- 以下按固定顺序执行所有流水线步骤 ----
            # 导入各步骤模块（延迟导入以减少启动时间）
            from .step010_demucs_vr import separate_all_audio_under_folder, cleanup_demucs
            from .step020_whisperx import transcribe_all_audio_under_folder, cleanup_whisperx
            from .step030_translation import translate_all_transcript_under_folder
            from .step040_tts import generate_all_wavs_under_folder
            from .step043_tts_indextts import cleanup_indextts
            from .step050_synthesize_video import synthesize_all_video_under_folder
            from .step060_generate_info import generate_all_info_under_folder
            from .step070_upload_bilibili import upload_all_videos_under_folder

            # 步骤 1: 音频分离（人声分离）
            separate_all_audio_under_folder(
                folder, model_name=demucs_model, device=device, progress=True, shifts=shifts)

            # 步骤 2: 语音识别（WhisperX 转录）
            transcribe_all_audio_under_folder(
                folder, model_name=whisper_model, download_root=whisper_download_root,
                device=device, batch_size=whisper_batch_size, diarization=whisper_diarization,
                min_speakers=whisper_min_speakers, max_speakers=whisper_max_speakers)

            # 步骤 3: 字幕翻译
            translate_all_transcript_under_folder(folder, target_language=translation_target_language)

            # 步骤 4: TTS 语音合成
            generate_all_wavs_under_folder(folder, force_bytedance=force_bytedance)

            # 步骤 5: 视频合成（画面 + 配音/原声 + 字幕）
            synthesize_all_video_under_folder(
                folder, subtitles=subtitles, use_original_audio=use_original_audio,
                speed_up=speed_up, fps=fps, resolution=target_resolution)

            # 步骤 6: 生成上传信息（标题、描述、封面）
            generate_all_info_under_folder(folder)

            # 步骤 7: 自动上传到 B 站（可选）
            if auto_upload_video:
                time.sleep(1)  # 短暂等待确保文件写入完成
                upload_all_videos_under_folder(folder)

            return True

        except Exception as e:
            # 捕获所有异常，记录后重试
            logger.error(f"Error processing video {info['title']}: {e}")
            try:
                # 清理 GPU 资源以便重试
                cleanup_demucs()
                cleanup_whisperx()
                cleanup_indextts()
            except Exception:
                pass  # 清理失败不影响重试

    return False  # 重试用尽后返回失败


def do_everything(
    root_folder, url=None, local_video_paths=None,
    num_videos=5, resolution="1080p",
    demucs_model="htdemucs_ft", device="auto", shifts=5,
    demucs_segment=10, demucs_max_chunk_seconds=600,
    whisper_model="large", whisper_download_root="models/ASR/whisper",
    whisper_batch_size=32, whisper_diarization=True,
    whisper_min_speakers=None, whisper_max_speakers=None,
    translation_target_language="简体中文",
    force_bytedance=True, subtitles=True,
    use_original_audio=False, speed_up=1.05, fps=30,
    target_resolution="1080p",
    max_workers=1, max_retries=3, auto_upload_video=False,
    selected_modules=None, skip_completed=True,
    selected_files=None, selected_folders=None
):
    """
    全自动处理视频 —— 系统的主入口函数，由 Gradio UI 或 CLI 直接调用。

    此函数是整个 YouDub 系统的最高层协调函数，支持三种处理模式：

    模式 1 —— URL 批量处理：
        从 YouTube URL 列表中获取视频信息，逐个下载并处理。
        通过 url 和 num_videos 参数控制。

    模式 2 —— 本地文件处理：
        从用户指定的本地视频文件路径列表中处理视频。
        通过 local_video_paths 参数控制。

    模式 3 —— 文件夹直接处理：
        直接处理已有内容的文件夹（跳过下载阶段）。
        通过 selected_folders 参数控制。

    参数说明：
        基础参数控制处理模式和流水线行为。
        selected_modules / skip_completed 控制模块选择性执行。
        selected_files 允许用户提供自定义输入文件覆盖。
        selected_folders 允许直接处理已有文件夹。

    Args:
        root_folder: 根输出文件夹路径
        url: YouTube 视频 URL（多 URL 用逗号/换行分隔）
        local_video_paths: 本地视频文件路径列表
        num_videos: 从 URL 列表获取的最大视频数
        resolution: 视频下载/输出分辨率
        demucs_model: Demucs 音频分离模型
        device: 推理设备 ("auto" | "cuda" | "cpu")
        shifts: Demucs 随机平移增强次数
        demucs_segment: Demucs 处理分段长度（秒）
        demucs_max_chunk_seconds: 音频最大分块长度（秒）
        whisper_model: Whisper 模型大小
        whisper_download_root: Whisper 模型下载根目录
        whisper_batch_size: Whisper 批处理大小
        whisper_diarization: 是否启用说话人分离
        whisper_min_speakers: 最小说话人数
        whisper_max_speakers: 最大说话人数
        translation_target_language: 翻译目标语言
        force_bytedance: 是否强制使用火山引擎 TTS
        subtitles: 是否在视频中添加字幕
        use_original_audio: 是否使用原音轨
        speed_up: 视频速度调整系数
        fps: 输出帧率
        target_resolution: 输出分辨率
        max_workers: 最大并行工作数（保留参数，暂未实现）
        max_retries: 模块失败时的最大重试次数
        auto_upload_video: 是否自动上传到 B 站
        selected_modules: 选择的模块 ID 列表，None 表示所有模块
        skip_completed: 是否跳过已完成的步骤
        selected_files: 用户手动选择的文件字典 {filename: filepath}
        selected_folders: 用户选择的目录路径列表，直接处理这些目录

    Returns:
        str: 处理结果摘要信息（含成功/失败计数）

    返回格式示例：
        "✅ 成功: 3\n❌ 失败: 1"（URL/本地文件模式）
        "✅ 成功: 2 个目录\n❌ 失败: 1 个目录"（文件夹模式）
        "Error: 未找到有效的 URL"（无有效输入时）
    """
    success_list = []   # 成功处理的视频列表
    fail_list = []      # 失败的视频列表（非文件夹模式）
    failed_folders = [] # 失败的文件夹列表（文件夹模式）

    # ---- 构建全局参数字典 ----
    # 将所有扁平化参数打包为字典，便于传递给各子函数
    params = {
        "resolution": resolution,
        "demucs_model": demucs_model,
        "device": device,
        "shifts": shifts,
        "demucs_segment": demucs_segment,
        "demucs_max_chunk_seconds": demucs_max_chunk_seconds,
        "whisper_model": whisper_model,
        "whisper_download_root": whisper_download_root,
        "whisper_batch_size": whisper_batch_size,
        "whisper_diarization": whisper_diarization,
        "whisper_min_speakers": whisper_min_speakers,
        "whisper_max_speakers": whisper_max_speakers,
        "translation_target_language": translation_target_language,
        "force_bytedance": force_bytedance,
        "subtitles": subtitles,
        "use_original_audio": use_original_audio,
        "speed_up": speed_up,
        "fps": fps,
        "target_resolution": target_resolution,
        "max_retries": max_retries,
        "auto_upload_video": auto_upload_video,
    }

    # ================================================================
    # 模式 1: 文件夹批量处理模式（selected_folders 参数）
    # ================================================================
    # 如果用户传入了 selected_folders，则跳过视频获取/下载阶段，
    # 直接处理已有内容的文件夹。
    if selected_folders:
        if isinstance(selected_folders, str):
            # 用户可能输入了多行路径字符串，按换行切分
            selected_folders = [sf.strip() for sf in selected_folders.split('\n') if sf.strip()]
        logger.info(f"Processing {len(selected_folders)} selected folders")

        for folder in selected_folders:
            folder = os.path.abspath(folder)  # 解析为绝对路径
            if not os.path.exists(folder):
                failed_folders.append(folder)
                logger.warning(f"Folder does not exist: {folder}")
                continue

            try:
                # 复制用户选择的自定义文件并执行模块流水线
                copy_user_selected_files(folder, selected_files)
                success = process_folder_with_modules(folder, params, selected_modules, skip_completed, selected_files)
                if success:
                    success_list.append({"title": os.path.basename(folder), "folder": folder})
                else:
                    failed_folders.append(folder)
            except Exception as e:
                logger.error(f"Error processing folder {folder}: {e}")
                failed_folders.append(folder)

        # ---- 构建文件夹模式的结果摘要 ----
        result_parts = [f"✅ 成功: {len(success_list)} 个目录"]
        if failed_folders:
            result_parts.append(f"❌ 失败: {len(failed_folders)} 个目录")
            # 只显示前 5 个失败目录避免输出过长
            for f in failed_folders[:5]:
                result_parts.append(f"  - {os.path.basename(f)}")
        return "\n".join(result_parts)

    # ================================================================
    # 模式 2 和 3: 从 URL 或本地文件获取视频信息
    # ================================================================
    video_info_iterator = None  # 视频信息迭代器

    if local_video_paths is not None and len(local_video_paths) > 0:
        # 模式 2: 从本地文件路径列表获取视频信息
        video_info_iterator = get_info_list_from_local(local_video_paths, root_folder)

    elif url is not None and url.strip():
        # 模式 3: 从 YouTube URL 获取视频信息
        import re as _re
        # 使用正则将逗号、中文逗号、换行分隔的 URL 列表拆分为数组
        parts = _re.split(r'[\n,，]', url.replace(' ', ''))
        # 过滤出有效的 URL（以 http:// 或 https:// 开头）
        urls = [p.strip() for p in parts if p.strip() and (p.strip().startswith('http://') or p.strip().startswith('https://'))]
        if not urls:
            return "Error: 未找到有效的 URL（URL 需以 http:// 或 https:// 开头）"
        video_info_iterator = get_info_list_from_url(urls, num_videos)

    else:
        # 既没有 URL 也没有本地文件
        return "Error: Please provide either video URL(s), local video file(s), or selected folders"

    # ================================================================
    # 逐个处理获取到的视频
    # ================================================================
    for info in video_info_iterator:
        try:
            if selected_modules is not None:
                # 新版：选择性执行模块
                success = process_video_with_modules(
                    info, root_folder, params,
                    selected_modules, skip_completed, selected_files
                )
            else:
                # 旧版：执行全流程（向后兼容）
                success = process_video(
                    info, root_folder, resolution,
                    demucs_model, device, shifts,
                    whisper_model, whisper_download_root,
                    whisper_batch_size, whisper_diarization,
                    whisper_min_speakers, whisper_max_speakers,
                    translation_target_language, force_bytedance,
                    subtitles, use_original_audio, speed_up,
                    fps, target_resolution, max_retries,
                    auto_upload_video
                )

            if success:
                success_list.append(info)
            else:
                fail_list.append(info)

        except Exception as e:
            # 顶层异常捕获，确保一个视频的失败不会阻塞后续视频
            logger.error(f"Error processing video {info.get('title', 'unknown')}: {e}")
            fail_list.append(info)

    # ---- 返回处理结果摘要 ----
    return f"Success: {len(success_list)}\nFail: {len(fail_list)}"
