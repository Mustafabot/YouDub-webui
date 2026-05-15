# -*- coding: utf-8 -*-
"""
模块执行器 —— 按拓扑序执行模块，支持跳过已完成步骤和重试失败步骤。

本模块是 YouDub 处理流水线的"执行引擎"，负责：
1. 构建执行计划（基于模块依赖关系进行拓扑排序）
2. 验证执行计划的可行性（网络、FFmpeg、配置、模型、输入文件等检查）
3. 按顺序执行模块函数，支持：
   - 跳过已完成模块（根据输出文件存在性判断）
   - 自动重试失败模块（最多可配置重试次数）
   - 智能的输入文件缺失提示（告知用户应运行哪个前置模块）
4. 提供全局默认执行器（单例模式），预先注册所有模块的处理函数

执行流程：
  构建计划 → 验证计划 → 执行初始化 → 逐模块执行（跳过/执行/重试）→ 返回结果
"""

import os       # 文件路径操作和 os.path.exists 检查
import time     # 重试间隔休眠
from loguru import logger  # 结构化日志输出
# 从注册表导入模块元数据查询和依赖解析函数
from .module_registry import (
    get_module, is_module_completed,
    get_available_modules, get_module_missing_config,
    get_input_file_producer
)
# 配置和环境检查工具
from .config import check_network, check_ffmpeg_available, ensure_ffmpeg_available
# 模型状态查询
from .model_manager import check_model_status, get_model_info


def build_execution_plan(selected_modules):
    """
    根据用户选择的模块构建执行计划。

    此函数仅做拓扑排序，不自动添加缺失的上游依赖。
    如果 selected_modules 中没有包含某个模块的上游依赖，该上游不会自动补全。
    这允许用户精确控制执行范围（例如只运行翻译和 TTS，不重新下载和分离）。

    注意：未选中的上游依赖可能意味着输入文件缺失，ModuleExecutor.execute()
    会在运行时检测并报告，或触发兜底逻辑（如 video_synthesis 缺少 TTS 音频时
    自动使用原音轨）。

    Args:
        selected_modules: 用户选择的模块 ID 列表（可能不完整）

    Returns:
        list[str]: 拓扑排序后的执行计划（仅包含用户选择的模块）
    """
    if not selected_modules:
        return []

    # 使用 get_execution_order 进行纯拓扑排序，不自动补全上游依赖
    # get_execution_order 仅对传入的模块集合排序，不添加集合外的模块
    from .module_registry import get_execution_order
    ordered = get_execution_order(selected_modules)
    return ordered


def validate_execution_plan(execution_plan):
    """
    验证执行计划的可行性，检查可能阻止模块执行的条件。

    检查项包括：
        1. 模块是否存在
        2. 网络连接是否满足在线模块需求
        3. FFmpeg 是否可用于需要它的模块
        4. 模块所需的配置项是否已设置
        5. 模块所需的 AI 模型是否已下载
        6. 输入文件的生产者模块是否在执行计划中

    注意：此函数只收集警告而不阻止执行，这是因为：
        - 某些条件（如模型未下载）可以在设置页补救
        - 某些输入文件可能已存在（之前运行过），不需要再次执行其生产者

    Args:
        execution_plan: 拓扑排序后的模块 ID 列表

    Returns:
        list[str]: 警告消息列表（空列表表示无警告）
    """
    warnings = []
    online = check_network()             # 检查网络连通性
    ffmpeg_available, ffmpeg_msg = check_ffmpeg_available()  # 检查 FFmpeg 是否就绪
    available_modules = [m["id"] for m in get_available_modules()]  # 当前可用模块
    plan_set = set(execution_plan)       # 转为集合以便快速查找

    # 逐模块验证
    for module_id in execution_plan:
        module = get_module(module_id)
        if not module:
            warnings.append(f"模块 {module_id} 不存在")
            continue

        # 检查网络需求：如果模块需要网络但当前离线
        if module["online_only"] and not online:
            warnings.append(f"模块 {module['name']} 需要网络连接")

        # 检查 FFmpeg 需求：如果模块需要 FFmpeg 但未安装
        if module.get("requires_ffmpeg", False) and not ffmpeg_available:
            warnings.append(f"模块 {module['name']} 需要 FFmpeg: {ffmpeg_msg}")

        # 检查配置完整性：如果模块不在可用列表中，可能是配置缺失
        if module_id not in available_modules:
            missing = get_module_missing_config(module_id)
            if missing:
                warnings.append(f"模块 {module['name']} 缺少配置: {', '.join(missing)}")

        # 检查模型就绪状态
        required_models = module.get("required_models", [])
        for model_id in required_models:
            status = check_model_status(model_id)
            if not status["downloaded"]:
                model_info = get_model_info(model_id)
                model_name = model_info["name"] if model_info else model_id
                warnings.append(f"模块 {module['name']} 需要模型「{model_name}」尚未下载，请在设置页面下载模型")

        # 检查输入文件的生产者是否在计划中
        # 如果生产者模块不在执行计划中，提示用户可能缺少输入文件
        input_files = module.get("input_files", [])
        if input_files:
            for infile in input_files:
                producer = get_input_file_producer(infile)
                if producer and producer not in plan_set:
                    producer_name = get_module(producer)["name"] if get_module(producer) else producer
                    warnings.append(
                        f"模块 {module['name']} 需要输入文件 {infile}（由模块「{producer_name}」产出），"
                        f"但该模块未在执行计划中。请确保输入文件已存在，或先运行模块「{producer_name}」"
                    )

    return warnings


def check_module_output_exists(folder, module_id):
    """
    检查模块输出文件是否存在于指定文件夹中。

    此为 is_module_completed 的便捷包装，供外部调用使用。

    Args:
        folder: 处理文件夹绝对路径
        module_id: 模块 ID

    Returns:
        bool: 所有输出文件是否存在
    """
    return is_module_completed(folder, module_id)


class ModuleExecutor:
    """
    模块执行器 —— 负责按拓扑排序顺序执行一组模块。

    职责：
        - 管理模块函数的注册
        - 按执行计划逐个运行模块
        - 跳过已完成模块（输出文件已存在）
        - 失败时自动重试（可配置次数）
        - 检测缺失的输入文件并给出智能提示
        - 特殊处理 video_synthesis 缺少 TTS 音频时的兜底逻辑

    使用方式：
        executor = ModuleExecutor()
        executor.register_module_function("tts", tts_func, init_func)
        executor.execute_init_functions(plan)
        results, success = executor.execute(folder, plan, params)
    """

    def __init__(self):
        """
        初始化执行器。

        module_functions: 存储模块 ID 到执行函数的映射
        init_functions:   存储模块 ID 到初始化函数的映射（执行前调用）
        """
        self.module_functions = {}  # {module_id: function}
        self.init_functions = {}    # {module_id: callable}

    def register_module_function(self, module_id, func, init_func=None):
        """
        注册一个模块的执行函数和可选的初始化函数。

        Args:
            module_id: 模块 ID
            func: 执行函数，签名 func(folder, **kwargs) -> Any
            init_func: 初始化函数，签名 init_func() -> None，在模块执行前调用
        """
        self.module_functions[module_id] = func
        if init_func:
            self.init_functions[module_id] = init_func

    def execute_init_functions(self, execution_plan):
        """
        依次执行执行计划中所有模块的初始化函数。

        初始化函数在模块实际执行前调用（例如：加载模型到 GPU）。
        即使初始化失败也不会阻止后续执行（仅记录警告）。
        这是为了容许单个模块初始化失败后其他模块仍可继续运行。

        Args:
            execution_plan: 拓扑排序后的模块 ID 列表
        """
        for module_id in execution_plan:
            if module_id in self.init_functions:
                try:
                    logger.info(f"初始化模块: {get_module(module_id)['name']}")
                    self.init_functions[module_id]()  # 调用初始化函数
                except Exception as e:
                    # 初始化失败仅记录警告，不中断流程
                    logger.warning(f"初始化模块 {module_id} 失败: {e}")

    def execute(self, folder, execution_plan, params, skip_completed=True, max_retries=3):
        """
        按执行计划顺序执行模块。

        执行流程（对每个模块）：
            1. 如果 skip_completed 且模块已完成（输出文件存在），跳过
            2. 检查 FFmpeg 是否就绪（如果需要）
            3. 检查所有输入文件是否存在（特殊处理 video_synthesis 缺少 TTS 音频的情况）
            4. 如果输入文件缺失，给出智能提示（告知应运行哪个前置模块）
            5. 调用注册的执行函数，失败时重试

        关于重试机制：
            - 最多重试 max_retries 次（默认 3 次）
            - 重试间隔 1 秒
            - KeyboardInterrupt 和 SystemExit 不重试，立即向上传播
            - 重试用尽后标记为失败，并记录失败模块 ID 用于下游模块的输入缺失提示

        Args:
            folder: 处理文件夹的绝对路径
            execution_plan: 拓扑排序后的模块 ID 列表
            params: 参数字典，包含所有模块的运行参数
            skip_completed: 是否跳过已完成的模块（默认 True）
            max_retries: 模块执行失败时的最大重试次数（默认 3）

        Returns:
            tuple: (results, success)
                results: list[dict] — 每个模块的执行结果
                    {"module": str, "status": "success"|"skipped"|"error", "result"|"error": ...}
                success: bool — 是否所有模块都执行成功
        """
        results = []
        total = len(execution_plan)
        # failed_predecessor 记录最近失败的模块 ID，
        # 用于为下游模块提供更精确的输入文件缺失原因说明
        failed_predecessor = None

        # 逐模块执行
        for idx, module_id in enumerate(execution_plan, 1):
            module = get_module(module_id)
            if not module:
                continue  # 模块不存在则跳过（不应发生）

            module_name = module["name"]

            # ---- 步骤 1: 检查是否可跳过 ----
            if skip_completed and check_module_output_exists(folder, module_id):
                logger.info(f"[{idx}/{total}] 跳过已完成模块: {module_name}")
                results.append({"module": module_id, "status": "skipped"})
                continue

            # ---- 步骤 2: 确保 FFmpeg 可用 ----
            if module.get("requires_ffmpeg", False):
                ffmpeg_ok, ffmpeg_msg = ensure_ffmpeg_available(auto_download=True)
                if not ffmpeg_ok:
                    logger.error(f"[{idx}/{total}] 模块 {module_name} 需要 FFmpeg: {ffmpeg_msg}")
                    results.append({"module": module_id, "status": "error", "error": ffmpeg_msg})
                    failed_predecessor = module_id
                    continue

            # 提取本模块所需的参数子集
            func_params = self._get_module_params(module_id, params)

            # ---- 步骤 3: 检查输入文件 ----
            input_files = module.get("input_files", [])
            missing_inputs = []
            for infile in input_files:
                if not os.path.exists(os.path.join(folder, infile)):
                    missing_inputs.append(infile)

            # ---- 特殊处理：video_synthesis 缺少 TTS 音频时的兜底逻辑 ----
            # 如果用户没有选择 TTS 步骤，audio_combined.wav 不存在是预期行为。
            # 此时自动切换到"使用原音轨"模式（视频将保留原声，不加配音）。
            if module_id == "video_synthesis" and "audio_combined.wav" in missing_inputs:
                logger.warning(
                    f"[{idx}/{total}] 模块 {module_name} 缺少合成音频 audio_combined.wav，"
                    f"将自动使用原视频音轨。如需配音请先执行 TTS 步骤。"
                )
                func_params["use_original_audio"] = True  # 启用原音轨兜底
                missing_inputs.remove("audio_combined.wav")  # 从缺失列表中移除

            # ---- 步骤 4: 输入文件缺失处理 ----
            if missing_inputs:
                suggestions = []
                for infile in missing_inputs:
                    producer = get_input_file_producer(infile)
                    if producer:
                        producer_name = get_module(producer)["name"] if get_module(producer) else producer
                        # 区分"生产者执行失败"和"生产者未执行"两种情况
                        if producer == failed_predecessor:
                            suggestions.append(f"{infile}（前置模块「{producer_name}」执行失败，无法生成此文件）")
                        else:
                            suggestions.append(f"{infile}（请先运行模块「{producer_name}」或手动准备该文件）")
                    else:
                        suggestions.append(f"{infile}（请手动准备该文件）")
                msg = f"模块 {module_name} 缺少输入文件: " + "、".join(suggestions)
                logger.error(f"[{idx}/{total}] {msg}")
                results.append({"module": module_id, "status": "error", "error": msg})
                failed_predecessor = module_id
                continue

            # ---- 步骤 5: 检查函数是否已注册 ----
            if module_id not in self.module_functions:
                logger.warning(f"[{idx}/{total}] 模块 {module_name} 没有注册执行函数")
                results.append({"module": module_id, "status": "error", "error": "No function registered"})
                continue

            # ---- 步骤 6: 执行模块函数（带重试机制） ----
            func = self.module_functions[module_id]

            for retry in range(max_retries):
                try:
                    logger.info(f"[{idx}/{total}] 执行模块: {module_name} (重试 {retry + 1}/{max_retries})")
                    result = func(folder, **func_params)  # 调用实际的处理函数
                    logger.info(f"[{idx}/{total}] 模块 {module_name} 执行成功")
                    results.append({"module": module_id, "status": "success", "result": result})
                    failed_predecessor = None  # 成功后清除失败标记
                    break  # 跳出重试循环
                except (KeyboardInterrupt, SystemExit):
                    # 用户主动中断或系统退出：立即向上传播，不重试
                    raise
                except Exception as e:
                    logger.error(f"[{idx}/{total}] 模块 {module_name} 执行失败: {e}")
                    if retry == max_retries - 1:
                        # 重试用尽，标记为永久失败
                        results.append({"module": module_id, "status": "error", "error": str(e)})
                        failed_predecessor = module_id
                    time.sleep(1)  # 重试前等待 1 秒

        # ---- 汇总结果 ----
        has_failure = any(r["status"] == "error" for r in results)
        return results, not has_failure

    def _get_module_params(self, module_id, params):
        """
        从全局参数字典中提取指定模块所需的参数字段。

        每个模块只提取其自身需要的参数，避免将无关参数传递给模块函数。

        Args:
            module_id: 模块 ID
            params: 全局参数字典（由调用方传入）

        Returns:
            dict: 该模块的参数字典（可能为空字典）
        """
        # 定义每个模块 ID 到其所需参数键名的映射
        param_mapping = {
            "video_download": {
                "url": params.get("url"),                    # 视频 URL
                "video_path": params.get("video_path"),      # 本地视频路径
                "resolution": params.get("resolution", "1080p"),  # 下载分辨率
                "num_videos": params.get("num_videos", 5),   # 最大下载数量
            },
            "audio_separation": {
                "model_name": params.get("demucs_model"),         # Demucs 模型名称
                "device": params.get("device"),                   # 推理设备（cpu/cuda）
                "progress": True,                                 # 显示进度条
                "shifts": params.get("shifts"),                   # 随机平移增强次数
                "segment": params.get("demucs_segment", 10),      # Demucs 分段长度（秒）
                "max_chunk_seconds": params.get("demucs_max_chunk_seconds", 600),  # 最大分块长度
            },
            "speech_recognition": {
                "model_name": params.get("whisper_model"),          # Whisper 模型大小
                "download_root": params.get("whisper_download_root"),  # 模型下载根目录
                "device": params.get("device"),                     # 推理设备
                "batch_size": params.get("whisper_batch_size"),     # 批处理大小
                "diarization": params.get("whisper_diarization"),   # 是否启用说话人分离
                "min_speakers": params.get("whisper_min_speakers"), # 最小说话人数
                "max_speakers": params.get("whisper_max_speakers"), # 最大说话人数
            },
            "translation": {
                "target_language": params.get("translation_target_language"),  # 翻译目标语言
            },
            "tts": {
                "force_bytedance": params.get("force_bytedance"),  # 是否强制使用火山引擎 TTS
            },
            "video_synthesis": {
                "subtitles": params.get("subtitles"),                          # 是否添加字幕
                "use_original_audio": params.get("use_original_audio", False), # 是否使用原音轨
                "speed_up": params.get("speed_up"),                           # 视频速度调整系数
                "fps": params.get("fps"),                                     # 输出帧率
                "resolution": params.get("target_resolution"),                # 输出分辨率
            },
        }
        return param_mapping.get(module_id, {})


# ---- 全局默认执行器（单例模式） ----
_default_executor = None  # 全局单例变量


def get_default_executor():
    """
    获取全局默认的模块执行器（单例模式）。

    首次调用时创建并注册所有模块函数，后续调用直接返回已创建的实例。
    这样可以避免重复导入模块和重复注册。

    Returns:
        ModuleExecutor: 全局默认执行器实例
    """
    global _default_executor
    if _default_executor is None:
        _default_executor = create_default_executor()  # 懒初始化
    return _default_executor


def create_default_executor():
    """
    创建并配置默认的模块执行器，预先注册流水线中所有步骤的执行函数和初始化函数。

    注册的模块函数：
        - 每个模块注册一个主执行函数（单视频处理模式）
        - 部分模块额外注册初始化函数（如 Demucs、WhisperX、IndexTTS 的模型加载）
        - 额外存储文件夹模式的处理函数映射（_folder_functions）

    Returns:
        ModuleExecutor: 配置完成的执行器实例
    """
    executor = ModuleExecutor()

    # ---- 导入所有步骤模块的处理函数 ----
    from .step000_video_downloader import download_all_videos_under_folder, download_videos_in_folders
    from .step010_demucs_vr import separate_all_audio_under_folder, separate_audio_in_folders, init_demucs
    from .step020_whisperx import transcribe_all_audio_under_folder, transcribe_audio_in_folders, init_whisperx
    from .step030_translation import translate_all_transcript_under_folder, translate_transcripts_in_folders
    from .step040_tts import generate_all_wavs_under_folder, generate_wavs_in_folders
    from .step043_tts_indextts import init_indextts  # IndexTTS 初始化函数
    from .step050_synthesize_video import synthesize_all_video_under_folder, synthesize_video_in_folders
    from .step060_generate_info import generate_all_info_under_folder, generate_info_in_folders
    from .step070_upload_bilibili import upload_all_videos_under_folder, upload_videos_in_folders

    # ---- 注册主执行函数（带可选的初始化函数） ----
    executor.register_module_function("video_download", download_all_videos_under_folder)
    executor.register_module_function("audio_separation", separate_all_audio_under_folder, init_demucs)
    executor.register_module_function("speech_recognition", transcribe_all_audio_under_folder, init_whisperx)
    executor.register_module_function("translation", translate_all_transcript_under_folder)
    executor.register_module_function("tts", generate_all_wavs_under_folder, init_indextts)
    executor.register_module_function("video_synthesis", synthesize_all_video_under_folder)
    executor.register_module_function("generate_info", generate_all_info_under_folder)
    executor.register_module_function("upload_bilibili", upload_all_videos_under_folder)

    # ---- 注册文件夹模式处理函数（用于批量处理多个文件夹） ----
    executor._folder_functions = {
        "video_download": download_videos_in_folders,
        "audio_separation": separate_audio_in_folders,
        "speech_recognition": transcribe_audio_in_folders,
        "translation": translate_transcripts_in_folders,
        "tts": generate_wavs_in_folders,
        "video_synthesis": synthesize_video_in_folders,
        "generate_info": generate_info_in_folders,
        "upload_bilibili": upload_videos_in_folders,
    }

    return executor


def get_all_module_names():
    """
    获取所有模块的 (ID, 名称) 元组列表，用于 UI 下拉选择框等组件。

    Returns:
        list[tuple[str, str]]: 每个元素为 (模块ID, 模块中文名称)
    """
    from .module_registry import get_all_modules
    return [(m["id"], m["name"]) for m in get_all_modules()]


def get_module_with_info():
    """
    获取包含详细信息的模块列表，用于 UI 展示模块选择面板。

    每个模块包含：
        - id: 模块唯一标识
        - name: 中文名称
        - description: 功能描述
        - dependencies: 前置依赖 ID 列表
        - online_only: 是否需要网络
        - required_config: 所需配置项列表
        - required_models: 所需模型列表

    Returns:
        list[dict]: 模块详细信息列表
    """
    from .module_registry import get_all_modules, get_module_dependencies
    result = []
    for m in get_all_modules():
        deps = get_module_dependencies(m["id"])  # 获取依赖信息
        result.append({
            "id": m["id"],
            "name": m["name"],
            "description": m["description"],
            "dependencies": deps,
            "online_only": m["online_only"],
            "required_config": m["required_config"],
            "required_models": m.get("required_models", []),
        })
    return result
