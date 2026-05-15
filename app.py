#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouDub-webui Gradio Web 界面主入口

本文件是 YouDub 视频本地化工具的 Web 用户界面，使用 Gradio 框架构建。
用户可以通过浏览器访问本界面，完成视频下载、人声分离、语音识别、字幕翻译、
语音合成、视频合成、信息生成和 B 站上传的全流程操作。

核心功能：
  - 设置管理：管理 OpenAI API、火山引擎 TTS、B站上传等配置
  - 下载视频：从 URL 下载或从本地导入视频文件
  - 人声分离：使用 Demucs 模型分离人声和伴奏
  - 语音识别：使用 WhisperX 识别语音生成字幕
  - 字幕翻译：使用大语言模型将字幕翻译为目标语言
  - 语音合成：使用 TTS 将翻译后的文字转为语音
  - 视频合成：将配音、字幕与原视频合成为最终视频
  - 信息生成：生成视频上传所需的摘要、标题等信息
  - 上传B站：将合成好的视频自动上传到 Bilibili
  - 全自动流程：一键完成从下载到上传的全部流程

使用方式：
  python app.py
  （自动在 http://127.0.0.1:19876 启动 Web 服务）
"""

# ============================================================================ #
# 环境初始化：在导入任何其他模块之前设置 Gradio 服务器端口
# 必须在 import gradio 之前设置，因为 gradio 会在导入时读取环境变量
# ============================================================================ #
import os  # 操作系统接口，用于设置环境变量和文件路径操作
os.environ["GRADIO_SERVER_PORT"] = "19876"  # 设置 Gradio 服务端口为 19876

# ============================================================================ #
# 第三方库和内部模块导入
# ============================================================================ #
import warnings  # 警告控制模块，用于过滤不必要的警告信息
# 忽略 pyannote 库产生的 UserWarning 警告（pyannote 是说话者分离库，会有较多非关键警告）
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import gradio as gr  # Gradio Web UI 框架，用于构建交互式界面

# 视频下载模块：支持从 URL 下载和本地文件导入
from youdub.step000_video_downloader import download_from_url, import_local_video

# 人声分离模块：使用 Demucs 模型将视频中的人声和伴奏分离
from youdub.step010_demucs_vr import separate_all_audio_under_folder, cleanup_demucs

# 语音识别模块：使用 WhisperX 模型将语音转换为文字转录
from youdub.step020_whisperx import transcribe_all_audio_under_folder, cleanup_whisperx

# 字幕翻译模块：使用大语言模型将字幕翻译为目标语言
from youdub.step030_translation import translate_all_transcript_under_folder

# 语音合成模块：使用 TTS 模型将翻译后的文字转为语音
from youdub.step040_tts import generate_all_wavs_under_folder

# 视频合成模块：将配音、字幕与原视频合成为最终视频
from youdub.step050_synthesize_video import synthesize_all_video_under_folder

# 信息生成模块：生成视频上传所需的信息（标题、描述、标签等）
from youdub.step060_generate_info import generate_all_info_under_folder

# B站上传模块：将合成好的视频上传到 Bilibili 平台
from youdub.step070_upload_bilibili import upload_all_videos_under_folder

# 全流程编排入口：一键执行从下载到上传的所有步骤
from youdub.do_everything import do_everything

# 配置管理模块：加载/保存配置、验证配置完整性、网络检测等
from youdub.config import load_config, save_config, get_config_status, validate_config, DEFAULT_CONFIG, check_network, get_offline_capabilities, PROJECT_ROOT

# 模块注册系统：模块元数据、依赖关系解析、模块信息查询
from youdub.module_registry import get_module, get_module_dependencies, resolve_dependencies, MODULES

# 模块执行器：获取带有信息的模块列表
from youdub.module_executor import get_module_with_info

# 模型管理模块：模型状态检查、下载、状态格式化
from youdub.model_manager import format_model_status, check_all_models_status, check_model_status, download_model, download_all_models, get_model_info

# 错误处理服务：异常分类和中文错误提示生成
from youdub.services.error_handler import ErrorHandler

# 文件工具服务：文件格式验证、临时目录管理、文件路径处理
from youdub.services.file_utils import FileUtils, INPUT_FILE_FORMATS

# 执行服务：模块执行包装，提供批量执行和单个执行接口
from youdub.services.execution_service import ExecutionService

# 日志系统
from loguru import logger  # loguru 日志库，提供结构化日志输出

# 日志缓冲区：用于在 UI 中显示实时日志
from youdub.log_config import get_log_buffer, clear_log_buffer


# ============================================================================ #
# 全局服务实例
# 使用单例模式，避免重复创建实例浪费资源
# ============================================================================ #
_error_handler = ErrorHandler()      # 错误处理器实例，统一处理各类异常并生成中文提示
_file_utils = FileUtils()            # 文件工具实例，提供文件验证和格式检查功能
_execution_service = ExecutionService()  # 执行服务实例，包装模块执行逻辑


# ============================================================================ #
# 设置保存函数
# 将 UI 中的各项配置保存到 config.json 中
# ============================================================================ #
def save_settings(openai_api_key, openai_api_base, model_name, temperature, top_p, max_tokens, extra_body,
                  translation_system_prompt, translation_fewshot, summary_system_prompt, summary_translation_prompt,
                  hf_token, hf_endpoint,
                  bytedance_appid, bytedance_access_token,
                  tts_stretch_min, tts_stretch_max,
                  indextts_model_dir, indextts_use_fp16, indextts_use_deepspeed,
                  bili_sessdata, bili_bili_jct, bili_base64,
                  pip_index_url, download_timeout, ffmpeg_path):
    """
    保存设置页面中的所有配置项到 config.json 文件

    参数:
        openai_api_key (str): OpenAI API 密钥
        openai_api_base (str): OpenAI API 基础 URL
        model_name (str): 使用的 OpenAI 模型名称
        temperature (float): 输出随机性控制参数
        top_p (float): 核采样参数
        max_tokens (str): 最大输出 Token 数
        extra_body (str): 额外的请求体参数 (JSON)
        translation_system_prompt (str): 翻译系统提示词
        translation_fewshot (str): 翻译少样本示例
        summary_system_prompt (str): 摘要系统提示词
        summary_translation_prompt (str): 摘要翻译提示词
        hf_token (str): HuggingFace Token
        hf_endpoint (str): HuggingFace 镜像端点
        bytedance_appid (str): 火山引擎 TTS App ID
        bytedance_access_token (str): 火山引擎 TTS Access Token
        tts_stretch_min (float): TTS 最小拉伸速度
        tts_stretch_max (float): TTS 最大拉伸速度
        indextts_model_dir (str): IndexTTS 模型目录
        indextts_use_fp16 (bool): 是否使用半精度推理
        indextts_use_deepspeed (bool): 是否使用 DeepSpeed 加速
        bili_sessdata (str): B站 SESSDATA Cookie
        bili_bili_jct (str): B站 bili_jct Cookie
        bili_base64 (str): B站封面 Base64
        pip_index_url (str): PyPI 镜像源地址
        download_timeout (int): 下载超时时间（秒）
        ffmpeg_path (str): FFmpeg 可执行文件路径

    返回:
        tuple: (保存结果消息, 配置状态文本)
    """
    # 将函数参数整理为配置字典，键名与 config.py 中的配置键名一致
    config = {
        "OPENAI_API_KEY": openai_api_key,
        "OPENAI_API_BASE": openai_api_base,
        "MODEL_NAME": model_name,
        "OPENAI_API_TEMPERATURE": temperature,
        "OPENAI_API_TOP_P": top_p,
        "OPENAI_API_MAX_TOKENS": max_tokens,
        "OPENAI_API_EXTRA_BODY": extra_body,
        "TRANSLATION_SYSTEM_PROMPT": translation_system_prompt,
        "TRANSLATION_FEWSHOT_EXAMPLES": translation_fewshot,
        "SUMMARY_SYSTEM_PROMPT": summary_system_prompt,
        "SUMMARY_TRANSLATION_PROMPT": summary_translation_prompt,
        "HF_TOKEN": hf_token,
        "HF_ENDPOINT": hf_endpoint,
        "PIP_INDEX_URL": pip_index_url,
        "BYTEDANCE_APPID": bytedance_appid,
        "BYTEDANCE_ACCESS_TOKEN": bytedance_access_token,
        "TTS_STRETCH_MIN_SPEED": tts_stretch_min,
        "TTS_STRETCH_MAX_SPEED": tts_stretch_max,
        "INDEXTTS_MODEL_DIR": indextts_model_dir,
        "INDEXTTS_USE_FP16": indextts_use_fp16,
        "INDEXTTS_USE_DEEPSPEED": indextts_use_deepspeed,
        "BILI_SESSDATA": bili_sessdata,
        "BILI_BILI_JCT": bili_bili_jct,
        "BILI_BASE64": bili_base64,
        "DOWNLOAD_TIMEOUT": download_timeout,
        "FFMPEG_PATH": ffmpeg_path,
    }
    # 调用 save_config 将配置写入文件
    save_config(config)
    # 刷新全局配置缓存，确保 UI 各组件能读取到最新配置
    _refresh_cfg()
    # 验证配置完整性，返回缺失的必填项
    missing = validate_config()
    if missing:
        # 如果有缺失的必填项，按功能分类排序后生成提示消息
        features = sorted(set(missing.values()))
        msg = f"配置已保存！以下功能缺少必填项：{', '.join(features)}"
    else:
        msg = "配置已保存！所有必填项均已设置。"
    # 返回保存结果消息和格式化的配置状态文本
    return msg, _format_status()


# ============================================================================ #
# 内部辅助函数：格式化配置状态显示
# ============================================================================ #
def _format_status():
    """
    格式化配置状态为可读的文本显示

    遍历所有配置项，对每个必填项显示其设置状态（已设置/未设置），
    同时包含模型状态和 FFmpeg 状态的显示。

    返回:
        str: 多行文本，每行显示一个配置项的状态
    """
    # 获取配置状态字典，包含所有配置项的设置情况
    status = get_config_status()
    lines = []  # 收集各状态行的列表
    # 特殊键（模型和 FFmpeg 状态需要单独格式化）
    special_keys = {"MODELS", "FFMPEG"}
    # 遍历除特殊键外的所有配置项
    for key, info in status.items():
        if key in special_keys:
            continue  # 跳过特殊键，单独处理
        if info["required"]:
            # 必填项：已设置显示 ✅，未设置显示 ❌
            mark = "✅" if info["set"] else "❌"
            # 格式：状态标记 配置键名 (所属功能)
            lines.append(f"{mark} {key} ({info['feature']})")
    # 单独处理模型状态
    if "MODELS" in status:
        model_info = status["MODELS"]
        mark = "✅" if model_info["set"] else "⚠️"
        lines.append(f"{mark} {model_info['message']} ({model_info['feature']})")
    # 单独处理 FFmpeg 状态
    if "FFMPEG" in status:
        ffmpeg_info = status["FFMPEG"]
        mark = "✅" if ffmpeg_info["set"] else "❌"
        lines.append(f"{mark} FFmpeg ({ffmpeg_info['feature']})")
    # 将所有状态行用换行符连接返回
    return "\n".join(lines)


def _format_network_status():
    """
    格式化网络连接状态为可读的文本显示

    检测当前网络连接状态，如果在线则显示绿色在线状态，
    如果离线则列出离线时可用的功能和不可用的功能。

    返回:
        str: 网络状态的多行文本
    """
    # 检测网络连接是否正常
    online = check_network()
    if online:
        # 在线状态：显示绿色信号
        return "🟢 在线 - 网络连接正常"
    # 离线状态：获取离线能力信息
    caps = get_offline_capabilities()
    lines = ["🔴 离线 - 网络不可用", "", "离线可用功能："]
    # 列出离线状态下仍然可用的功能
    for f in caps["available"]:
        lines.append(f"  ✅ {f}")
    lines.append("")
    lines.append("离线不可用功能：")
    # 列出离线状态下不可用的功能
    for f in caps["unavailable"]:
        lines.append(f"  ❌ {f}")
    return "\n".join(lines)


def _format_model_status_ui():
    """
    格式化模型状态用于 UI 显示（直接调用 format_model_status）

    返回:
        str: 模型状态的格式化文本
    """
    return format_model_status()


def _download_all_missing_models(progress=gr.Progress()):
    """
    下载所有缺失的 AI 模型

    检查所有注册模型的下载状态，自动下载缺失的模型，
    并使用 Gradio 的 Progress 组件显示下载进度。

    参数:
        progress (gr.Progress): Gradio 进度条组件实例

    返回:
        str: 下载结果的多行文本，包含每个模型的下载状态
    """
    # 检查所有模型的状态，返回字典 {模型ID: {downloaded: bool, ...}}
    statuses = check_all_models_status()
    # 筛选出所有尚未下载的模型 ID
    missing = [mid for mid, s in statuses.items() if not s["downloaded"]]
    if not missing:
        # 没有缺失的模型，直接返回
        return "所有模型已下载，无需额外操作"

    lines = []  # 收集每条下载结果
    total = len(missing)  # 需要下载的模型总数
    progress(0, desc="准备下载模型...")  # 初始化进度条
    # 逐个下载缺失的模型
    for i, model_id in enumerate(missing):
        # 获取模型信息以显示友好的模型名称
        info = get_model_info(model_id)
        name = info["name"] if info else model_id
        # 更新进度条：当前进度 = i/total
        progress((i) / total, desc=f"[{i+1}/{total}] 正在下载: {name}")
        try:
            # 执行模型下载
            download_model(model_id)
            lines.append(f"✅ [{i+1}/{total}] {name} 下载完成")
        except ValueError as e:
            # 值错误：通常是配置问题（如缺少 HF_TOKEN）
            lines.append(f"❌ [{i+1}/{total}] {name} 下载失败: {e}")
        except ImportError as e:
            # 导入错误：缺少依赖库
            lines.append(f"❌ [{i+1}/{total}] {name} 库未安装: {e}")
        except Exception as e:
            # 其他未知异常
            lines.append(f"❌ [{i+1}/{total}] {name} 下载失败: {str(e)}")
    # 统计成功和失败的数量
    success = sum(1 for l in lines if l.startswith("✅"))
    fail = sum(1 for l in lines if l.startswith("❌"))
    # 追加汇总信息
    lines.append(f"\n下载完成：成功 {success} 个，失败 {fail} 个")
    progress(1.0, desc="下载完成")  # 设置进度条为完成状态
    return "\n".join(lines)


def _refresh_model_status():
    """刷新模型状态显示，调用 format_model_status 返回最新状态"""
    return format_model_status()


# ============================================================================ #
# 全自动流程的主包装函数
# 接收来自 UI 的所有参数，调用 do_everything 执行完整流程
# ============================================================================ #
def do_everything_wrapper(input_mode, url, local_files, root_folder, num_videos, resolution, translation_target_language, subtitles, use_original_audio, auto_upload_video,
                          demucs_model, demucs_device, shifts, demucs_segment, demucs_max_chunk, whisper_model, whisper_batch_size, whisper_diarization, whisper_min_speakers, whisper_max_speakers, whisper_download_root,
                          speed_up, fps, max_workers, max_retries, force_bytedance, selected_modules, skip_completed, use_module_selection, selected_files,
                          de_folder_list_text, de_folder_select_files):
    """
    全自动流程的包装函数

    整合所有参数验证逻辑，调用核心 do_everything 函数执行全流程处理。
    处理输入模式切换（URL下载/本地文件）、参数默认值设置、
    错误处理和日志收集等功能。

    参数:
        input_mode (str): 输入模式，'URL 下载' 或 '本地文件'
        url (str): 视频 URL 或播放列表/频道链接
        local_files (list): 本地视频文件路径列表
        root_folder (str): 视频文件根目录
        num_videos (int): 要下载的视频数量
        resolution (str): 视频分辨率
        translation_target_language (str): 翻译目标语言
        subtitles (bool): 是否添加字幕
        use_original_audio (bool): 是否使用原视频音轨
        auto_upload_video (bool): 是否自动上传到B站
        demucs_model (str): Demucs 音频分离模型名称
        demucs_device (str): 计算设备 (auto/cuda/cpu)
        shifts (int): Demucs 移位数量
        demucs_segment (int): Demucs 内部段长（秒）
        demucs_max_chunk (int): 最大分块大小（秒）
        whisper_model (str): Whisper 模型名称
        whisper_batch_size (int): Whisper 批处理大小
        whisper_diarization (bool): 是否启用说话者分离
        whisper_min_speakers (int): 最小说话人数
        whisper_max_speakers (int): 最大说话人数
        whisper_download_root (str): Whisper 模型下载路径
        speed_up (float): 视频加速倍率
        fps (int): 合成视频帧率
        max_workers (int): 并发工作线程数
        max_retries (int): 最大重试次数
        force_bytedance (bool): 是否强制使用火山引擎 TTS
        selected_modules (list): 用户选择的模块列表
        skip_completed (bool): 是否跳过已完成步骤
        use_module_selection (bool): 是否启用模块选择
        selected_files (dict): 用户选择的输入文件字典
        de_folder_list_text (str): 手动输入的目录路径列表（文本）
        de_folder_select_files (list): 通过文件选择器选择的文件列表

    返回:
        str: 处理结果的文本，包含日志和状态信息
    """
    # 初始化本地视频路径列表为 None
    local_video_paths = None
    # 合并两种方式指定的目录列表：文本输入和文件选择器
    selected_folders = _file_utils.merge_folder_lists(de_folder_list_text, de_folder_select_files)

    # 处理目录选择优先的情况：如果选择了已有目录，则直接从目录获取视频
    if selected_folders:
        # 即使有 selected_folders，也要检查输入模式是否匹配
        if input_mode == '本地文件':
            # 本地文件模式：检查是否同时上传了本地文件（仅用于目录不存在时回退）
            if local_files is None or len(local_files) == 0:
                pass  # 没有上传文件也没关系，使用已选目录
            else:
                # 有上传文件时，也将其纳入处理范围
                local_video_paths = [f.name if hasattr(f, 'name') else f for f in local_files]
        elif input_mode == 'URL 下载':
            # URL 模式，URL 可能为空（使用已选目录时不需要 URL）
            if not url or not url.strip():
                pass
    else:
        # 没有选择已有目录的情况：按常规流程处理视频来源
        if input_mode == '本地文件':
            # 本地文件模式：必须选择视频文件
            if local_files is None or len(local_files) == 0:
                # 未选择文件，返回错误提示
                return _error_handler.format_error(
                    "未选择视频文件",
                    ["没有选择任何本地视频文件"],
                    ["请点击「上传文件」按钮选择一个或多个视频文件"]
                )
            # 提取文件路径（Gradio 的 File 组件返回的对象可能有 name 属性）
            local_video_paths = [f.name if hasattr(f, 'name') else f for f in local_files]
        else:
            # URL 下载模式：必须填写 URL
            if not url or not url.strip():
                # URL 为空，返回错误提示
                return _error_handler.format_error(
                    "未填写视频链接",
                    ["视频链接为空"],
                    ["请输入有效的视频、播放列表或频道链接"]
                )

    # 检查离线状态并收集警告信息
    offline_warnings = []
    if not check_network() and input_mode == 'URL 下载':
        # 离线状态下无法从 URL 下载视频
        caps = get_offline_capabilities()
        offline_warnings.append("⚠️ 当前处于离线模式，无法从 URL 下载视频")
        offline_warnings.append("请选择「本地文件」模式或选择已有目录")
        offline_warnings.append("")

    # 加载当前配置
    cfg = load_config()

    # 处理模块选择参数
    modules_param = None
    if use_module_selection and selected_modules and len(selected_modules) > 0:
        modules_param = selected_modules  # 使用用户选择的模块列表

    # 检查 OpenAI API Key 是否配置
    if not cfg.get('OPENAI_API_KEY'):
        if modules_param is None or 'translation' in modules_param:
            # 如果启用了翻译模块但未配置 API Key，返回错误
            return _error_handler.format_error(
                "缺少 OpenAI API Key",
                ["翻译功能需要 OpenAI API Key 才能运行", "API Key 未在设置页面中配置"],
                ["前往设置页面填写 OpenAI API Key", "获取 API Key：https://platform.openai.com/api-keys"]
            )

    # 清空日志缓冲区，为新一轮处理做准备
    clear_log_buffer()
    try:
        # 调用核心 do_everything 函数执行全流程
        result = do_everything(
            root_folder=root_folder,
            url=url if input_mode == 'URL 下载' else None,
            local_video_paths=local_video_paths,
            num_videos=int(num_videos),
            resolution=resolution,
            demucs_model=demucs_model,
            device=demucs_device,
            shifts=int(shifts),
            demucs_segment=int(demucs_segment),
            demucs_max_chunk_seconds=int(demucs_max_chunk),
            whisper_model=whisper_model,
            whisper_download_root=whisper_download_root,
            whisper_batch_size=int(whisper_batch_size),
            whisper_diarization=whisper_diarization,
            whisper_min_speakers=whisper_min_speakers,
            whisper_max_speakers=whisper_max_speakers,
            translation_target_language=translation_target_language,
            force_bytedance=force_bytedance,
            subtitles=subtitles,
            use_original_audio=use_original_audio,
            speed_up=float(speed_up),
            fps=int(fps),
            target_resolution=resolution,
            max_workers=int(max_workers),
            max_retries=int(max_retries),
            auto_upload_video=auto_upload_video,
            selected_modules=modules_param,
            skip_completed=skip_completed,
            selected_files=selected_files,
            selected_folders=selected_folders if selected_folders else None
        )
        # 美化输出：确保结果以 ✅ 或 ❌ 开头，若不以标记开头则加上 ✅
        output = f"✅ {result}" if result and not str(result).startswith("✅") and not str(result).startswith("❌") else result
        # 如果有离线警告，附加在结果前面
        if offline_warnings:
            output = "\n".join(offline_warnings) + output
        # 获取运行日志并附加到输出结果中
        logs = get_log_buffer()
        if logs:
            return f"{logs}\n\n{output}"
        return output
    except Exception as e:
        # 捕获所有异常，记录日志并返回格式化错误信息
        logger.error(f"全自动流程执行失败: {e}")
        logs = get_log_buffer()
        error_result = _error_handler.classify_error(e)
        if logs:
            return f"{logs}\n\n{error_result}"
        return error_result


# ============================================================================ #
# 各功能模块的包装函数
# 为 Gradio UI 提供各步骤单独执行的接口
# ============================================================================ #

def demucs_wrapper(files, output_dir, model, device, progress, shifts, segment, max_chunk_seconds):
    """
    人声分离模块的包装函数

    封装 separate_audio_in_folders 调用，提供统一的执行入口和清理逻辑。

    参数:
        files (list): 视频文件路径列表
        output_dir (str): 输出目录路径
        model (str): Demucs 模型名称
        device (str): 计算设备 (auto/cuda/cpu)
        progress: Gradio 进度条
        shifts (int): 移位数量
        segment (int): 内部段长（秒）
        max_chunk_seconds (int): 最大分块大小（秒）

    返回:
        tuple: (输出文本, 生成文件列表)
    """
    # 延迟导入 separate_audio_in_folders（仅在需要时导入，避免启动时间过长）
    from youdub.step010_demucs_vr import separate_audio_in_folders
    # 使用执行服务执行批量处理
    return _execution_service.execute_batch_with_files(
        module_name="人声分离",
        batch_function=separate_audio_in_folders,
        file_paths=files,
        target_filename="download.mp4",  # 在目录中查找的目标文件
        output_dir=output_dir,
        cleanup_function=cleanup_demucs,  # 完成后清理临时文件
        model_name=model, device=device, progress=progress,
        shifts=int(shifts), segment=int(segment),
        max_chunk_seconds=int(max_chunk_seconds)
    )


def whisper_wrapper(files, output_dir, model, diarization, download_root, device, batch_size, min_speakers, max_speakers):
    """
    语音识别模块的包装函数

    封装 transcribe_audio_in_folders 调用，支持说话者分离。

    参数:
        files (list): 音频文件路径列表
        output_dir (str): 输出目录路径
        model (str): Whisper 模型名称
        diarization (bool): 是否启用说话者分离
        download_root (str): 模型下载路径
        device (str): 计算设备
        batch_size (int): 批处理大小
        min_speakers: 最小说话人数
        max_speakers: 最大说话人数

    返回:
        tuple: (输出文本, 生成文件列表)
    """
    from youdub.step020_whisperx import transcribe_audio_in_folders
    return _execution_service.execute_batch_with_files(
        module_name="语音识别",
        batch_function=transcribe_audio_in_folders,
        file_paths=files,
        target_filename="audio_vocals.wav",  # 在目录中查找的人声音频文件
        output_dir=output_dir,
        cleanup_function=cleanup_whisperx,  # 完成后清理临时文件
        model_name=model, download_root=download_root, device=device,
        batch_size=int(batch_size), diarization=diarization,
        min_speakers=min_speakers, max_speakers=max_speakers
    )


def synthesize_wrapper(video_file, translation_file, audio_combined_file, output_dir, subtitles, use_original_audio, resolution, speed_up, fps):
    """
    视频合成模块的包装函数

    封装 synthesize_all_video_under_folder 调用，将配音、字幕与原视频合成。

    参数:
        video_file: 视频文件路径
        translation_file: 翻译结果文件路径
        audio_combined_file: 合成音频文件路径
        output_dir (str): 输出目录
        subtitles (bool): 是否添加字幕
        use_original_audio (bool): 是否使用原视频音轨
        resolution (str): 输出分辨率
        speed_up (float): 加速倍率
        fps (int): 视频帧率

    返回:
        tuple: (输出文本, 生成文件列表)
    """
    return _execution_service.execute_single_with_files(
        module_name="视频合成",
        single_function=synthesize_all_video_under_folder,
        # 文件映射：将 UI 选择的文件映射到模块期望的文件名
        file_map={
            "download.mp4": video_file,
            "translation.json": translation_file,
            "audio_combined.wav": audio_combined_file,
        },
        output_dir=output_dir,
        subtitles=subtitles, use_original_audio=use_original_audio,
        speed_up=float(speed_up), fps=int(fps), resolution=resolution
    )


# ============================================================================ #
# 自动执行的初始化代码
# ============================================================================ #
_cfg = {}  # 配置缓存字典，用于 UI 初始化时读取配置


def _refresh_cfg():
    """刷新全局配置缓存，从配置文件重新加载"""
    global _cfg
    _cfg = load_config()


_refresh_cfg()  # 模块加载时自动刷新一次配置缓存

# 分辨率选项列表，供 UI 下拉选择
RESOLUTION_CHOICES = ['1080p', '720p', '480p']

# 获取模块信息列表，用于模块选择 UI
MODULES_INFO = get_module_with_info()
# 生成模块选择元组列表：(显示名称, 模块ID)
MODULE_CHOICES = [(m["name"], m["id"]) for m in MODULES_INFO]
# 默认选中所有模块
DEFAULT_MODULES = [m["id"] for m in MODULES_INFO]


# ============================================================================ #
# UI 辅助函数
# ============================================================================ #

def format_execution_order(selected_modules):
    """
    格式化模块执行顺序为可读文本

    根据模块间依赖关系进行拓扑排序，显示各模块的执行顺序及其输入文件需求。

    参数:
        selected_modules (list): 选中的模块 ID 列表

    返回:
        str: 格式化后的执行顺序文本
    """
    if not selected_modules or len(selected_modules) == 0:
        return "未选择任何模块"  # 没有选择模块时的提示

    try:
        # 解析依赖关系，返回拓扑排序后的模块顺序
        ordered = resolve_dependencies(selected_modules)
        lines = ["模块执行顺序："]
        # 逐个显示排序后的模块
        for i, mid in enumerate(ordered, 1):
            module = get_module(mid)  # 获取模块详细信息
            if module:
                input_info = ""
                if module.get("input_files"):
                    # 如果有输入文件要求，显示在括号中
                    input_info = f"（需要: {', '.join(module['input_files'])}）"
                lines.append(f"  {i}. {module['name']}{input_info}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"格式化模块执行顺序失败: {e}")
        return "模块选择无效"  # 依赖解析失败时的提示


def select_all_modules():
    """
    全选所有模块

    用于 UI 中的「全选」按钮回调。

    返回:
        tuple: (所有模块ID列表, 格式化后的执行顺序)
    """
    return DEFAULT_MODULES, format_execution_order(DEFAULT_MODULES)


def deselect_all_modules():
    """
    取消选择所有模块

    用于 UI 中的「全不选」按钮回调。

    返回:
        tuple: (空列表, 格式化后的执行顺序)
    """
    return [], format_execution_order([])


def get_default_output_dirs():
    """
    获取默认输出目录列表

    确保 output、videos、results 目录存在，并返回它们。

    返回:
        list: 默认输出目录名称列表
    """
    default_dirs = ['output', 'videos', 'results']
    # 确保目录存在，如果不存在则创建
    for d in default_dirs:
        dir_path = os.path.join(str(PROJECT_ROOT), d)
        os.makedirs(dir_path, exist_ok=True)
    return default_dirs


def set_output_dir(selected_dir):
    """
    设置输出目录的回调函数

    参数:
        selected_dir (str): 选中的目录名称

    返回:
        str: 选中的目录名称
    """
    return selected_dir


def toggle_input_mode(mode):
    """
    切换输入模式的 UI 回调

    根据选择的输入模式，显示/隐藏对应的 UI 组件。

    参数:
        mode (str): 输入模式，'本地文件' 或 'URL 下载'

    返回:
        tuple: (URL 输入框的可见性更新, 本地文件选择器的可见性更新)
    """
    if mode == '本地文件':
        # 本地文件模式：隐藏 URL 输入，显示文件上传组件
        return gr.update(visible=False), gr.update(visible=True)
    else:
        # URL 下载模式：显示 URL 输入，隐藏文件上传组件
        return gr.update(visible=True), gr.update(visible=False)


def import_local_videos_wrapper(local_files, folder_path, title=None, uploader=None, upload_date=None):
    """
    导入本地视频的包装函数

    将用户选择的本地视频文件导入到工作目录中，支持自定义元数据。

    参数:
        local_files (list): 本地视频文件路径列表
        folder_path (str): 目标文件夹路径
        title (str, optional): 自定义标题，留空则使用文件名
        uploader (str, optional): 上传者名称，留空默认为 Local
        upload_date (str, optional): 上传日期，留空使用文件修改日期

    返回:
        str: 导入结果文本，包含成功/失败统计和详细日志
    """
    # 检查是否选择了文件
    if local_files is None or len(local_files) == 0:
        return _error_handler.format_error(
            "未选择视频文件",
            ["没有选择任何本地视频文件"],
            ["请点击「上传文件」按钮选择一个或多个视频文件"]
        )
    # 清空日志缓冲区
    clear_log_buffer()
    # 统计成功和失败数量
    success_count = 0
    failed_count = 0
    results = []
    # 遍历所有选择的文件进行导入
    for f in local_files:
        # 获取文件路径（Gradio File 组件可能返回对象或字符串）
        file_path = f.name if hasattr(f, 'name') else f
        try:
            # 调用核心导入函数
            result = import_local_video(
                file_path,
                folder_path,
                title=title if title else None,
                uploader=uploader if uploader else None,
                upload_date=upload_date if upload_date else None
            )
            if result:
                success_count += 1
                results.append(f"✅ {os.path.basename(file_path)} -> {result}")
            else:
                failed_count += 1
                results.append(f"❌ {os.path.basename(file_path)} -> 导入失败")
        except Exception as e:
            # 捕获导入过程中的异常
            logger.error(f"导入视频失败 {file_path}: {e}")
            failed_count += 1
            results.append(f"❌ {os.path.basename(file_path)} -> 导入异常: {e}")
    # 汇总结果
    output = "\n".join(results) + f"\n\n成功: {success_count}\n失败: {failed_count}"
    # 获取运行日志
    logs = get_log_buffer()
    if logs:
        return f"{logs}\n\n{output}"
    return output


def translation_wrapper(transcript_file, info_file, output_dir, lang):
    """
    字幕翻译模块的包装函数

    封装 translate_all_transcript_under_folder 调用，将识别结果翻译为目标语言。

    参数:
        transcript_file: 识别结果文件 (transcript.json)
        info_file: 视频信息文件 (download.info.json)
        output_dir (str): 输出目录
        lang (str): 目标语言

    返回:
        tuple: (输出文本, 生成文件列表)
    """
    return _execution_service.execute_single_with_files(
        module_name="字幕翻译",
        single_function=translate_all_transcript_under_folder,
        file_map={
            "transcript.json": transcript_file,      # 语音识别结果
            "download.info.json": info_file,          # 视频元信息
        },
        output_dir=output_dir,
        target_language=lang
    )


def tts_wrapper(translation_file, vocals_file, instruments_file, output_dir, force_bytedance):
    """
    语音合成模块的包装函数

    封装 generate_all_wavs_under_folder 调用，将翻译文字转为语音。

    参数:
        translation_file: 翻译结果文件 (translation.json)
        vocals_file: 人声音频文件 (audio_vocals.wav)
        instruments_file: 伴奏音频文件 (audio_instruments.wav)
        output_dir (str): 输出目录
        force_bytedance (bool): 是否强制使用火山引擎 TTS

    返回:
        tuple: (输出文本, 生成文件列表)
    """
    return _execution_service.execute_single_with_files(
        module_name="语音合成",
        single_function=generate_all_wavs_under_folder,
        file_map={
            "translation.json": translation_file,      # 翻译后的文本
            "audio_vocals.wav": vocals_file,           # 人声参考音频
            "audio_instruments.wav": instruments_file, # 背景音乐音频
        },
        output_dir=output_dir,
        force_bytedance=force_bytedance
    )


def generate_info_wrapper(summary_file, info_file, thumbnail_file, output_dir):
    """
    信息生成模块的包装函数

    封装 generate_all_info_under_folder 调用，生成视频上传所需信息。

    参数:
        summary_file: 摘要文件 (summary.json)
        info_file: 视频信息文件 (download.info.json)
        thumbnail_file: 缩略图文件 (download.jpg/png)
        output_dir (str): 输出目录

    返回:
        tuple: (输出文本, 生成文件列表)
    """
    return _execution_service.execute_single_with_files(
        module_name="信息生成",
        single_function=generate_all_info_under_folder,
        file_map={
            "summary.json": summary_file,              # 视频摘要
            "download.info.json": info_file,           # 视频元信息
            "download.jpg": thumbnail_file,            # 视频缩略图
        },
        output_dir=output_dir
    )


def upload_bilibili_wrapper(video_file, summary_file, cover_file, info_file):
    """
    B站上传模块的包装函数

    封装 upload_all_videos_under_folder 调用，将视频上传到 Bilibili。

    参数:
        video_file: 合成视频文件 (video.mp4)
        summary_file: 摘要文件 (summary.json)
        cover_file: 封面文件 (video.png)
        info_file: 视频信息文件 (download.info.json)

    返回:
        tuple: (输出文本, 生成文件列表)
    """
    return _execution_service.execute_single_with_files(
        module_name="B站上传",
        single_function=upload_all_videos_under_folder,
        file_map={
            "video.mp4": video_file,                   # 最终视频
            "summary.json": summary_file,              # 视频摘要
            "video.png": cover_file,                   # 视频封面
            "download.info.json": info_file,           # 视频元信息
        }
    )


def update_folder_display(folder_text, select_files):
    """
    更新目录显示的回调函数

    合并文本输入和文件选择器中的目录列表，并格式化显示。

    参数:
        folder_text (str): 文本框中输入的目录路径（每行一个）
        select_files (list): 通过文件选择器选择的文件列表

    返回:
        str: 格式化后的目录列表文本
    """
    # 合并两种来源的目录列表
    folders = _file_utils.merge_folder_lists(folder_text, select_files)
    # 格式化显示，如果空则显示提示
    return "\n".join(folders) if folders else "未选择任何目录"


# ============================================================================ #
# Gradio 界面构建
# 使用 gr.Blocks 构建多标签页的 Web 界面
# ============================================================================ #
with gr.Blocks(title='YouDub') as app:
    # 创建主标签页容器
    with gr.Tabs():
        # ==================================================================== #
        # 标签页 1：设置
        # 管理所有配置项：API Key、模型参数、服务端点等
        # ==================================================================== #
        with gr.Tab('设置'):
            # 快速入门折叠面板：首次使用时展示基本指引
            with gr.Accordion("快速入门", open=False):
                gr.Markdown(
                    "### 快速入门指南\n"
                    "1. 运行 `setup_windows.bat` 安装依赖\n"
                    "2. 复制 `.env.example` 为 `.env` 并填写配置\n"
                    "3. 在本页面填写必要配置并保存\n"
                    "4. 运行 `run_windows.bat` 启动应用\n"
                    "5. 在「全自动」页面输入视频链接开始处理"
                )
            # 页面顶部说明
            gr.Markdown("配置 YouDub 所需的各项参数，带 * 的为必填项")
            # 第一行：网络状态和配置状态并排显示
            with gr.Row():
                network_display = gr.Textbox(label="网络状态", value=_format_network_status(), interactive=False, lines=6, scale=1)
                status_display = gr.Textbox(label="配置状态", value=_format_status(), interactive=False, lines=6, scale=1)
            # 保存结果消息文本框
            save_result = gr.Textbox(label="保存结果")

            # ---- OpenAI 翻译服务设置 ----
            with gr.Accordion("OpenAI 翻译服务", open=True):
                openai_api_key = gr.Textbox(label='OpenAI API Key *', type='password',
                    value=_cfg.get('OPENAI_API_KEY', ''),
                    info='用于翻译功能，获取方式：https://platform.openai.com/api-keys')
                openai_api_base = gr.Textbox(label='OpenAI API Base',
                    value=_cfg.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                    info='OpenAI API 基础 URL，默认值即可，如需使用代理请修改')
                model_name = gr.Textbox(label='Model Name',
                    value=_cfg.get('MODEL_NAME', 'gpt-3.5-turbo'),
                    info='使用的 OpenAI 模型，推荐使用 gpt-3.5-turbo 或 gpt-4')
                with gr.Row():
                    temperature = gr.Slider(label='Temperature', minimum=0.0, maximum=2.0, step=0.1,
                        value=float(_cfg.get('OPENAI_API_TEMPERATURE', 1.0)),
                        info='控制输出随机性，值越高越随机，0 为确定性输出')
                    top_p = gr.Slider(label='Top P', minimum=0.0, maximum=1.0, step=0.05,
                        value=float(_cfg.get('OPENAI_API_TOP_P', 1.0)),
                        info='核采样参数，与 Temperature 建议只修改一个')
                max_tokens = gr.Textbox(label='Max Tokens',
                    value=_cfg.get('OPENAI_API_MAX_TOKENS', ''),
                    info='最大输出 Token 数，留空表示不限制')
                extra_body = gr.Code(label='Extra Body (JSON 格式)', language='json',
                    value=_cfg.get('OPENAI_API_EXTRA_BODY', ''))
                translation_system_prompt = gr.Textbox(
                    label='Translation System Prompt',
                    value=_cfg.get('TRANSLATION_SYSTEM_PROMPT', ''),
                    lines=4,
                    info='逐句翻译的 system 提示词。支持 {info} 和 {target_language} 占位符。留空使用默认。')
                translation_fewshot = gr.Code(
                    label='Translation Few-shot Examples (JSON 数组格式，每项需包含 role 和 content 字段。留空使用默认)',
                    value=_cfg.get('TRANSLATION_FEWSHOT_EXAMPLES', ''),
                    language='json',
                    lines=8)
                summary_system_prompt = gr.Textbox(
                    label='Summary System Prompt',
                    value=_cfg.get('SUMMARY_SYSTEM_PROMPT', ''),
                    lines=3,
                    info='视频摘要的 system 提示词。留空使用默认。')
                summary_translation_prompt = gr.Textbox(
                    label='Summary Translation Prompt',
                    value=_cfg.get('SUMMARY_TRANSLATION_PROMPT', ''),
                    lines=3,
                    info='摘要翻译的 system 提示词。支持 {target_language} 占位符。留空使用默认。')

            # ---- HuggingFace 设置 ----
            with gr.Accordion("HuggingFace 设置", open=True):
                hf_token = gr.Textbox(label='HuggingFace Token *', type='password',
                    value=_cfg.get('HF_TOKEN', ''),
                    info='用于说话者分离和模型下载。获取方式：https://huggingface.co/settings/tokens')
                hf_endpoint = gr.Textbox(label='HuggingFace Endpoint',
                    value=_cfg.get('HF_ENDPOINT', ''),
                    info='例如：https://hf-mirror.com，用于加速模型下载')

            # ---- 语音合成设置 ----
            with gr.Accordion("语音合成", open=True):
                bytedance_appid = gr.Textbox(label='Bytedance App ID *',
                    value=_cfg.get('BYTEDANCE_APPID', ''),
                    info='火山引擎 TTS 服务的 App ID，获取方式：https://console.volcengine.com/')
                bytedance_access_token = gr.Textbox(label='Bytedance Access Token *', type='password',
                    value=_cfg.get('BYTEDANCE_ACCESS_TOKEN', ''),
                    info='火山引擎 TTS 服务的 Access Token')
                with gr.Row():
                    tts_stretch_min = gr.Slider(label='TTS 最小拉伸速度', minimum=0.3, maximum=1.0, step=0.05,
                        value=float(_cfg.get('TTS_STRETCH_MIN_SPEED', 0.6)),
                        info='TTS 音频拉伸的最小速度因子，值越小越慢')
                    tts_stretch_max = gr.Slider(label='TTS 最大拉伸速度', minimum=1.0, maximum=2.0, step=0.05,
                        value=float(_cfg.get('TTS_STRETCH_MAX_SPEED', 1.3)),
                        info='TTS 音频拉伸的最大速度因子，值越大越快')
                # IndexTTS 设置折叠面板
                with gr.Accordion("IndexTTS 设置", open=False):
                    indextts_model_dir = gr.Textbox(label='IndexTTS 模型目录',
                        value=_cfg.get('INDEXTTS_MODEL_DIR', 'models/index-tts'),
                        info='IndexTTS-2 模型文件所在目录路径（相对于项目根目录或绝对路径）')
                    indextts_use_fp16 = gr.Checkbox(label='FP16 推理',
                        value=_cfg.get('INDEXTTS_USE_FP16', True),
                        info='启用半精度推理，降低显存占用，轻微损失质量')
                    indextts_use_deepspeed = gr.Checkbox(label='DeepSpeed 加速',
                        value=_cfg.get('INDEXTTS_USE_DEEPSPEED', False),
                        info='启用 DeepSpeed 加速推理（部分系统可能不兼容）')

            # ---- B站上传设置 ----
            with gr.Accordion("B站上传", open=True):
                bili_sessdata = gr.Textbox(label='BiliBili SESSDATA *', type='password',
                    value=_cfg.get('BILI_SESSDATA', ''),
                    info='B站上传功能所需的 SESSDATA，从浏览器 Cookie 中获取')
                bili_bili_jct = gr.Textbox(label='BiliBili bili_jct *', type='password',
                    value=_cfg.get('BILI_BILI_JCT', ''),
                    info='B站上传功能所需的 bili_jct，从浏览器 Cookie 中获取')
                bili_base64 = gr.Textbox(label='BiliBili Cover Base64',
                    value=_cfg.get('BILI_BASE64', ''),
                    info='B站视频封面的 Base64 编码，可选')

            # ---- 包管理和系统设置 ----
            with gr.Accordion("包管理 & 系统设置", open=False):
                pip_index_url = gr.Textbox(label='PyPI 镜像源',
                    value=_cfg.get('PIP_INDEX_URL', ''),
                    info='例如：https://pypi.tuna.tsinghua.edu.cn/simple，用于加速 Python 包安装')
                download_timeout = gr.Slider(label='下载超时 (秒)', minimum=30, maximum=600, step=10,
                    value=int(_cfg.get('DOWNLOAD_TIMEOUT', 120)),
                    info='视频下载和模型下载的超时时间')
                ffmpeg_path = gr.Textbox(label='FFmpeg 路径',
                    value=_cfg.get('FFMPEG_PATH', ''),
                    placeholder='留空则自动检测',
                    info='FFmpeg 可执行文件路径，留空则使用系统 PATH 或项目内置 FFmpeg')

            # ---- 模型管理 ----
            with gr.Accordion("模型管理", open=True):
                gr.Markdown(
                    "管理 YouDub 全流程所需的本地 AI 模型。首次使用前请先下载所需模型。\n\n"
                    "需要 HF_TOKEN 的模型需先在「HuggingFace 设置」中设置 HuggingFace Token。\n\n"
                    "如需使用镜像加速，请在「HuggingFace 设置」中设置 HuggingFace Endpoint（例如 https://hf-mirror.com），保存后再下载模型。"
                )
                # 显示所有模型的下载状态
                model_status_display = gr.Textbox(label="模型状态", value=_format_model_status_ui(), interactive=False, lines=12)
                with gr.Row():
                    refresh_model_btn = gr.Button("🔄 刷新模型状态", variant="secondary")
                    download_all_btn = gr.Button("⬇️ 下载全部缺失模型", variant="primary")
                download_result = gr.Textbox(label="下载结果")
                # 刷新模型状态按钮的事件绑定
                refresh_model_btn.click(
                    fn=_refresh_model_status,
                    inputs=[],
                    outputs=[model_status_display]
                )
                # 下载全部缺失模型按钮的事件绑定
                download_all_btn.click(
                    fn=_download_all_missing_models,
                    inputs=[],
                    outputs=[download_result]
                ).then(
                    fn=_refresh_model_status,
                    inputs=[],
                    outputs=[model_status_display]
                )

            # ---- 保存配置按钮 ----
            save_btn = gr.Button("保存配置", variant="primary")
            save_btn.click(
                fn=save_settings,
                inputs=[openai_api_key, openai_api_base, model_name, temperature, top_p, max_tokens, extra_body,
                        translation_system_prompt, translation_fewshot, summary_system_prompt, summary_translation_prompt,
                        hf_token, hf_endpoint, bytedance_appid, bytedance_access_token,
                        tts_stretch_min, tts_stretch_max,
                        indextts_model_dir, indextts_use_fp16, indextts_use_deepspeed,
                        bili_sessdata, bili_bili_jct, bili_base64,
                        pip_index_url, download_timeout, ffmpeg_path],
                outputs=[save_result, status_display]
            )

        # ==================================================================== #
        # 标签页 2：下载视频
        # 支持从 URL 下载和从本地导入两种方式
        # ==================================================================== #
        with gr.Tab('下载视频'):
            gr.Markdown("从视频平台下载视频，支持单个视频、播放列表和频道")
            # 使用内部标签页区分 URL 下载和本地导入
            with gr.Tabs():
                # ---- 从 URL 下载子标签页 ----
                with gr.Tab('从 URL 下载'):
                    gr.Markdown("输入视频、播放列表或频道链接下载视频")
                    with gr.Column():
                        url_input = gr.Textbox(label='Video URL', placeholder='Video or Playlist or Channel URL',
                                              value='https://www.bilibili.com/list/1263732318')
                        url_folder = gr.Textbox(label='Output Folder', value='videos')
                        url_resolution = gr.Radio(RESOLUTION_CHOICES, label='Resolution', value='1080p')
                        url_num_videos = gr.Slider(minimum=1, maximum=100, step=1, label='Number of videos to download', value=5)
                        url_output = gr.Textbox(label='输出')
                        url_btn = gr.Button("开始下载", variant="primary")
                        # 下载按钮事件：调用下载函数并包装日志
                        url_btn.click(
                            fn=lambda url, folder, resolution, num_videos: _execution_service.wrap_with_logs(download_from_url, url, folder, resolution, num_videos),
                            inputs=[url_input, url_folder, url_resolution, url_num_videos],
                            outputs=url_output
                        )

                # ---- 从本地导入子标签页 ----
                with gr.Tab('从本地导入'):
                    gr.Markdown("导入本地视频文件到工作目录，以便后续处理")
                    with gr.Column():
                        local_files = gr.File(label='本地视频文件', file_count='multiple',
                                             type='filepath')
                        local_folder = gr.Textbox(label='Output Folder', value='videos',
                                                 info='视频文件将被复制到此文件夹的子目录中')
                        with gr.Accordion("可选元数据（留空则自动生成）", open=False):
                            local_title = gr.Textbox(label='自定义标题（仅适用于单文件）',
                                                    placeholder='留空则使用文件名')
                            local_uploader = gr.Textbox(label='上传者名称',
                                                       placeholder='留空则默认为 Local')
                            local_upload_date = gr.Textbox(label='上传日期（YYYYMMDD 格式）',
                                                          placeholder='留空则使用文件修改日期')
                        local_output = gr.Textbox(label='输出')
                        local_btn = gr.Button("开始导入", variant="primary")
                        # 导入按钮事件
                        local_btn.click(
                            fn=import_local_videos_wrapper,
                            inputs=[local_files, local_folder, local_title, local_uploader, local_upload_date],
                            outputs=local_output
                        )

        # ==================================================================== #
        # 标签页 3：人声分离
        # 使用 Demucs 模型分离视频中的人声和伴奏
        # ==================================================================== #
        with gr.Tab('人声分离'):
            gr.Markdown("使用 Demucs 模型将视频中的人声和伴奏分离。选择视频文件，系统将自动创建临时目录并处理。")
            # 视频文件选择组件（支持多文件上传）
            dm_files = gr.File(label='选择视频文件', file_count='multiple', type='filepath',
                               file_types=['.mp4', '.avi', '.mkv', '.mov', '.flv'])
            # 输出目录选择区域：提供预设按钮和下拉选择
            gr.Markdown("**输出目录选择**")
            with gr.Row():
                dm_btn_output = gr.Button("📁 output", variant="primary")
                dm_btn_videos = gr.Button("🎬 videos")
                dm_btn_results = gr.Button("📊 results")
                dm_btn_refresh = gr.Button("🔄 刷新")
            dm_dir_dropdown = gr.Dropdown(
                label='或选择其他目录',
                choices=get_default_output_dirs(),
                value='output',
                allow_custom_value=False
            )
            dm_output_dir = gr.State('output')  # 隐藏状态，存储最终的输出目录值
            # 预设按钮和下拉选择的事件绑定
            # 注意：每个按钮需要同时更新下拉选择框和隐藏状态
            dm_btn_output.click(lambda: ('output', 'output'), outputs=[dm_dir_dropdown, dm_output_dir])
            dm_btn_videos.click(lambda: ('videos', 'videos'), outputs=[dm_dir_dropdown, dm_output_dir])
            dm_btn_results.click(lambda: ('results', 'results'), outputs=[dm_dir_dropdown, dm_output_dir])
            dm_btn_refresh.click(get_default_output_dirs, outputs=dm_dir_dropdown)
            dm_dir_dropdown.change(lambda x: x, inputs=dm_dir_dropdown, outputs=dm_output_dir)

            # Demucs 模型选择
            dm_model = gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'],
                        label='Model', value='htdemucs_ft')
            # 高级设置折叠面板
            with gr.Accordion("高级设置", open=False):
                dm_device = gr.Radio(['auto', 'cuda', 'cpu'], label='Device', value='auto')
                dm_progress = gr.Checkbox(label='Progress Bar in Console', value=True)
                dm_shifts = gr.Slider(minimum=0, maximum=10, step=1, label='Number of shifts', value=5)
                dm_segment = gr.Slider(minimum=5, maximum=30, step=1, label='内部段长 (秒)', value=10, info='Demucs 内部分段推理长度，越小显存占用越低')
                dm_max_chunk = gr.Slider(minimum=120, maximum=1800, step=60, label='最大分块 (秒)', value=600, info='长音频外部分块大小，越小内存占用越低，0表示不限制')
            with gr.Row():
                dm_output = gr.Textbox(label='输出', scale=3)
                dm_output_files = gr.File(label='生成文件', scale=2)
            dm_btn = gr.Button("开始分离", variant="primary")
            # 开始分离按钮的事件绑定
            dm_btn.click(
                fn=demucs_wrapper,
                inputs=[dm_files, dm_output_dir, dm_model, dm_device, dm_progress, dm_shifts, dm_segment, dm_max_chunk],
                outputs=[dm_output, dm_output_files]
            )

        # ==================================================================== #
        # 标签页 4：语音识别
        # 使用 WhisperX 模型将语音转换为文字
        # ==================================================================== #
        with gr.Tab('语音识别'):
            gr.Markdown("使用 WhisperX 模型将语音转换为文字，支持说话者分离。选择人声音频文件，系统将自动创建临时目录并处理。")
            ws_files = gr.File(label='选择人声音频文件', file_count='multiple', type='filepath',
                               file_types=['.wav', '.mp3', '.flac', '.m4a', '.aac'])
            # 输出目录选择
            gr.Markdown("**输出目录选择**")
            with gr.Row():
                ws_btn_output = gr.Button("📁 output", variant="primary")
                ws_btn_videos = gr.Button("🎬 videos")
                ws_btn_results = gr.Button("📊 results")
                ws_btn_refresh = gr.Button("🔄 刷新")
            ws_dir_dropdown = gr.Dropdown(
                label='或选择其他目录',
                choices=get_default_output_dirs(),
                value='output',
                allow_custom_value=False
            )
            ws_output_dir = gr.State('output')
            # 输出目录按钮事件
            ws_btn_output.click(lambda: ('output', 'output'), outputs=[ws_dir_dropdown, ws_output_dir])
            ws_btn_videos.click(lambda: ('videos', 'videos'), outputs=[ws_dir_dropdown, ws_output_dir])
            ws_btn_results.click(lambda: ('results', 'results'), outputs=[ws_dir_dropdown, ws_output_dir])
            ws_btn_refresh.click(get_default_output_dirs, outputs=ws_dir_dropdown)
            ws_dir_dropdown.change(lambda x: x, inputs=ws_dir_dropdown, outputs=ws_output_dir)

            ws_model = gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label='Model', value='large')
            ws_diarization = gr.Checkbox(label='Diarization', value=True,
                         info='启用说话者分离，区分不同说话人')
            # 高级设置
            with gr.Accordion("高级设置", open=False):
                ws_download_root = gr.Textbox(label='Download Root', value='models/ASR/whisper')
                ws_device = gr.Radio(['auto', 'cuda', 'cpu'], label='Device', value='auto')
                ws_batch_size = gr.Slider(minimum=1, maximum=128, step=1, label='Batch Size', value=32)
                ws_min_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   label='Min Speakers', value=None)
                ws_max_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   label='Max Speakers', value=None)
            with gr.Row():
                ws_output = gr.Textbox(label='输出', scale=3)
                ws_output_files = gr.File(label='生成文件', scale=2)
            ws_btn = gr.Button("开始识别", variant="primary")
            ws_btn.click(
                fn=whisper_wrapper,
                inputs=[ws_files, ws_output_dir, ws_model, ws_diarization, ws_download_root, ws_device, ws_batch_size, ws_min_speakers, ws_max_speakers],
                outputs=[ws_output, ws_output_files]
            )

        # ==================================================================== #
        # 标签页 5：字幕翻译
        # 使用大语言模型将字幕翻译为目标语言
        # ==================================================================== #
        with gr.Tab('字幕翻译'):
            gr.Markdown("使用大语言模型将字幕翻译为目标语言。选择识别结果和视频信息文件，系统将自动创建临时目录并处理。")
            tl_transcript_file = gr.File(label='识别结果文件 (transcript.json)', file_count='single', type='filepath',
                                         file_types=['.json'])
            tl_info_file = gr.File(label='视频信息文件 (download.info.json)', file_count='single', type='filepath',
                                   file_types=['.json'])
            # 输出目录选择
            gr.Markdown("**输出目录选择**")
            with gr.Row():
                tl_btn_output = gr.Button("📁 output", variant="primary")
                tl_btn_videos = gr.Button("🎬 videos")
                tl_btn_results = gr.Button("📊 results")
                tl_btn_refresh = gr.Button("🔄 刷新")
            tl_dir_dropdown = gr.Dropdown(
                label='或选择其他目录',
                choices=get_default_output_dirs(),
                value='output',
                allow_custom_value=False
            )
            tl_output_dir = gr.State('output')
            tl_btn_output.click(lambda: ('output', 'output'), outputs=[tl_dir_dropdown, tl_output_dir])
            tl_btn_videos.click(lambda: ('videos', 'videos'), outputs=[tl_dir_dropdown, tl_output_dir])
            tl_btn_results.click(lambda: ('results', 'results'), outputs=[tl_dir_dropdown, tl_output_dir])
            tl_btn_refresh.click(get_default_output_dirs, outputs=tl_dir_dropdown)
            tl_dir_dropdown.change(lambda x: x, inputs=tl_dir_dropdown, outputs=tl_output_dir)

            tl_lang = gr.Dropdown(['简体中文', '繁体中文', 'English', 'Deutsch', 'Français', 'русский'],
                        label='Target Language', value='简体中文')
            with gr.Row():
                tl_output = gr.Textbox(label='输出', scale=3)
                tl_output_files = gr.File(label='生成文件', scale=2)
            tl_btn = gr.Button("开始翻译", variant="primary")
            tl_btn.click(
                fn=translation_wrapper,
                inputs=[tl_transcript_file, tl_info_file, tl_output_dir, tl_lang],
                outputs=[tl_output, tl_output_files]
            )

        # ==================================================================== #
        # 标签页 6：语音合成
        # 使用 TTS 将翻译后的文字转为语音
        # ==================================================================== #
        with gr.Tab('语音合成'):
            gr.Markdown("使用 TTS 模型将翻译后的文字生成语音。选择翻译结果和音频文件，系统将自动创建临时目录并处理。")
            tts_translation_file = gr.File(label='翻译结果文件 (translation.json)', file_count='single', type='filepath',
                                           file_types=['.json'])
            tts_vocals_file = gr.File(label='人声音频文件 (audio_vocals.wav)', file_count='single', type='filepath',
                                      file_types=['.wav', '.mp3', '.flac', '.m4a', '.aac'])
            tts_instruments_file = gr.File(label='伴奏音频文件 (audio_instruments.wav)', file_count='single', type='filepath',
                                           file_types=['.wav', '.mp3', '.flac', '.m4a', '.aac'])
            # 输出目录选择
            gr.Markdown("**输出目录选择**")
            with gr.Row():
                tts_btn_output = gr.Button("📁 output", variant="primary")
                tts_btn_videos = gr.Button("🎬 videos")
                tts_btn_results = gr.Button("📊 results")
                tts_btn_refresh = gr.Button("🔄 刷新")
            tts_dir_dropdown = gr.Dropdown(
                label='或选择其他目录',
                choices=get_default_output_dirs(),
                value='output',
                allow_custom_value=False
            )
            tts_output_dir = gr.State('output')
            tts_btn_output.click(lambda: ('output', 'output'), outputs=[tts_dir_dropdown, tts_output_dir])
            tts_btn_videos.click(lambda: ('videos', 'videos'), outputs=[tts_dir_dropdown, tts_output_dir])
            tts_btn_results.click(lambda: ('results', 'results'), outputs=[tts_dir_dropdown, tts_output_dir])
            tts_btn_refresh.click(get_default_output_dirs, outputs=tts_dir_dropdown)
            tts_dir_dropdown.change(lambda x: x, inputs=tts_dir_dropdown, outputs=tts_output_dir)

            tts_force_bytedance = gr.Checkbox(label='Force Bytedance', value=False,
                    info='强制使用火山引擎 TTS，而非 IndexTTS 声音克隆')
            with gr.Row():
                tts_output = gr.Textbox(label='输出', scale=3)
                tts_output_files = gr.File(label='生成文件', scale=2)
            tts_btn = gr.Button("开始合成", variant="primary")
            tts_btn.click(
                fn=tts_wrapper,
                inputs=[tts_translation_file, tts_vocals_file, tts_instruments_file, tts_output_dir, tts_force_bytedance],
                outputs=[tts_output, tts_output_files]
            )

        # ==================================================================== #
        # 标签页 7：视频合成
        # 将配音、字幕与原视频合成为最终视频
        # ==================================================================== #
        with gr.Tab('视频合成'):
            gr.Markdown("将配音、字幕与原视频合成为最终视频。选择视频、翻译结果和合成音频文件，系统将自动创建临时目录并处理。")
            sv_video_file = gr.File(label='视频文件 (download.mp4)', file_count='single', type='filepath',
                                    file_types=['.mp4', '.avi', '.mkv', '.mov', '.flv'])
            sv_translation_file = gr.File(label='翻译结果文件 (translation.json)', file_count='single', type='filepath',
                                          file_types=['.json'])
            sv_audio_combined_file = gr.File(label='合成音频文件 (audio_combined.wav)', file_count='single', type='filepath',
                                             file_types=['.wav', '.mp3', '.flac', '.m4a', '.aac'])
            # 输出目录选择
            gr.Markdown("**输出目录选择**")
            with gr.Row():
                sv_btn_output = gr.Button("📁 output", variant="primary")
                sv_btn_videos = gr.Button("🎬 videos")
                sv_btn_results = gr.Button("📊 results")
                sv_btn_refresh = gr.Button("🔄 刷新")
            sv_dir_dropdown = gr.Dropdown(
                label='或选择其他目录',
                choices=get_default_output_dirs(),
                value='output',
                allow_custom_value=False
            )
            sv_output_dir = gr.State('output')
            sv_btn_output.click(lambda: ('output', 'output'), outputs=[sv_dir_dropdown, sv_output_dir])
            sv_btn_videos.click(lambda: ('videos', 'videos'), outputs=[sv_dir_dropdown, sv_output_dir])
            sv_btn_results.click(lambda: ('results', 'results'), outputs=[sv_dir_dropdown, sv_output_dir])
            sv_btn_refresh.click(get_default_output_dirs, outputs=sv_dir_dropdown)
            sv_dir_dropdown.change(lambda x: x, inputs=sv_dir_dropdown, outputs=sv_output_dir)

            sv_subtitles = gr.Checkbox(label='Subtitles', value=True)
            sv_use_original_audio = gr.Checkbox(label='使用原视频音轨（不配音）', value=False,
                info='勾选后将跳过 TTS 配音，使用原视频音轨合成。此时无需选择合成音频文件。')
            sv_resolution = gr.Radio(RESOLUTION_CHOICES, label='Resolution', value='1080p')
            with gr.Accordion("高级设置", open=False):
                sv_speed_up = gr.Slider(minimum=0.5, maximum=2, step=0.05, label='Speed Up', value=1.05)
                sv_fps = gr.Slider(minimum=1, maximum=60, step=1, label='FPS', value=30)
            with gr.Row():
                sv_output = gr.Textbox(label='输出', scale=3)
                sv_output_files = gr.File(label='生成文件', scale=2)
            sv_btn = gr.Button("开始合成", variant="primary")
            sv_btn.click(
                fn=synthesize_wrapper,
                inputs=[sv_video_file, sv_translation_file, sv_audio_combined_file, sv_output_dir, sv_subtitles, sv_use_original_audio, sv_resolution, sv_speed_up, sv_fps],
                outputs=[sv_output, sv_output_files]
            )

        # ==================================================================== #
        # 标签页 8：信息生成
        # 生成视频上传所需的信息（标题、描述、标签等）
        # ==================================================================== #
        with gr.Tab('信息生成'):
            gr.Markdown("生成视频上传所需信息。选择摘要、视频信息和缩略图文件，系统将自动创建临时目录并处理。")
            gi_summary_file = gr.File(label='摘要文件 (summary.json)', file_count='single', type='filepath',
                                      file_types=['.json'])
            gi_info_file = gr.File(label='视频信息文件 (download.info.json)', file_count='single', type='filepath',
                                   file_types=['.json'])
            gi_thumbnail_file = gr.File(label='缩略图文件 (download.jpg/png)', file_count='single', type='filepath',
                                        file_types=['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
            # 输出目录选择
            gr.Markdown("**输出目录选择**")
            with gr.Row():
                gi_btn_output = gr.Button("📁 output", variant="primary")
                gi_btn_videos = gr.Button("🎬 videos")
                gi_btn_results = gr.Button("📊 results")
                gi_btn_refresh = gr.Button("🔄 刷新")
            gi_dir_dropdown = gr.Dropdown(
                label='或选择其他目录',
                choices=get_default_output_dirs(),
                value='output',
                allow_custom_value=False
            )
            gi_output_dir = gr.State('output')
            gi_btn_output.click(lambda: ('output', 'output'), outputs=[gi_dir_dropdown, gi_output_dir])
            gi_btn_videos.click(lambda: ('videos', 'videos'), outputs=[gi_dir_dropdown, gi_output_dir])
            gi_btn_results.click(lambda: ('results', 'results'), outputs=[gi_dir_dropdown, gi_output_dir])
            gi_btn_refresh.click(get_default_output_dirs, outputs=gi_dir_dropdown)
            gi_dir_dropdown.change(lambda x: x, inputs=gi_dir_dropdown, outputs=gi_output_dir)
            with gr.Row():
                gi_output = gr.Textbox(label='输出', scale=3)
                gi_output_files = gr.File(label='生成文件', scale=2)
            gi_btn = gr.Button("开始生成", variant="primary")
            gi_btn.click(
                fn=generate_info_wrapper,
                inputs=[gi_summary_file, gi_info_file, gi_thumbnail_file, gi_output_dir],
                outputs=[gi_output, gi_output_files]
            )

        # ==================================================================== #
        # 标签页 9：上传B站
        # 将合成好的视频上传到 Bilibili 平台
        # ==================================================================== #
        with gr.Tab('上传B站'):
            gr.Markdown("将合成好的视频上传到 Bilibili。选择视频、摘要、封面和信息文件，系统将自动创建临时目录并处理。")
            ub_video_file = gr.File(label='合成视频文件 (video.mp4)', file_count='single', type='filepath',
                                    file_types=['.mp4', '.avi', '.mkv', '.mov'])
            ub_summary_file = gr.File(label='摘要文件 (summary.json)', file_count='single', type='filepath',
                                      file_types=['.json'])
            ub_cover_file = gr.File(label='封面文件 (video.png)', file_count='single', type='filepath',
                                    file_types=['.png', '.jpg', '.jpeg'])
            ub_info_file = gr.File(label='视频信息文件 (download.info.json)', file_count='single', type='filepath',
                                   file_types=['.json'])
            with gr.Row():
                ub_output = gr.Textbox(label='输出', scale=3)
                ub_output_files = gr.File(label='生成文件', scale=2)
            ub_btn = gr.Button("开始上传", variant="primary")
            ub_btn.click(
                fn=upload_bilibili_wrapper,
                inputs=[ub_video_file, ub_summary_file, ub_cover_file, ub_info_file],
                outputs=[ub_output, ub_output_files]
            )

        # ==================================================================== #
        # 标签页 10：全自动
        # 一键完成从视频下载到配音合成的全流程
        # ==================================================================== #
        with gr.Tab('全自动'):
            gr.Markdown("一键完成从视频下载到配音合成的全流程，支持选择性执行特定模块")
            # 输入模式选择：URL 下载或本地文件
            de_input_mode = gr.Radio(['URL 下载', '本地文件'], label='输入模式', value='URL 下载',
                                     info='选择从网络下载视频或使用本地视频文件')
            de_url = gr.Textbox(label='Video URL', placeholder='Video or Playlist or Channel URL',
                        value='https://www.bilibili.com/list/1263732318',
                        info='支持视频、播放列表或频道链接', visible=True)
            de_local_files = gr.File(label='本地视频文件', file_count='multiple',
                                    type='filepath', visible=False)
            de_root_folder = gr.Textbox(label='Root Folder', value='videos',
                         info='视频文件的根目录')
            de_num_videos = gr.Slider(minimum=1, maximum=100, step=1, label='Number of videos to download', value=5,
                          info='要下载的视频数量（仅 URL 模式有效）')
            de_resolution = gr.Radio(RESOLUTION_CHOICES, label='Resolution', value='1080p',
                         info='下载和合成视频的分辨率')
            de_translation_target_language = gr.Dropdown(['简体中文', '繁体中文', 'English', 'Deutsch', 'Français', 'русский'],
                         label='Translation Target Language', value='简体中文',
                         info='翻译的目标语言')
            de_subtitles = gr.Checkbox(label='Subtitles', value=True,
                       info='是否在合成视频中添加字幕')
            de_use_original_audio = gr.Checkbox(label='使用原视频音轨（不配音）', value=False,
                       info='勾选后将跳过 TTS 配音步骤，使用原视频音轨合成视频')
            de_auto_upload = gr.Checkbox(label='Auto Upload Video', value=False,
                         info='自动上传到B站（建议先确认效果再开启）')

            # ---- 模块选择折叠面板 ----
            with gr.Accordion("模块选择", open=True):
                gr.Markdown("选择要执行的处理模块。取消勾选「启用模块选择」则执行完整流程。注意：请确保所选模块的输入文件已存在，缺失输入文件的模块将被跳过。")
                de_use_module_selection = gr.Checkbox(label='启用模块选择', value=False,
                    info='开启后可选择性执行特定模块（可单选/多选），关闭则执行完整流程')
                de_selected_modules = gr.CheckboxGroup(
                    choices=MODULE_CHOICES,
                    value=DEFAULT_MODULES,
                    label='选择要执行的模块',
                    info='仅执行选中的模块，不会自动添加其他模块。请确保输入文件已存在。'
                )
                de_skip_completed = gr.Checkbox(label='跳过已完成步骤', value=True,
                    info='检测到模块输出文件已存在时跳过执行')
                # 显示模块依赖排序后的执行顺序
                de_execution_order = gr.Textbox(label='执行顺序预览', value=format_execution_order(DEFAULT_MODULES),
                    interactive=False, lines=10)

                with gr.Row():
                    select_all_btn = gr.Button("全选", size="sm")
                    deselect_all_btn = gr.Button("全不选", size="sm")

                # 全选/全不选按钮事件
                select_all_btn.click(
                    fn=select_all_modules,
                    outputs=[de_selected_modules, de_execution_order]
                )
                deselect_all_btn.click(
                    fn=deselect_all_modules,
                    outputs=[de_selected_modules, de_execution_order]
                )

                # 模块选择变化时更新执行顺序预览
                de_selected_modules.change(
                    fn=format_execution_order,
                    inputs=[de_selected_modules],
                    outputs=[de_execution_order]
                )

            # ---- 手动选择输入文件折叠面板 ----
            with gr.Accordion("手动选择输入文件", open=False):
                gr.Markdown("为各模块手动选择输入文件。选择的文件将在执行时自动复制到处理文件夹。")

                # 状态存储：保存所有文件选择的字典
                de_selected_files = gr.State(value=_file_utils.create_selected_files_dict())
                # 文件选择状态的文本显示
                de_selected_files_status = gr.Textbox(label="文件选择状态", value=_file_utils.format_selected_files_status(None),
                    interactive=False, lines=15)

                # 动态生成文件输入组件：遍历所有模块的输入文件要求
                file_inputs = {}
                file_statuses = {}
                for module_id, module_info in MODULES.items():
                    if module_info.get("input_files"):
                        # 每个有输入文件要求的模块创建一个折叠面板
                        with gr.Accordion(f"{module_info['name']} - 输入文件", open=False):
                            for input_file in module_info["input_files"]:
                                # 获取文件标签和格式信息
                                format_info = INPUT_FILE_FORMATS.get(input_file, {})
                                file_label = _file_utils.get_file_label(input_file)
                                with gr.Row():
                                    # 文件选择器和验证状态显示
                                    file_inputs[input_file] = gr.File(
                                        label=file_label,
                                        file_count="single",
                                        type="filepath"
                                    )
                                    file_statuses[input_file] = gr.Textbox(
                                        label=f"{input_file} - 验证状态",
                                        value="未选择文件",
                                        interactive=False,
                                        scale=2
                                    )

                # 为每个文件输入组件绑定变更事件
                for filename, file_input in file_inputs.items():
                    def create_file_callback(fname):
                        """
                        创建文件选择回调函数的闭包工厂

                        每个文件输入组件需要不同回调函数来处理各自的文件验证和状态更新。
                        使用闭包捕获文件名变量，避免循环中的变量捕获问题。

                        参数:
                            fname (str): 文件名（模块的输入文件键名）

                        返回:
                            function: 文件选择回调函数
                        """
                        def callback(filepath, selected_files=None):
                            # 如果 selected_files 为 None，创建一个新的空字典
                            if selected_files is None:
                                selected_files = _file_utils.create_selected_files_dict()

                            if filepath:
                                # 验证文件格式是否符合要求
                                is_valid, msg = _file_utils.validate_file_format(filepath, fname)
                                if is_valid:
                                    # 文件格式正确，保存到选择字典
                                    selected_files[fname] = filepath
                                    status_msg = f"✅ {msg}"
                                else:
                                    # 文件格式不正确，清除选择
                                    selected_files[fname] = None
                                    status_msg = f"❌ {msg}"
                            else:
                                # 没有选择文件（清空选择）
                                selected_files[fname] = None
                                status_msg = "未选择文件"

                            # 返回更新后的选择字典、文件状态文本、总体选择状态
                            return selected_files, status_msg, _file_utils.format_selected_files_status(selected_files)
                        return callback

                    # 绑定事件：文件选择变化时验证并更新状态
                    file_input.change(
                        fn=create_file_callback(filename),
                        inputs=[file_input, de_selected_files],
                        outputs=[de_selected_files, file_statuses[filename], de_selected_files_status]
                    )

                with gr.Row():
                    clear_files_btn = gr.Button("清除所有文件选择", size="sm")

                def clear_file_selections():
                    """
                    清除所有文件选择

                    创建空的选择字典，并重置所有文件输入组件。
                    用于「清除所有文件选择」按钮回调。

                    返回:
                        list: 包含空选择字典、空状态文本、以及所有文件输入组件的 None 值
                    """
                    empty_files = _file_utils.create_selected_files_dict()
                    # 返回顺序：选择字典、状态显示文本、然后是所有文件输入组件的 None
                    return [empty_files, _file_utils.format_selected_files_status(empty_files)] + [None] * len(file_inputs)

                clear_files_btn.click(
                    fn=clear_file_selections,
                    outputs=[de_selected_files, de_selected_files_status] + list(file_inputs.values())
                )

            # ---- 选择处理目录折叠面板 ----
            with gr.Accordion("选择处理目录", open=False):
                gr.Markdown("选择要处理的已有目录（每个目录应包含 download.mp4 文件）。选择目录后，将直接处理这些目录而不进行下载。")
                with gr.Row():
                    de_folder_list_text = gr.Textbox(
                        label='目录路径列表（每行一个路径）',
                        placeholder='D:\\videos\\Uploader\\20250101 My Video\nD:\\videos\\Uploader\\20250102 Another Video',
                        lines=4,
                        scale=3
                    )
                with gr.Row():
                    de_folder_select_files = gr.File(
                        label='从文件选择目录（选择目录中的任意文件即可自动提取目录路径）',
                        file_count='multiple',
                        type='filepath',
                        scale=3
                    )
                with gr.Row():
                    de_folder_list_display = gr.Textbox(
                        label='已识别的目录列表',
                        value='',
                        interactive=False,
                        lines=4,
                        scale=3
                    )
                    de_refresh_folders_btn = gr.Button("刷新目录列表", scale=1)

                # 刷新目录列表按钮事件
                de_refresh_folders_btn.click(
                    fn=update_folder_display,
                    inputs=[de_folder_list_text, de_folder_select_files],
                    outputs=[de_folder_list_display]
                )

            # ---- 高级设置折叠面板 ----
            with gr.Accordion("高级设置", open=False):
                de_demucs_model = gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'],
                            label='Demucs Model', value='htdemucs_ft',
                            info='音频分离模型，htdemucs_ft 效果较好')
                de_demucs_device = gr.Radio(['auto', 'cuda', 'cpu'], label='Demucs Device', value='auto',
                            info='音频分离的计算设备')
                de_shifts = gr.Slider(minimum=0, maximum=10, step=1, label='Number of shifts', value=5,
                         info='音频分离的移位数，越大质量越好但越慢')
                de_demucs_segment = gr.Slider(minimum=5, maximum=30, step=1, label='内部段长 (秒)', value=10,
                         info='Demucs 内部分段推理长度，越小显存占用越低')
                de_demucs_max_chunk = gr.Slider(minimum=120, maximum=1800, step=60, label='最大分块 (秒)', value=600,
                         info='长音频外部分块大小，越小内存占用越低')
                de_whisper_model = gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label='Whisper Model', value='large',
                            info='语音识别模型，large 最准确但最慢')
                de_whisper_batch_size = gr.Slider(minimum=1, maximum=128, step=1, label='Whisper Batch Size', value=32,
                              info='语音识别的批处理大小')
                de_whisper_diarization = gr.Checkbox(label='Whisper Diarization', value=True,
                                 info='启用说话者分离，区分不同说话人')
                de_whisper_min_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               label='Min Speakers', value=None,
                               info='说话者分离的最小说话人数')
                de_whisper_max_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               label='Max Speakers', value=None,
                               info='说话者分离的最大说话人数')
                de_whisper_download_root = gr.Textbox(label='Whisper Download Root', value='models/ASR/whisper',
                               info='Whisper 模型下载/加载路径')
                de_speed_up = gr.Slider(minimum=0.5, maximum=2, step=0.05, label='Speed Up', value=1.05,
                        info='视频加速倍率，1.0 为原速')
                de_fps = gr.Slider(minimum=1, maximum=60, step=1, label='FPS', value=30,
                      info='合成视频的帧率')
                de_max_workers = gr.Slider(minimum=1, maximum=100, step=1, label='Max Workers', value=1,
                          info='并发处理的工作线程数')
                de_max_retries = gr.Slider(minimum=1, maximum=10, step=1, label='Max Retries', value=3,
                          info='失败后的最大重试次数')
                de_force_bytedance = gr.Checkbox(label='Force Bytedance', value=True,
                             info='强制使用火山引擎 TTS，而非 IndexTTS 声音克隆')

            # 全自动流程输出文本框
            de_output = gr.Textbox(label='输出')
            de_btn = gr.Button("开始执行", variant="primary")

            # 输入模式切换事件：URL 下载模式显示 URL 输入框，本地文件模式显示文件上传组件
            de_input_mode.change(
                fn=toggle_input_mode,
                inputs=[de_input_mode],
                outputs=[de_url, de_local_files]
            )

            # 开始执行按钮事件：将所有参数传递给包装函数
            de_btn.click(
                fn=do_everything_wrapper,
                inputs=[de_input_mode, de_url, de_local_files, de_root_folder, de_num_videos, de_resolution, de_translation_target_language,
                        de_subtitles, de_use_original_audio, de_auto_upload, de_demucs_model, de_demucs_device, de_shifts,
                        de_demucs_segment, de_demucs_max_chunk,
                        de_whisper_model, de_whisper_batch_size, de_whisper_diarization,
                        de_whisper_min_speakers, de_whisper_max_speakers, de_whisper_download_root,
                        de_speed_up, de_fps, de_max_workers, de_max_retries, de_force_bytedance,
                        de_selected_modules, de_skip_completed, de_use_module_selection, de_selected_files,
                        de_folder_list_text, de_folder_select_files],
                outputs=de_output
            )


# ============================================================================ #
# 依赖兼容性检查函数
# 检查 Numba 和 NumPy 版本兼容性，防止运行时崩溃
# ============================================================================ #
def _check_dependency_compatibility():
    """
    检查 Numba 和 NumPy 之间的版本兼容性

    Numba 对 NumPy 版本有严格的要求，不同版本的 Numba 支持不同范围的 NumPy 版本。
    如果不兼容，JIT 编译功能会出错，导致某些模块运行失败。

    返回:
        bool: 版本兼容返回 True，不兼容返回 False
    """
    try:
        # 尝试导入 Numba 和 NumPy
        import numba
        import numpy
        # 提取主次版本号
        nv = tuple(int(x) for x in numba.__version__.split('.')[:2])
        npv = tuple(int(x) for x in numpy.__version__.split('.')[:2])
        # Numba 版本到 NumPy 最高支持版本的映射表
        max_np = {
            (0, 60): (2, 0), (0, 61): (2, 1), (0, 62): (2, 1),
            (0, 63): (2, 3), (0, 64): (2, 4), (0, 65): (2, 4),
        }.get(nv, (2, 0))
        if npv > max_np:
            # 版本不兼容，生成错误消息和修复命令
            msg = (
                f"Numba {numba.__version__} 不支持 NumPy {numpy.__version__}，"
                f"最高支持 NumPy {'.'.join(map(str, max_np))}\n"
                f"修复命令: pip install \"numpy<{'.'.join(map(str, max_np))}\" \"numba>=0.63\""
            )
            logger.error(msg)
            logger.error(f"\n{'='*60}")
            logger.error(f"  ❌ 依赖库版本不兼容")
            logger.error(f"  {msg}")
            logger.error(f"{'='*60}\n")
            return False
        logger.info(f"依赖库版本检查通过: Numba {numba.__version__} + NumPy {numpy.__version__}")
        return True
    except ImportError:
        # Numba 或 NumPy 未安装时不影响其他功能
        logger.warning("无法检查 Numba/NumPy 版本兼容性（可能未安装）")
        return True
    except Exception as e:
        logger.debug(f"版本兼容性检查异常: {e}")
        return True


# ============================================================================ #
# 程序入口点
# ============================================================================ #
if __name__ == '__main__':
    # 启动前先检查依赖兼容性
    _check_dependency_compatibility()
    # 启动 Gradio 应用
    # server_port=19876: 指定服务端口
    # inbrowser=True: 自动在默认浏览器中打开页面
    app.launch(server_port=19876, inbrowser=True)
