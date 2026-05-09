import os
os.environ["GRADIO_SERVER_PORT"] = "19876"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import gradio as gr
from youdub.step000_video_downloader import download_from_url, import_local_video
from youdub.step010_demucs_vr import separate_all_audio_under_folder, cleanup_demucs
from youdub.step020_whisperx import transcribe_all_audio_under_folder, cleanup_whisperx
from youdub.step030_translation import translate_all_transcript_under_folder
from youdub.step040_tts import generate_all_wavs_under_folder
from youdub.step050_synthesize_video import synthesize_all_video_under_folder
from youdub.step060_generate_info import generate_all_info_under_folder
from youdub.step070_upload_bilibili import upload_all_videos_under_folder
from youdub.do_everything import do_everything
from youdub.config import load_config, save_config, get_config_status, validate_config, DEFAULT_CONFIG, check_network, get_offline_capabilities, PROJECT_ROOT
from youdub.module_registry import get_module, get_module_dependencies, get_execution_order, resolve_dependencies, MODULES, get_module_input_files
from youdub.module_executor import get_module_with_info
from youdub.model_manager import format_model_status, check_all_models_status, check_model_status, download_model, download_all_models, get_model_info
from youdub.services.error_handler import ErrorHandler
from youdub.services.file_utils import FileUtils, INPUT_FILE_FORMATS
from youdub.services.execution_service import ExecutionService
from loguru import logger
from youdub.log_config import get_log_buffer, clear_log_buffer


_error_handler = ErrorHandler()
_file_utils = FileUtils()
_execution_service = ExecutionService()


def save_settings(openai_api_key, openai_api_base, model_name, temperature, top_p, max_tokens, extra_body,
                  translation_system_prompt, translation_fewshot, summary_system_prompt, summary_translation_prompt,
                  hf_token, hf_endpoint,
                  bytedance_appid, bytedance_access_token,
                  tts_stretch_min, tts_stretch_max,
                  indextts_model_dir, indextts_use_fp16, indextts_use_deepspeed,
                  bili_sessdata, bili_bili_jct, bili_base64,
                  pip_index_url, download_timeout, ffmpeg_path):
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
    save_config(config)
    missing = validate_config()
    if missing:
        features = sorted(set(missing.values()))
        msg = f"配置已保存！以下功能缺少必填项：{', '.join(features)}"
    else:
        msg = "配置已保存！所有必填项均已设置。"
    return msg, _format_status()


def _format_status():
    status = get_config_status()
    lines = []
    special_keys = {"MODELS", "FFMPEG"}
    for key, info in status.items():
        if key in special_keys:
            continue
        if info["required"]:
            mark = "✅" if info["set"] else "❌"
            lines.append(f"{mark} {key} ({info['feature']})")
    if "MODELS" in status:
        model_info = status["MODELS"]
        mark = "✅" if model_info["set"] else "⚠️"
        lines.append(f"{mark} {model_info['message']} ({model_info['feature']})")
    if "FFMPEG" in status:
        ffmpeg_info = status["FFMPEG"]
        mark = "✅" if ffmpeg_info["set"] else "❌"
        lines.append(f"{mark} FFmpeg ({ffmpeg_info['feature']})")
    return "\n".join(lines)


def _format_network_status():
    online = check_network()
    if online:
        return "🟢 在线 - 网络连接正常"
    caps = get_offline_capabilities()
    lines = ["🔴 离线 - 网络不可用", "", "离线可用功能："]
    for f in caps["available"]:
        lines.append(f"  ✅ {f}")
    lines.append("")
    lines.append("离线不可用功能：")
    for f in caps["unavailable"]:
        lines.append(f"  ❌ {f}")
    return "\n".join(lines)


def _format_model_status_ui():
    return format_model_status()


def _download_all_missing_models(progress=gr.Progress()):
    statuses = check_all_models_status()
    missing = [mid for mid, s in statuses.items() if not s["downloaded"]]
    if not missing:
        return "所有模型已下载，无需额外操作"

    lines = []
    total = len(missing)
    progress(0, desc="准备下载模型...")
    for i, model_id in enumerate(missing):
        info = get_model_info(model_id)
        name = info["name"] if info else model_id
        progress((i) / total, desc=f"[{i+1}/{total}] 正在下载: {name}")
        try:
            download_model(model_id)
            lines.append(f"✅ [{i+1}/{total}] {name} 下载完成")
        except ValueError as e:
            lines.append(f"❌ [{i+1}/{total}] {name} 下载失败: {e}")
        except ImportError as e:
            lines.append(f"❌ [{i+1}/{total}] {name} 库未安装: {e}")
        except Exception as e:
            lines.append(f"❌ [{i+1}/{total}] {name} 下载失败: {str(e)}")
    success = sum(1 for l in lines if l.startswith("✅"))
    fail = sum(1 for l in lines if l.startswith("❌"))
    lines.append(f"\n下载完成：成功 {success} 个，失败 {fail} 个")
    progress(1.0, desc="下载完成")
    return "\n".join(lines)


def _refresh_model_status():
    return format_model_status()


def do_everything_wrapper(input_mode, url, local_files, root_folder, num_videos, resolution, translation_target_language, subtitles, use_original_audio, auto_upload_video,
                          demucs_model, demucs_device, shifts, demucs_segment, demucs_max_chunk, whisper_model, whisper_batch_size, whisper_diarization, whisper_min_speakers, whisper_max_speakers, whisper_download_root,
                          speed_up, fps, max_workers, max_retries, force_bytedance, selected_modules, skip_completed, use_module_selection, selected_files,
                          de_folder_list_text, de_folder_select_files):
    local_video_paths = None
    selected_folders = _file_utils.merge_folder_lists(de_folder_list_text, de_folder_select_files)
    
    if selected_folders:
        if input_mode == '本地文件':
            if local_files is None or len(local_files) == 0:
                pass
            else:
                local_video_paths = [f.name if hasattr(f, 'name') else f for f in local_files]
        elif input_mode == 'URL 下载':
            if not url or not url.strip():
                pass
    else:
        if input_mode == '本地文件':
            if local_files is None or len(local_files) == 0:
                return _error_handler.format_error(
                    "未选择视频文件",
                    ["没有选择任何本地视频文件"],
                    ["请点击「上传文件」按钮选择一个或多个视频文件"]
                )
            local_video_paths = [f.name if hasattr(f, 'name') else f for f in local_files]
        else:
            if not url or not url.strip():
                return _error_handler.format_error(
                    "未填写视频链接",
                    ["视频链接为空"],
                    ["请输入有效的视频、播放列表或频道链接"]
                )
    
    offline_warnings = []
    if not check_network() and input_mode == 'URL 下载':
        caps = get_offline_capabilities()
        offline_warnings.append("⚠️ 当前处于离线模式，无法从 URL 下载视频")
        offline_warnings.append("请选择「本地文件」模式或选择已有目录")
        offline_warnings.append("")
    cfg = load_config()
    if not cfg.get('OPENAI_API_KEY'):
        return _error_handler.format_error(
            "缺少 OpenAI API Key",
            ["翻译功能需要 OpenAI API Key 才能运行", "API Key 未在设置页面中配置"],
            ["前往设置页面填写 OpenAI API Key", "获取 API Key：https://platform.openai.com/api-keys"]
        )
    
    modules_param = None
    if use_module_selection and selected_modules and len(selected_modules) > 0:
        modules_param = selected_modules
    
    clear_log_buffer()
    try:
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
        output = f"✅ {result}" if result and not str(result).startswith("❌") else result
        if offline_warnings:
            output = "\n".join(offline_warnings) + output
        logs = get_log_buffer()
        if logs:
            return f"{logs}\n\n{output}"
        return output
    except Exception as e:
        logger.error(f"全自动流程执行失败: {e}")
        logs = get_log_buffer()
        error_result = _error_handler.classify_error(e)
        if logs:
            return f"{logs}\n\n{error_result}"
        return error_result


def demucs_wrapper(folder, model, device, progress, shifts, segment, max_chunk_seconds, folder_list_text=None, folder_select_files=None):
    from youdub.step010_demucs_vr import separate_audio_in_folders
    return _execution_service.execute_with_folders_support(
        folder, folder_list_text, folder_select_files,
        batch_function=separate_audio_in_folders,
        single_function=separate_all_audio_under_folder,
        module_name="人声分离",
        cleanup_function=cleanup_demucs,
        model_name=model, device=device, progress=progress, shifts=int(shifts),
        segment=int(segment), max_chunk_seconds=int(max_chunk_seconds)
    )


def whisper_wrapper(folder, model, diarization, download_root, device, batch_size, min_speakers, max_speakers, folder_list_text=None, folder_select_files=None):
    from youdub.step020_whisperx import transcribe_audio_in_folders
    return _execution_service.execute_with_folders_support(
        folder, folder_list_text, folder_select_files,
        batch_function=transcribe_audio_in_folders,
        single_function=transcribe_all_audio_under_folder,
        module_name="语音识别",
        cleanup_function=cleanup_whisperx,
        model_name=model, download_root=download_root, device=device,
        batch_size=int(batch_size), diarization=diarization,
        min_speakers=min_speakers, max_speakers=max_speakers
    )


def synthesize_wrapper(folder, subtitles, use_original_audio, resolution, speed_up, fps, folder_list_text=None, folder_select_files=None):
    from youdub.step050_synthesize_video import synthesize_video_in_folders
    return _execution_service.execute_with_folders_support(
        folder, folder_list_text, folder_select_files,
        batch_function=synthesize_video_in_folders,
        single_function=synthesize_all_video_under_folder,
        module_name="视频合成",
        subtitles=subtitles, use_original_audio=use_original_audio, speed_up=float(speed_up), fps=int(fps), resolution=resolution
    )


_cfg = load_config()

RESOLUTION_CHOICES = ['1080p', '720p', '480p']

MODULES_INFO = get_module_with_info()
MODULE_CHOICES = [(m["name"], m["id"]) for m in MODULES_INFO]
DEFAULT_MODULES = [m["id"] for m in MODULES_INFO]


def format_execution_order(selected_modules):
    if not selected_modules or len(selected_modules) == 0:
        return "未选择任何模块"
    
    try:
        ordered = resolve_dependencies(selected_modules)
        lines = ["模块执行顺序："]
        for i, mid in enumerate(ordered, 1):
            module = get_module(mid)
            if module:
                input_info = ""
                if module.get("input_files"):
                    input_info = f"（需要: {', '.join(module['input_files'])}）"
                lines.append(f"  {i}. {module['name']}{input_info}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"格式化模块执行顺序失败: {e}")
        return "模块选择无效"


def select_all_modules():
    return DEFAULT_MODULES, format_execution_order(DEFAULT_MODULES)


def deselect_all_modules():
    return [], format_execution_order([])


def toggle_input_mode(mode):
    if mode == '本地文件':
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)


def import_local_videos_wrapper(local_files, folder_path, title=None, uploader=None, upload_date=None):
    if local_files is None or len(local_files) == 0:
        return _error_handler.format_error(
            "未选择视频文件",
            ["没有选择任何本地视频文件"],
            ["请点击「上传文件」按钮选择一个或多个视频文件"]
        )
    clear_log_buffer()
    success_count = 0
    failed_count = 0
    results = []
    for f in local_files:
        file_path = f.name if hasattr(f, 'name') else f
        try:
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
            logger.error(f"导入视频失败 {file_path}: {e}")
            failed_count += 1
            results.append(f"❌ {os.path.basename(file_path)} -> 导入异常: {e}")
    output = "\n".join(results) + f"\n\n成功: {success_count}\n失败: {failed_count}"
    logs = get_log_buffer()
    if logs:
        return f"{logs}\n\n{output}"
    return output


def translation_wrapper(folder, lang, folder_list_text=None, folder_select_files=None):
    selected_folders = _file_utils.merge_folder_lists(folder_list_text, folder_select_files)
    if selected_folders:
        from youdub.step030_translation import translate_transcripts_in_folders
        return _execution_service.wrap_with_logs(translate_transcripts_in_folders, selected_folders, lang)
    return _execution_service.wrap_with_logs(translate_all_transcript_under_folder, folder, lang)


def tts_wrapper(folder, force_bytedance, folder_list_text=None, folder_select_files=None):
    selected_folders = _file_utils.merge_folder_lists(folder_list_text, folder_select_files)
    if selected_folders:
        from youdub.step040_tts import generate_wavs_in_folders
        return _execution_service.wrap_with_logs(generate_wavs_in_folders, selected_folders, force_bytedance)
    return _execution_service.wrap_with_logs(generate_all_wavs_under_folder, folder, force_bytedance)


def generate_info_wrapper(folder, folder_list_text=None, folder_select_files=None):
    selected_folders = _file_utils.merge_folder_lists(folder_list_text, folder_select_files)
    if selected_folders:
        from youdub.step060_generate_info import generate_info_in_folders
        return _execution_service.wrap_with_logs(generate_info_in_folders, selected_folders)
    return _execution_service.wrap_with_logs(generate_all_info_under_folder, folder)


def upload_bilibili_wrapper(folder, folder_list_text=None, folder_select_files=None):
    selected_folders = _file_utils.merge_folder_lists(folder_list_text, folder_select_files)
    if selected_folders:
        from youdub.step070_upload_bilibili import upload_videos_in_folders
        return _execution_service.wrap_with_logs(upload_videos_in_folders, selected_folders)
    return _execution_service.wrap_with_logs(upload_all_videos_under_folder, folder)


def update_folder_display(folder_text, select_files):
    folders = _file_utils.merge_folder_lists(folder_text, select_files)
    return "\n".join(folders) if folders else "未选择任何目录"


def create_folder_selector():
    with gr.Accordion("选择处理目录", open=False):
        gr.Markdown("选择要处理的已有目录。每个目录需包含相应的输入文件。")
        folder_list_text = gr.Textbox(
            label='目录路径列表（每行一个路径）',
            placeholder='D:\\videos\\Uploader\\20250101 My Video',
            lines=3
        )
        folder_select_files = gr.File(
            label='从文件选择目录（选择目录中任意文件自动提取路径）',
            file_count='multiple',
            type='filepath'
        )
        folder_list_display = gr.Textbox(
            label='已识别的目录列表',
            value='',
            interactive=False,
            lines=3
        )
        refresh_btn = gr.Button("刷新目录列表", size="sm")
        refresh_btn.click(
            fn=update_folder_display,
            inputs=[folder_list_text, folder_select_files],
            outputs=[folder_list_display]
        )
    return folder_list_text, folder_select_files


with gr.Blocks(title='YouDub') as app:
    with gr.Tabs():
        with gr.Tab('设置'):
            with gr.Accordion("快速入门", open=False):
                gr.Markdown(
                    "### 快速入门指南\n"
                    "1. 运行 `setup_windows.bat` 安装依赖\n"
                    "2. 复制 `.env.example` 为 `.env` 并填写配置\n"
                    "3. 在本页面填写必要配置并保存\n"
                    "4. 运行 `run_windows.bat` 启动应用\n"
                    "5. 在「全自动」页面输入视频链接开始处理"
                )
            gr.Markdown("配置 YouDub 所需的各项参数，带 * 的为必填项")
            with gr.Row():
                network_display = gr.Textbox(label="网络状态", value=_format_network_status(), interactive=False, lines=6, scale=1)
                status_display = gr.Textbox(label="配置状态", value=_format_status(), interactive=False, lines=6, scale=1)
            save_result = gr.Textbox(label="保存结果")
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
            with gr.Accordion("HuggingFace 设置", open=True):
                hf_token = gr.Textbox(label='HuggingFace Token *', type='password',
                    value=_cfg.get('HF_TOKEN', ''),
                    info='用于说话者分离和模型下载。获取方式：https://huggingface.co/settings/tokens')
                hf_endpoint = gr.Textbox(label='HuggingFace Endpoint',
                    value=_cfg.get('HF_ENDPOINT', ''),
                    info='例如：https://hf-mirror.com，用于加速模型下载')
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
            with gr.Accordion("模型管理", open=True):
                gr.Markdown(
                    "管理 YouDub 全流程所需的本地 AI 模型。首次使用前请先下载所需模型。\n\n"
                    "需要 HF_TOKEN 的模型需先在「HuggingFace 设置」中设置 HuggingFace Token。\n\n"
                    "如需使用镜像加速，请在「HuggingFace 设置」中设置 HuggingFace Endpoint（例如 https://hf-mirror.com），保存后再下载模型。"
                )
                model_status_display = gr.Textbox(label="模型状态", value=_format_model_status_ui(), interactive=False, lines=12)
                with gr.Row():
                    refresh_model_btn = gr.Button("🔄 刷新模型状态", variant="secondary")
                    download_all_btn = gr.Button("⬇️ 下载全部缺失模型", variant="primary")
                download_result = gr.Textbox(label="下载结果")
                refresh_model_btn.click(
                    fn=_refresh_model_status,
                    inputs=[],
                    outputs=[model_status_display]
                )
                download_all_btn.click(
                    fn=_download_all_missing_models,
                    inputs=[],
                    outputs=[download_result]
                ).then(
                    fn=_refresh_model_status,
                    inputs=[],
                    outputs=[model_status_display]
                )
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
        with gr.Tab('下载视频'):
            gr.Markdown("从视频平台下载视频，支持单个视频、播放列表和频道")
            with gr.Tabs():
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
                        url_btn.click(
                            fn=lambda url, folder, resolution, num_videos: _execution_service.wrap_with_logs(download_from_url, url, folder, resolution, num_videos),
                            inputs=[url_input, url_folder, url_resolution, url_num_videos],
                            outputs=url_output
                        )
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
                        local_btn.click(
                            fn=import_local_videos_wrapper,
                            inputs=[local_files, local_folder, local_title, local_uploader, local_upload_date],
                            outputs=local_output
                        )
        with gr.Tab('人声分离'):
            gr.Markdown("使用 Demucs 模型将视频中的人声和伴奏分离")
            dm_folder = gr.Textbox(label='Folder', value='videos')
            dm_folder_list_text, dm_folder_select_files = create_folder_selector()
            dm_model = gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'],
                        label='Model', value='htdemucs_ft')
            with gr.Accordion("高级设置", open=False):
                dm_device = gr.Radio(['auto', 'cuda', 'cpu'], label='Device', value='auto')
                dm_progress = gr.Checkbox(label='Progress Bar in Console', value=True)
                dm_shifts = gr.Slider(minimum=0, maximum=10, step=1, label='Number of shifts', value=5)
                dm_segment = gr.Slider(minimum=5, maximum=30, step=1, label='内部段长 (秒)', value=10, info='Demucs 内部分段推理长度，越小显存占用越低')
                dm_max_chunk = gr.Slider(minimum=120, maximum=1800, step=60, label='最大分块 (秒)', value=600, info='长音频外部分块大小，越小内存占用越低，0表示不限制')
            dm_output = gr.Textbox(label='输出')
            dm_btn = gr.Button("开始分离", variant="primary")
            dm_btn.click(
                fn=demucs_wrapper,
                inputs=[dm_folder, dm_model, dm_device, dm_progress, dm_shifts, dm_segment, dm_max_chunk, dm_folder_list_text, dm_folder_select_files],
                outputs=dm_output
            )
        with gr.Tab('语音识别'):
            gr.Markdown("使用 WhisperX 模型将语音转换为文字，支持说话者分离")
            ws_folder = gr.Textbox(label='Folder', value='videos')
            ws_folder_list_text, ws_folder_select_files = create_folder_selector()
            ws_model = gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label='Model', value='large')
            ws_diarization = gr.Checkbox(label='Diarization', value=True,
                         info='启用说话者分离，区分不同说话人')
            with gr.Accordion("高级设置", open=False):
                ws_download_root = gr.Textbox(label='Download Root', value='models/ASR/whisper')
                ws_device = gr.Radio(['auto', 'cuda', 'cpu'], label='Device', value='auto')
                ws_batch_size = gr.Slider(minimum=1, maximum=128, step=1, label='Batch Size', value=32)
                ws_min_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   label='Min Speakers', value=None)
                ws_max_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   label='Max Speakers', value=None)
            ws_output = gr.Textbox(label='输出')
            ws_btn = gr.Button("开始识别", variant="primary")
            ws_btn.click(
                fn=whisper_wrapper,
                inputs=[ws_folder, ws_model, ws_diarization, ws_download_root, ws_device, ws_batch_size, ws_min_speakers, ws_max_speakers, ws_folder_list_text, ws_folder_select_files],
                outputs=ws_output
            )
        with gr.Tab('字幕翻译'):
            gr.Markdown("使用大语言模型将字幕翻译为目标语言")
            tl_folder = gr.Textbox(label='Folder', value='videos')
            tl_folder_list_text, tl_folder_select_files = create_folder_selector()
            tl_lang = gr.Dropdown(['简体中文', '繁体中文', 'English', 'Deutsch', 'Français', 'русский'],
                        label='Target Language', value='简体中文')
            tl_output = gr.Textbox(label='输出')
            tl_btn = gr.Button("开始翻译", variant="primary")
            tl_btn.click(
                fn=translation_wrapper,
                inputs=[tl_folder, tl_lang, tl_folder_list_text, tl_folder_select_files],
                outputs=tl_output
            )
        with gr.Tab('语音合成'):
            gr.Markdown("使用 TTS 模型将翻译后的文字生成语音")
            tts_folder = gr.Textbox(label='Folder', value='videos')
            tts_folder_list_text, tts_folder_select_files = create_folder_selector()
            tts_force_bytedance = gr.Checkbox(label='Force Bytedance', value=False,
                    info='强制使用火山引擎 TTS，而非 IndexTTS 声音克隆')
            tts_output = gr.Textbox(label='输出')
            tts_btn = gr.Button("开始合成", variant="primary")
            tts_btn.click(
                fn=tts_wrapper,
                inputs=[tts_folder, tts_force_bytedance, tts_folder_list_text, tts_folder_select_files],
                outputs=tts_output
            )
        with gr.Tab('视频合成'):
            gr.Markdown("将配音、字幕与原视频合成为最终视频")
            sv_folder = gr.Textbox(label='Folder', value='videos')
            sv_folder_list_text, sv_folder_select_files = create_folder_selector()
            sv_subtitles = gr.Checkbox(label='Subtitles', value=True)
            sv_use_original_audio = gr.Checkbox(label='使用原视频音轨（不配音）', value=False,
                info='勾选后将跳过 TTS 配音，使用原视频音轨合成')
            sv_resolution = gr.Radio(RESOLUTION_CHOICES, label='Resolution', value='1080p')
            with gr.Accordion("高级设置", open=False):
                sv_speed_up = gr.Slider(minimum=0.5, maximum=2, step=0.05, label='Speed Up', value=1.05)
                sv_fps = gr.Slider(minimum=1, maximum=60, step=1, label='FPS', value=30)
            sv_output = gr.Textbox(label='输出')
            sv_btn = gr.Button("开始合成", variant="primary")
            sv_btn.click(
                fn=synthesize_wrapper,
                inputs=[sv_folder, sv_subtitles, sv_use_original_audio, sv_resolution, sv_speed_up, sv_fps, sv_folder_list_text, sv_folder_select_files],
                outputs=sv_output
            )
        with gr.Tab('信息生成'):
            gr.Markdown("生成视频上传所需信息")
            gi_folder = gr.Textbox(label='Folder', value='videos')
            gi_folder_list_text, gi_folder_select_files = create_folder_selector()
            gi_output = gr.Textbox(label='输出')
            gi_btn = gr.Button("开始生成", variant="primary")
            gi_btn.click(
                fn=generate_info_wrapper,
                inputs=[gi_folder, gi_folder_list_text, gi_folder_select_files],
                outputs=gi_output
            )
        with gr.Tab('上传B站'):
            gr.Markdown("将合成好的视频上传到 Bilibili")
            ub_folder = gr.Textbox(label='Folder', value='videos')
            ub_folder_list_text, ub_folder_select_files = create_folder_selector()
            ub_output = gr.Textbox(label='输出')
            ub_btn = gr.Button("开始上传", variant="primary")
            ub_btn.click(
                fn=upload_bilibili_wrapper,
                inputs=[ub_folder, ub_folder_list_text, ub_folder_select_files],
                outputs=ub_output
            )
        with gr.Tab('全自动'):
            gr.Markdown("一键完成从视频下载到配音合成的全流程，支持选择性执行特定模块")
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
                de_execution_order = gr.Textbox(label='执行顺序预览', value=format_execution_order(DEFAULT_MODULES),
                    interactive=False, lines=10)
                
                with gr.Row():
                    select_all_btn = gr.Button("全选", size="sm")
                    deselect_all_btn = gr.Button("全不选", size="sm")
                
                select_all_btn.click(
                    fn=select_all_modules,
                    outputs=[de_selected_modules, de_execution_order]
                )
                deselect_all_btn.click(
                    fn=deselect_all_modules,
                    outputs=[de_selected_modules, de_execution_order]
                )
                
                de_selected_modules.change(
                    fn=format_execution_order,
                    inputs=[de_selected_modules],
                    outputs=[de_execution_order]
                )
            
            with gr.Accordion("手动选择输入文件", open=False):
                gr.Markdown("为各模块手动选择输入文件。选择的文件将在执行时自动复制到处理文件夹。")
                
                de_selected_files = gr.State(value=_file_utils.create_selected_files_dict())
                de_selected_files_status = gr.Textbox(label="文件选择状态", value=_file_utils.format_selected_files_status(None),
                    interactive=False, lines=15)
                
                file_inputs = {}
                file_statuses = {}
                for module_id, module_info in MODULES.items():
                    if module_info.get("input_files"):
                        with gr.Accordion(f"{module_info['name']} - 输入文件", open=False):
                            for input_file in module_info["input_files"]:
                                format_info = INPUT_FILE_FORMATS.get(input_file, {})
                                file_label = _file_utils.get_file_label(input_file)
                                with gr.Row():
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
                
                for filename, file_input in file_inputs.items():
                    def create_file_callback(fname):
                        def callback(filepath, selected_files=None):
                            if selected_files is None:
                                selected_files = _file_utils.create_selected_files_dict()
                            
                            if filepath:
                                is_valid, msg = _file_utils.validate_file_format(filepath, fname)
                                if is_valid:
                                    selected_files[fname] = filepath
                                    status_msg = f"✅ {msg}"
                                else:
                                    selected_files[fname] = None
                                    status_msg = f"❌ {msg}"
                            else:
                                selected_files[fname] = None
                                status_msg = "未选择文件"
                            
                            return selected_files, status_msg, _file_utils.format_selected_files_status(selected_files)
                        return callback
                    
                    file_input.change(
                        fn=create_file_callback(filename),
                        inputs=[file_input, de_selected_files],
                        outputs=[de_selected_files, file_statuses[filename], de_selected_files_status]
                    )
                
                with gr.Row():
                    clear_files_btn = gr.Button("清除所有文件选择", size="sm")
                
                def clear_file_selections():
                    empty_files = _file_utils.create_selected_files_dict()
                    return [empty_files, _file_utils.format_selected_files_status(empty_files)] + [None] * len(file_inputs)
                
                clear_files_btn.click(
                    fn=clear_file_selections,
                    outputs=[de_selected_files, de_selected_files_status] + list(file_inputs.values())
                )
            
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
                
                de_refresh_folders_btn.click(
                    fn=update_folder_display,
                    inputs=[de_folder_list_text, de_folder_select_files],
                    outputs=[de_folder_list_display]
                )
            
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
            de_output = gr.Textbox(label='输出')
            de_btn = gr.Button("开始执行", variant="primary")
            
            de_input_mode.change(
                fn=toggle_input_mode,
                inputs=[de_input_mode],
                outputs=[de_url, de_local_files]
            )
            
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

def _check_dependency_compatibility():
    try:
        import numba
        import numpy
        nv = tuple(int(x) for x in numba.__version__.split('.')[:2])
        npv = tuple(int(x) for x in numpy.__version__.split('.')[:2])
        max_np = {
            (0, 60): (2, 0), (0, 61): (2, 1), (0, 62): (2, 1),
            (0, 63): (2, 3), (0, 64): (2, 4), (0, 65): (2, 4),
        }.get(nv, (2, 0))
        if npv > max_np:
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
        logger.warning("无法检查 Numba/NumPy 版本兼容性（可能未安装）")
        return True
    except Exception as e:
        logger.debug(f"版本兼容性检查异常: {e}")
        return True


if __name__ == '__main__':
    _check_dependency_compatibility()
    app.launch(server_port=19876)
