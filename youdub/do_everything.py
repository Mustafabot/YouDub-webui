import json
import os
import time
import shutil
from loguru import logger
from .step000_video_downloader import get_info_list_from_url, download_single_video, get_target_folder, process_local_video, get_info_list_from_local
from .module_registry import (
    get_module, get_module_ids, resolve_dependencies, 
    get_execution_order, is_module_completed
)
from .module_executor import create_default_executor, build_execution_plan, validate_execution_plan


def copy_user_selected_files(folder, selected_files):
    """将用户选择的文件复制到处理文件夹"""
    if selected_files is None:
        return
    
    for filename, filepath in selected_files.items():
        if filepath and os.path.exists(filepath):
            dest_path = os.path.join(folder, filename)
            try:
                shutil.copy2(filepath, dest_path)
                logger.info(f"已复制用户选择的文件：{filename} -> {dest_path}")
            except Exception as e:
                logger.warning(f"复制文件失败 {filename}: {e}")


def process_video_with_modules(info, root_folder, params, selected_modules=None, skip_completed=True, selected_files=None):
    """使用选择性模块处理视频
    
    Args:
        selected_files: 用户手动选择的输入文件字典，格式为 {filename: filepath}
    """
    is_local = info.get("is_local", False)
    
    if is_local:
        folder = process_local_video(info, root_folder)
    else:
        folder = get_target_folder(info, root_folder)
    
    if folder is None:
        logger.warning(f"Failed to get target folder for video {info['title']}")
        return False
    
    if os.path.exists(os.path.join(folder, "bilibili.json")):
        with open(os.path.join(folder, "bilibili.json"), "r", encoding="utf-8") as f:
            bilibili_info = json.load(f)
        if bilibili_info.get("results", [{}])[0].get("code") == 0:
            logger.info(f"Video already uploaded in {folder}")
            return True
    
    if not is_local:
        folder = download_single_video(info, root_folder, params.get("resolution", "1080p"))
        if folder is None:
            logger.warning(f"Failed to download video {info['title']}")
            return True
    
    # 复制用户选择的输入文件到处理文件夹
    copy_user_selected_files(folder, selected_files)
    
    logger.info(f"Process video in {folder}")
    
    if selected_modules is None:
        selected_modules = get_module_ids()
    
    execution_plan = build_execution_plan(selected_modules)
    if not execution_plan:
        logger.warning("No valid modules to execute")
        return True
    
    warnings = validate_execution_plan(execution_plan)
    for warning in warnings:
        logger.warning(warning)
    
    logger.info(f"Execution plan: {[get_module(mid)['name'] for mid in execution_plan]}")
    
    executor = create_default_executor()
    executor.execute_init_functions(execution_plan)
    
    max_retries = params.get("max_retries", 3)
    results, success = executor.execute(folder, execution_plan, params, skip_completed, max_retries)
    
    if not success:
        for result in results:
            if result.get("status") == "error":
                logger.error(f"Module {result['module']} failed: {result.get('error')}")
        return False
    
    return True


def process_video(info, root_folder, resolution, demucs_model, device, shifts, whisper_model, whisper_download_root, whisper_batch_size, whisper_diarization, whisper_min_speakers, whisper_max_speakers, translation_target_language, force_bytedance, subtitles, speed_up, fps, target_resolution, max_retries, auto_upload_video):
    """原有全流程处理函数（保持向后兼容）"""
    local_time = time.localtime()
    
    for retry in range(max_retries):
        try:
            is_local = info.get("is_local", False)
            
            if is_local:
                folder = process_local_video(info, root_folder)
            else:
                folder = get_target_folder(info, root_folder)
            
            if folder is None:
                logger.warning(f"Failed to get target folder for video {info['title']}")
                return False
            
            if os.path.exists(os.path.join(folder, "bilibili.json")):
                with open(os.path.join(folder, "bilibili.json"), "r", encoding="utf-8") as f:
                    bilibili_info = json.load(f)
                if bilibili_info.get("results", [{}])[0].get("code") == 0:
                    logger.info(f"Video already uploaded in {folder}")
                    return True
                
            if not is_local:
                folder = download_single_video(info, root_folder, resolution)
                if folder is None:
                    logger.warning(f"Failed to download video {info['title']}")
                    return True
            
            logger.info(f"Process video in {folder}")
            
            from .step010_demucs_vr import separate_all_audio_under_folder, cleanup_demucs
            from .step020_whisperx import transcribe_all_audio_under_folder, cleanup_whisperx
            from .step030_translation import translate_all_transcript_under_folder
            from .step040_tts import generate_all_wavs_under_folder
            from .step043_tts_f5 import cleanup_f5tts
            from .step050_synthesize_video import synthesize_all_video_under_folder
            from .step060_generate_info import generate_all_info_under_folder
            from .step070_upload_bilibili import upload_all_videos_under_folder
            
            separate_all_audio_under_folder(
                folder, model_name=demucs_model, device=device, progress=True, shifts=shifts)
            transcribe_all_audio_under_folder(
                folder, model_name=whisper_model, download_root=whisper_download_root, device=device, batch_size=whisper_batch_size, diarization=whisper_diarization, 
                min_speakers=whisper_min_speakers,
                max_speakers=whisper_max_speakers)
            
            translate_all_transcript_under_folder(
                folder, target_language=translation_target_language
            )
            generate_all_wavs_under_folder(folder, force_bytedance=force_bytedance)
            synthesize_all_video_under_folder(folder, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=target_resolution)
            generate_all_info_under_folder(folder)
            if auto_upload_video:
                time.sleep(1)
                upload_all_videos_under_folder(folder)
            return True
        except Exception as e:
            logger.error(f"Error processing video {info['title']}: {e}")
            try:
                cleanup_demucs()
                cleanup_whisperx()
                cleanup_f5tts()
            except Exception:
                pass
    return False


def do_everything(root_folder, url=None, local_video_paths=None, num_videos=5, resolution="1080p", demucs_model="htdemucs_ft", device="auto", shifts=5, whisper_model="large", whisper_download_root="models/ASR/whisper", whisper_batch_size=32, whisper_diarization=True, whisper_min_speakers=None, whisper_max_speakers=None, translation_target_language="简体中文", force_bytedance=True, subtitles=True, speed_up=1.05, fps=30, target_resolution="1080p", max_workers=1, max_retries=3, auto_upload_video=False, selected_modules=None, skip_completed=True, selected_files=None):
    """
    全自动处理视频
    
    Args:
        selected_modules: 选择的模块ID列表，None表示执行所有模块
        skip_completed: 是否跳过已完成的步骤
        selected_files: 用户手动选择的输入文件字典，格式为 {filename: filepath}
    """
    success_list = []
    fail_list = []

    video_info_iterator = None
    if local_video_paths is not None and len(local_video_paths) > 0:
        video_info_iterator = get_info_list_from_local(local_video_paths, root_folder)
    elif url is not None and url.strip():
        url = url.replace(" ", "").replace("，", "\n").replace(",", "\n")
        urls = [_ for _ in url.split("\n") if _]
        video_info_iterator = get_info_list_from_url(urls, num_videos)
    else:
        return "Error: Please provide either video URL(s) or local video file(s)"
    
    params = {
        "resolution": resolution,
        "demucs_model": demucs_model,
        "device": device,
        "shifts": shifts,
        "whisper_model": whisper_model,
        "whisper_download_root": whisper_download_root,
        "whisper_batch_size": whisper_batch_size,
        "whisper_diarization": whisper_diarization,
        "whisper_min_speakers": whisper_min_speakers,
        "whisper_max_speakers": whisper_max_speakers,
        "translation_target_language": translation_target_language,
        "force_bytedance": force_bytedance,
        "subtitles": subtitles,
        "speed_up": speed_up,
        "fps": fps,
        "target_resolution": target_resolution,
        "max_retries": max_retries,
        "auto_upload_video": auto_upload_video,
    }
    
    for info in video_info_iterator:
        try:
            if selected_modules is not None:
                success = process_video_with_modules(info, root_folder, params, selected_modules, skip_completed, selected_files)
            else:
                success = process_video(info, root_folder, resolution, demucs_model, device, shifts, whisper_model, whisper_download_root, whisper_batch_size, whisper_diarization, whisper_min_speakers, whisper_max_speakers, translation_target_language, force_bytedance, subtitles, speed_up, fps, target_resolution, max_retries, auto_upload_video)
            
            if success:
                success_list.append(info)
            else:
                fail_list.append(info)
        except Exception as e:
            logger.error(f"Error processing video {info.get('title', 'unknown')}: {e}")
            fail_list.append(info)

    return f"Success: {len(success_list)}\nFail: {len(fail_list)}"
