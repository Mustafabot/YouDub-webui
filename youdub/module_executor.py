import os
import time
from loguru import logger
from .module_registry import (
    get_module, is_module_completed, resolve_dependencies, 
    get_execution_order, get_available_modules, get_module_missing_config,
    get_module_input_files, get_input_file_producer
)
from .config import check_network, check_ffmpeg_available, ensure_ffmpeg_available
from .model_manager import check_model_status, get_model_info

def build_execution_plan(selected_modules):
    """根据选择的模块构建执行计划（仅排序，不自动添加上游依赖）"""
    if not selected_modules:
        return []
    
    ordered = resolve_dependencies(selected_modules)
    return ordered


def validate_execution_plan(execution_plan):
    """验证执行计划的可行性，检查输入文件是否可能就绪"""
    warnings = []
    online = check_network()
    ffmpeg_available, ffmpeg_msg = check_ffmpeg_available()
    available_modules = [m["id"] for m in get_available_modules()]
    plan_set = set(execution_plan)
    
    for module_id in execution_plan:
        module = get_module(module_id)
        if not module:
            warnings.append(f"模块 {module_id} 不存在")
            continue
        
        if module["online_only"] and not online:
            warnings.append(f"模块 {module['name']} 需要网络连接")
        
        if module.get("requires_ffmpeg", False) and not ffmpeg_available:
            warnings.append(f"模块 {module['name']} 需要 FFmpeg: {ffmpeg_msg}")
        
        if module_id not in available_modules:
            missing = get_module_missing_config(module_id)
            if missing:
                warnings.append(f"模块 {module['name']} 缺少配置: {', '.join(missing)}")
        
        required_models = module.get("required_models", [])
        for model_id in required_models:
            status = check_model_status(model_id)
            if not status["downloaded"]:
                model_info = get_model_info(model_id)
                model_name = model_info["name"] if model_info else model_id
                warnings.append(f"模块 {module['name']} 需要模型「{model_name}」尚未下载，请在设置页面下载模型")
        
        input_files = module.get("input_files", [])
        if input_files:
            for infile in input_files:
                producer = get_input_file_producer(infile)
                if producer and producer not in plan_set:
                    producer_name = get_module(producer)["name"] if get_module(producer) else producer
                    warnings.append(f"模块 {module['name']} 需要输入文件 {infile}（由模块「{producer_name}」产出），但该模块未在执行计划中。请确保输入文件已存在，或先运行模块「{producer_name}」")
    
    return warnings


def check_module_output_exists(folder, module_id):
    """检查模块输出文件是否存在"""
    return is_module_completed(folder, module_id)


class ModuleExecutor:
    """模块执行器"""
    
    def __init__(self):
        self.module_functions = {}
        self.init_functions = {}
    
    def register_module_function(self, module_id, func, init_func=None):
        """注册模块执行函数和初始化函数"""
        self.module_functions[module_id] = func
        if init_func:
            self.init_functions[module_id] = init_func
    
    def execute_init_functions(self, execution_plan):
        """执行模块初始化函数（并行）"""
        for module_id in execution_plan:
            if module_id in self.init_functions:
                try:
                    logger.info(f"初始化模块: {get_module(module_id)['name']}")
                    self.init_functions[module_id]()
                except Exception as e:
                    logger.warning(f"初始化模块 {module_id} 失败: {e}")
    
    def execute(self, folder, execution_plan, params, skip_completed=True, max_retries=3):
        """执行模块列表"""
        results = []
        total = len(execution_plan)
        
        for idx, module_id in enumerate(execution_plan, 1):
            module = get_module(module_id)
            if not module:
                continue
            
            module_name = module["name"]
            
            if skip_completed and check_module_output_exists(folder, module_id):
                logger.info(f"[{idx}/{total}] 跳过已完成模块: {module_name}")
                results.append({"module": module_id, "status": "skipped"})
                continue
            
            if module.get("requires_ffmpeg", False):
                ffmpeg_ok, ffmpeg_msg = ensure_ffmpeg_available(auto_download=True)
                if not ffmpeg_ok:
                    logger.error(f"[{idx}/{total}] 模块 {module_name} 需要 FFmpeg: {ffmpeg_msg}")
                    results.append({"module": module_id, "status": "error", "error": ffmpeg_msg})
                    return results, False
            
            input_files = module.get("input_files", [])
            missing_inputs = []
            for infile in input_files:
                if not os.path.exists(os.path.join(folder, infile)):
                    missing_inputs.append(infile)
            
            if missing_inputs:
                suggestions = []
                for infile in missing_inputs:
                    producer = get_input_file_producer(infile)
                    if producer:
                        producer_name = get_module(producer)["name"] if get_module(producer) else producer
                        suggestions.append(f"{infile}（请先运行模块「{producer_name}」或手动准备该文件）")
                    else:
                        suggestions.append(f"{infile}（请手动准备该文件）")
                msg = f"模块 {module_name} 缺少输入文件: " + "、".join(suggestions)
                logger.error(f"[{idx}/{total}] {msg}")
                results.append({"module": module_id, "status": "error", "error": msg})
                return results, False
            
            if module_id not in self.module_functions:
                logger.warning(f"[{idx}/{total}] 模块 {module_name} 没有注册执行函数")
                results.append({"module": module_id, "status": "error", "error": "No function registered"})
                continue
            
            func = self.module_functions[module_id]
            func_params = self._get_module_params(module_id, params)
            
            for retry in range(max_retries):
                try:
                    logger.info(f"[{idx}/{total}] 执行模块: {module_name} (重试 {retry + 1}/{max_retries})")
                    result = func(folder, **func_params)
                    logger.info(f"[{idx}/{total}] 模块 {module_name} 执行成功")
                    results.append({"module": module_id, "status": "success", "result": result})
                    break
                except Exception as e:
                    logger.error(f"[{idx}/{total}] 模块 {module_name} 执行失败: {e}")
                    if retry == max_retries - 1:
                        results.append({"module": module_id, "status": "error", "error": str(e)})
                        return results, False
                    time.sleep(1)
        
        return results, True
    
    def _get_module_params(self, module_id, params):
        """根据模块ID提取对应的参数"""
        param_mapping = {
            "video_download": {
                "url": params.get("url"),
                "video_path": params.get("video_path"),
                "resolution": params.get("resolution", "1080p"),
                "num_videos": params.get("num_videos", 5),
            },
            "audio_separation": {
                "model_name": params.get("demucs_model"),
                "device": params.get("device"),
                "progress": True,
                "shifts": params.get("shifts"),
                "segment": params.get("demucs_segment", 10),
                "max_chunk_seconds": params.get("demucs_max_chunk_seconds", 600),
            },
            "speech_recognition": {
                "model_name": params.get("whisper_model"),
                "download_root": params.get("whisper_download_root"),
                "device": params.get("device"),
                "batch_size": params.get("whisper_batch_size"),
                "diarization": params.get("whisper_diarization"),
                "min_speakers": params.get("whisper_min_speakers"),
                "max_speakers": params.get("whisper_max_speakers"),
            },
            "translation": {
                "target_language": params.get("translation_target_language"),
            },
            "tts": {
                "force_bytedance": params.get("force_bytedance"),
            },
            "video_synthesis": {
                "subtitles": params.get("subtitles"),
                "use_original_audio": params.get("use_original_audio", False),
                "speed_up": params.get("speed_up"),
                "fps": params.get("fps"),
                "resolution": params.get("target_resolution"),
            },
        }
        return param_mapping.get(module_id, {})


_default_executor = None


def get_default_executor():
    """获取全局默认执行器（单例）"""
    global _default_executor
    if _default_executor is None:
        _default_executor = create_default_executor()
    return _default_executor


def create_default_executor():
    """创建默认的模块执行器（预先注册所有模块函数）"""
    executor = ModuleExecutor()
    
    from .step000_video_downloader import download_all_videos_under_folder, download_videos_in_folders
    from .step010_demucs_vr import separate_all_audio_under_folder, separate_audio_in_folders, init_demucs
    from .step020_whisperx import transcribe_all_audio_under_folder, transcribe_audio_in_folders, init_whisperx
    from .step030_translation import translate_all_transcript_under_folder, translate_transcripts_in_folders
    from .step040_tts import generate_all_wavs_under_folder, generate_wavs_in_folders
    from .step043_tts_f5 import init_F5TTS
    from .step050_synthesize_video import synthesize_all_video_under_folder, synthesize_video_in_folders
    from .step060_generate_info import generate_all_info_under_folder, generate_info_in_folders
    from .step070_upload_bilibili import upload_all_videos_under_folder, upload_videos_in_folders
    
    executor.register_module_function("video_download", download_all_videos_under_folder)
    executor.register_module_function("audio_separation", separate_all_audio_under_folder, init_demucs)
    executor.register_module_function("speech_recognition", transcribe_all_audio_under_folder, init_whisperx)
    executor.register_module_function("translation", translate_all_transcript_under_folder)
    executor.register_module_function("tts", generate_all_wavs_under_folder, init_F5TTS)
    executor.register_module_function("video_synthesis", synthesize_all_video_under_folder)
    executor.register_module_function("generate_info", generate_all_info_under_folder)
    executor.register_module_function("upload_bilibili", upload_all_videos_under_folder)
    
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
    """获取所有模块的名称列表（用于UI显示）"""
    from .module_registry import get_all_modules
    return [(m["id"], m["name"]) for m in get_all_modules()]


def get_module_with_info():
    """获取带详细信息的模块列表（用于UI显示）"""
    from .module_registry import get_all_modules, get_module_dependencies
    result = []
    for m in get_all_modules():
        deps = get_module_dependencies(m["id"])
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
