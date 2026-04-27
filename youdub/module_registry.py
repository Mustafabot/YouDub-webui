import os
from .config import get_config, check_network, get_config_status

MODULES = {
    "video_download": {
        "id": "video_download",
        "name": "视频下载",
        "description": "从URL下载视频或导入本地文件",
        "dependencies": [],
        "input_files": [],
        "required_config": [],
        "requires_ffmpeg": True,
        "output_files": ["download.mp4", "download.info.json"],
        "function": "download_all_videos_under_folder",
        "module": "step000_video_downloader",
        "online_only": True,
    },
    "audio_separation": {
        "id": "audio_separation",
        "name": "音频分离",
        "description": "分离人声和伴奏",
        "dependencies": ["video_download"],
        "input_files": ["download.mp4"],
        "required_config": [],
        "requires_ffmpeg": True,
        "output_files": ["audio_vocals.wav", "audio_instruments.wav"],
        "function": "separate_all_audio_under_folder",
        "module": "step010_demucs_vr",
        "online_only": False,
    },
    "speech_recognition": {
        "id": "speech_recognition",
        "name": "语音识别",
        "description": "将语音转换为文字",
        "dependencies": ["audio_separation"],
        "input_files": ["audio_vocals.wav"],
        "required_config": ["HF_TOKEN"],
        "requires_ffmpeg": False,
        "output_files": ["transcript.json"],
        "function": "transcribe_all_audio_under_folder",
        "module": "step020_whisperx",
        "online_only": False,
    },
    "translation": {
        "id": "translation",
        "name": "字幕翻译",
        "description": "将字幕翻译为目标语言",
        "dependencies": ["speech_recognition"],
        "input_files": ["transcript.json", "download.info.json"],
        "required_config": ["OPENAI_API_KEY"],
        "requires_ffmpeg": False,
        "output_files": ["translation.json", "summary.json"],
        "function": "translate_all_transcript_under_folder",
        "module": "step030_translation",
        "online_only": True,
    },
    "tts": {
        "id": "tts",
        "name": "语音合成",
        "description": "将翻译后的文字生成语音",
        "dependencies": ["translation"],
        "input_files": ["translation.json", "audio_vocals.wav", "audio_instruments.wav"],
        "required_config": ["BYTEDANCE_APPID", "BYTEDANCE_ACCESS_TOKEN"],
        "requires_ffmpeg": False,
        "output_files": ["audio_combined.wav", "audio_tts.wav"],
        "function": "generate_all_wavs_under_folder",
        "module": "step040_tts",
        "online_only": True,
    },
    "video_synthesis": {
        "id": "video_synthesis",
        "name": "视频合成",
        "description": "将配音、字幕与原视频合成",
        "dependencies": ["tts"],
        "input_files": ["download.mp4", "audio_combined.wav", "translation.json"],
        "required_config": [],
        "requires_ffmpeg": True,
        "output_files": ["video.mp4"],
        "function": "synthesize_all_video_under_folder",
        "module": "step050_synthesize_video",
        "online_only": False,
    },
    "generate_info": {
        "id": "generate_info",
        "name": "信息生成",
        "description": "生成视频上传所需信息",
        "dependencies": ["video_synthesis"],
        "input_files": ["summary.json", "download.info.json"],
        "required_config": [],
        "requires_ffmpeg": False,
        "output_files": ["video.txt", "video.png"],
        "function": "generate_all_info_under_folder",
        "module": "step060_genrate_info",
        "online_only": False,
    },
    "upload_bilibili": {
        "id": "upload_bilibili",
        "name": "B站上传",
        "description": "将视频上传到Bilibili",
        "dependencies": ["generate_info"],
        "input_files": ["video.mp4", "summary.json", "video.png"],
        "required_config": ["BILI_SESSDATA", "BILI_BILI_JCT"],
        "requires_ffmpeg": False,
        "output_files": ["bilibili.json"],
        "function": "upload_all_videos_under_folder",
        "module": "step070_upload_bilibili",
        "online_only": True,
    },
}


def get_module(module_id):
    """获取模块信息"""
    return MODULES.get(module_id)


def get_all_modules():
    """获取所有模块列表"""
    return list(MODULES.values())


def get_module_ids():
    """获取所有模块ID列表"""
    return list(MODULES.keys())


def get_available_modules():
    """获取可用模块（检查配置和网络）"""
    online = check_network()
    status = get_config_status()
    available = []
    
    for module in get_all_modules():
        if module["online_only"] and not online:
            continue
        
        config_ok = True
        for cfg_key in module["required_config"]:
            if not status.get(cfg_key, {}).get("set", False):
                config_ok = False
                break
        
        if config_ok:
            available.append(module)
    
    return available


def get_module_input_files(module_id):
    """获取模块执行所需的输入文件列表"""
    module = get_module(module_id)
    if not module:
        return []
    return module.get("input_files", [])


def get_input_file_producer(filename):
    """根据输入文件名反查产出该文件的模块ID"""
    for mid, module in MODULES.items():
        if filename in module.get("output_files", []):
            return mid
    return None


def get_module_dependencies(module_id):
    """获取模块的直接依赖列表"""
    module = get_module(module_id)
    if not module:
        return []
    return list(module["dependencies"])


def get_module_reverse_dependencies(module_id):
    """获取依赖此模块的所有模块（反向依赖）"""
    reverse_deps = []
    for mid, module in MODULES.items():
        if module_id in module["dependencies"]:
            reverse_deps.append(mid)
    return reverse_deps


def is_module_completed(folder, module_id):
    """检查模块是否已完成（输出文件存在）"""
    module = get_module(module_id)
    if not module:
        return False
    
    for output_file in module["output_files"]:
        file_path = os.path.join(folder, output_file)
        if not os.path.exists(file_path):
            return False
    
    return True


def check_module_config_available(module_id):
    """检查模块所需配置是否完整"""
    module = get_module(module_id)
    if not module:
        return False
    
    for cfg_key in module["required_config"]:
        if not get_config(cfg_key):
            return False
    
    return True


def get_module_missing_config(module_id):
    """获取模块缺失的配置项"""
    module = get_module(module_id)
    if not module:
        return []
    
    missing = []
    for cfg_key in module["required_config"]:
        if not get_config(cfg_key):
            missing.append(cfg_key)
    
    return missing


def resolve_dependencies(selected_modules):
    """解析选中模块并返回拓扑排序后的列表（不自动添加上游依赖）"""
    if not selected_modules:
        return []
    resolved = list(selected_modules)
    return get_execution_order(resolved)


def get_execution_order(module_ids):
    """获取模块执行顺序（拓扑排序）"""
    in_degree = {mid: 0 for mid in module_ids}
    graph = {mid: [] for mid in module_ids}
    
    for mid in module_ids:
        module = get_module(mid)
        if module:
            for dep in module["dependencies"]:
                if dep in module_ids:
                    graph[dep].append(mid)
                    in_degree[mid] += 1
    
    queue = [mid for mid in module_ids if in_degree[mid] == 0]
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result
