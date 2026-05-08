import os
import shutil
from typing import Dict, List, Optional, Tuple
from loguru import logger
from ..config import PROJECT_ROOT


INPUT_FILE_FORMATS = {
    "download.mp4": {
        "extensions": [".mp4", ".avi", ".mkv", ".mov", ".flv"],
        "description": "视频文件"
    },
    "audio_vocals.wav": {
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac"],
        "description": "人声音频文件"
    },
    "audio_instruments.wav": {
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac"],
        "description": "伴奏音频文件"
    },
    "transcript.json": {
        "extensions": [".json"],
        "description": "识别结果文件"
    },
    "download.info.json": {
        "extensions": [".json"],
        "description": "视频信息文件"
    },
    "translation.json": {
        "extensions": [".json"],
        "description": "翻译结果文件"
    },
    "summary.json": {
        "extensions": [".json"],
        "description": "摘要文件"
    },
    "audio_combined.wav": {
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac"],
        "description": "合成音频文件"
    },
    "audio_tts.wav": {
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac"],
        "description": "TTS输出文件"
    },
    "video.mp4": {
        "extensions": [".mp4", ".avi", ".mkv", ".mov"],
        "description": "合成视频文件"
    },
    "video.txt": {
        "extensions": [".txt"],
        "description": "视频标题描述文件"
    },
    "video.png": {
        "extensions": [".png", ".jpg", ".jpeg"],
        "description": "视频封面文件"
    }
}


class FileUtils:

    @staticmethod
    def validate_file_format(file_path: str, expected_filename: str) -> Tuple[bool, str]:
        if file_path is None or not os.path.exists(file_path):
            return False, "文件不存在"

        file_ext = os.path.splitext(file_path)[1].lower()
        format_info = INPUT_FILE_FORMATS.get(expected_filename)

        if format_info is None:
            return True, "无格式验证规则"

        if file_ext not in format_info["extensions"]:
            return False, f"文件格式不符合要求。支持的格式：{', '.join(format_info['extensions'])}"

        return True, f"文件格式正确（{format_info['description']}）"

    @staticmethod
    def copy_selected_file_to_folder(src_file: str, dest_folder: str, target_filename: str) -> Tuple[bool, str]:
        if src_file is None or not os.path.exists(src_file):
            return False, "源文件不存在"

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, target_filename)

        try:
            shutil.copy2(src_file, dest_path)
            return True, f"文件已复制到：{dest_path}"
        except Exception as e:
            logger.error(f"复制文件失败: {e}")
            return False, f"复制文件失败：{str(e)}"

    @staticmethod
    def get_file_label(filename: str) -> str:
        format_info = INPUT_FILE_FORMATS.get(filename)
        if format_info:
            return f"{filename} ({format_info['description']})"
        return filename

    @staticmethod
    def create_selected_files_dict() -> Dict[str, Optional[str]]:
        return {filename: None for filename in INPUT_FILE_FORMATS.keys()}

    @staticmethod
    def update_file_selection(selected_files: Dict[str, Optional[str]],
                              filename: str, filepath: str) -> Dict[str, Optional[str]]:
        if selected_files is None:
            selected_files = FileUtils.create_selected_files_dict()
        selected_files[filename] = filepath
        return selected_files

    @staticmethod
    def format_selected_files_status(selected_files: Dict[str, Optional[str]]) -> str:
        if selected_files is None:
            selected_files = FileUtils.create_selected_files_dict()

        lines = ["文件选择状态："]
        for filename, filepath in selected_files.items():
            if filepath:
                status = f"✅ 已选择：{os.path.basename(filepath)}"
            else:
                status = "❌ 未选择"
            lines.append(f"  {FileUtils.get_file_label(filename)}: {status}")

        return "\n".join(lines)

    @staticmethod
    def resolve_folder_path(folder: str) -> str:
        if not folder or os.path.isabs(folder):
            return folder
        resolved = str(PROJECT_ROOT / folder)
        return resolved

    @staticmethod
    def extract_folders_from_files(file_paths: list) -> List[str]:
        if not file_paths:
            return []
        folders = set()
        for fp in file_paths:
            fpath = fp.name if hasattr(fp, 'name') else fp
            folder = os.path.dirname(os.path.abspath(fpath))
            folders.add(folder)
        return sorted(folders)

    @staticmethod
    def resolve_folder_list_text(folder_text: str) -> List[str]:
        if not folder_text or not folder_text.strip():
            return []
        lines = [line.strip() for line in folder_text.split('\n') if line.strip()]
        return [os.path.abspath(line) for line in lines]

    @staticmethod
    def merge_folder_lists(folder_text: str, selected_files: list) -> List[str]:
        folders = set(FileUtils.resolve_folder_list_text(folder_text))
        folders.update(FileUtils.extract_folders_from_files(selected_files))
        return sorted(folders)
