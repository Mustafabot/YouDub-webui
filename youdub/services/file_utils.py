import os
import shutil
import uuid
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
    },
    "download.jpg": {
        "extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        "description": "缩略图文件"
    },
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
    def create_temp_working_dir(prefix: str) -> str:
        temp_root = os.path.join(str(PROJECT_ROOT), ".temp")
        os.makedirs(temp_root, exist_ok=True)
        dir_name = f"{prefix}_{uuid.uuid4().hex[:8]}"
        temp_dir = os.path.join(temp_root, dir_name)
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"创建临时工作目录: {temp_dir}")
        return temp_dir

    @staticmethod
    def prepare_single_input_dirs(file_paths: list, target_filename: str) -> List[Tuple[str, str]]:
        """为每个输入文件创建临时工作目录，将文件复制进去

        Returns:
            List[Tuple[str, str]]: 每个元素为 (working_dir, source_dir)，source_dir 是输入文件的原始目录
        """
        dirs = []
        for file_path in file_paths:
            fpath = file_path.name if hasattr(file_path, 'name') else file_path
            if not fpath or not os.path.exists(fpath):
                logger.warning(f"文件不存在，跳过: {fpath}")
                continue
            base = os.path.splitext(os.path.basename(fpath))[0]
            working_dir = FileUtils.create_temp_working_dir(base)
            shutil.copy2(fpath, os.path.join(working_dir, target_filename))
            source_dir = os.path.dirname(os.path.abspath(fpath))
            dirs.append((working_dir, source_dir))
        return dirs

    @staticmethod
    def prepare_multi_input_dir(file_map: Dict[str, str]) -> str:
        working_dir = FileUtils.create_temp_working_dir("multi")
        for target_filename, file_path in file_map.items():
            fpath = file_path.name if hasattr(file_path, 'name') else file_path
            if not fpath or not os.path.exists(fpath):
                logger.warning(f"文件不存在，跳过: {target_filename} <- {fpath}")
                continue
            shutil.copy2(fpath, os.path.join(working_dir, target_filename))
        return working_dir

    @staticmethod
    def collect_output_files(working_dir: str) -> List[str]:
        if not os.path.exists(working_dir):
            return []
        files = []
        for f in os.listdir(working_dir):
            fpath = os.path.join(working_dir, f)
            if os.path.isfile(fpath):
                files.append(fpath)
        return sorted(files)

    @staticmethod
    def copy_output_files_back(working_dir: str, source_dir: str,
                                input_filenames: Optional[List[str]] = None) -> List[str]:
        """将 working_dir 中处理产生的新文件复制回 source_dir

        Args:
            working_dir: 临时工作目录
            source_dir: 原始源文件目录
            input_filenames: 输入文件名列表（这些文件不会被复制回源目录）

        Returns:
            List[str]: 成功复制到 source_dir 的文件路径列表
        """
        if not os.path.exists(working_dir):
            return []
        if not os.path.exists(source_dir):
            os.makedirs(source_dir, exist_ok=True)

        exclude = set(input_filenames or [])
        copied = []
        for f in os.listdir(working_dir):
            if f in exclude:
                continue
            src_path = os.path.join(working_dir, f)
            if not os.path.isfile(src_path):
                continue
            dest_path = os.path.join(source_dir, f)
            try:
                shutil.copy2(src_path, dest_path)
                logger.info(f"输出文件已回拷: {f} -> {dest_path}")
                copied.append(dest_path)
            except Exception as e:
                logger.warning(f"回拷文件失败 {f}: {e}")
        return copied

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
