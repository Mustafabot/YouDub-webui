import os
import shutil
import tempfile
import time
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
    def _is_system_temp_dir(path: str) -> bool:
        """检测路径是否位于系统临时目录"""
        temp_dir = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
        path_norm = os.path.normcase(os.path.abspath(path))
        if path_norm == temp_dir:
            return True
        return path_norm.startswith(temp_dir + os.sep)

    @staticmethod
    def _resolve_output_dir(fpath: str, user_output_dir: Optional[str] = None) -> str:
        """解析最终输出目录

        Args:
            fpath: 输入文件路径
            user_output_dir: 用户指定的输出目录（直接使用，不创建子目录）

        Returns:
            str: 最终输出目录路径
        """
        # 用户指定优先 - 直接使用用户选择的目录，不创建子目录
        if user_output_dir and user_output_dir.strip():
            output_dir = FileUtils.resolve_folder_path(user_output_dir)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"使用用户指定输出目录: {output_dir}")
            return output_dir

        # 否则用原文件所在目录
        source_dir = os.path.dirname(os.path.abspath(fpath))

        # 如果原文件在临时目录（Gradio上传），用默认 output 目录
        if FileUtils._is_system_temp_dir(source_dir):
            output_root = os.path.join(str(PROJECT_ROOT), "output")
            os.makedirs(output_root, exist_ok=True)
            filename_base = os.path.splitext(os.path.basename(fpath))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_root, f"{filename_base}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"检测到 Gradio 临时目录，默认输出到: {output_dir}")
            return output_dir

        logger.info(f"使用原文件目录作为输出目录: {source_dir}")
        return source_dir

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
    def _copy_with_retry(src: str, dst: str, max_retries: int = 3) -> None:
        """带重试的文件复制，应对 Windows 文件锁定（杀毒软件、Explorer 缩略图等）"""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                shutil.copy2(src, dst)
                return
            except PermissionError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"文件被占用，{attempt}/{max_retries} 次重试，等待 {attempt} 秒: {src}")
                    time.sleep(attempt)
        raise PermissionError(
            f"无法复制文件，文件可能被其他程序占用（请关闭视频播放器、文件管理器等）:\n"
            f"  源文件: {src}\n"
            f"  目标: {dst}\n"
            f"  已重试 {max_retries} 次均失败"
        ) from last_error

    @staticmethod
    def prepare_single_input_dirs(file_paths: list, target_filename: str,
                                  output_dir: Optional[str] = None) -> List[Tuple[str, str]]:
        """为每个输入文件创建临时工作目录，将文件复制进去

        Args:
            output_dir: 用户指定的输出目录。如果指定，回拷目标使用此目录；
                        否则使用原文件所在目录；如果原文件在临时目录，
                        则使用默认 output 目录。

        Returns:
            List[Tuple[str, str]]: 每个元素为 (working_dir, source_dir)，source_dir 是回拷目标目录
        """
        dirs = []
        for file_path in file_paths:
            fpath = file_path.name if hasattr(file_path, 'name') else file_path
            if not fpath or not os.path.exists(fpath):
                logger.warning(f"文件不存在，跳过: {fpath}")
                continue
            base = os.path.splitext(os.path.basename(fpath))[0]
            working_dir = FileUtils.create_temp_working_dir(base)
            FileUtils._copy_with_retry(fpath, os.path.join(working_dir, target_filename))
            source_dir = FileUtils._resolve_output_dir(fpath, output_dir)
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
            FileUtils._copy_with_retry(fpath, os.path.join(working_dir, target_filename))
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
