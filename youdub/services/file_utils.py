"""
file_utils.py — 文件格式验证与临时目录管理模块

本模块提供 YouDub 流水线中的文件处理工具，包括：
    1. 文件格式验证（根据 INPUT_FILE_FORMATS 白名单）
    2. 临时工作目录的创建与管理
    3. 文件的复制、回拷（带 Windows 文件锁定重试）
    4. 文件夹路径解析（相对/绝对路径、多文件夹合并）

设计要点：
    - 所有方法均为静态方法，通过 FileUtils 类组织
    - 支持 Gradio 上传文件的临时目录检测（_is_system_temp_dir）
    - 用户指定输出目录优先，否则使用原文件所在目录
    - Windows 环境下文件可能被其他进程锁定，提供带重试的复制
"""

import os
import shutil
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Tuple
from loguru import logger
from ..config import PROJECT_ROOT


# ──────────────────────────────────────────────
# 输入文件格式白名单
# ──────────────────────────────────────────────

INPUT_FILE_FORMATS = {
    # 视频文件
    "download.mp4": {
        "extensions": [".mp4", ".avi", ".mkv", ".mov", ".flv"],
        "description": "视频文件"
    },
    # 人声分离后的人声音频
    "audio_vocals.wav": {
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac"],
        "description": "人声音频文件"
    },
    # 人声分离后的伴奏音频
    "audio_instruments.wav": {
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac"],
        "description": "伴奏音频文件"
    },
    # WhisperX 识别结果 JSON
    "transcript.json": {
        "extensions": [".json"],
        "description": "识别结果文件"
    },
    # yt-dlp 下载的视频元信息
    "download.info.json": {
        "extensions": [".json"],
        "description": "视频信息文件"
    },
    # LLM 翻译结果 JSON
    "translation.json": {
        "extensions": [".json"],
        "description": "翻译结果文件"
    },
    # LLM 摘要 JSON
    "summary.json": {
        "extensions": [".json"],
        "description": "摘要文件"
    },
    # 合成后的音频（非 TTS）
    "audio_combined.wav": {
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac"],
        "description": "合成音频文件"
    },
    # TTS 输出音频
    "audio_tts.wav": {
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac"],
        "description": "TTS输出文件"
    },
    # 最终合成的视频
    "video.mp4": {
        "extensions": [".mp4", ".avi", ".mkv", ".mov"],
        "description": "合成视频文件"
    },
    # 视频标题/描述文本
    "video.txt": {
        "extensions": [".txt"],
        "description": "视频标题描述文件"
    },
    # 视频封面图片
    "video.png": {
        "extensions": [".png", ".jpg", ".jpeg"],
        "description": "视频封面文件"
    },
    # YouTube 缩略图
    "download.jpg": {
        "extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        "description": "缩略图文件"
    },
}


class FileUtils:
    """文件工具类

    提供文件格式验证、临时目录创建、文件复制回拷、路径解析等
    静态工具方法，不维护实例状态。
    """

    @staticmethod
    def validate_file_format(file_path: str, expected_filename: str) -> Tuple[bool, str]:
        """验证文件格式是否符合预期

        根据 INPUT_FILE_FORMATS 白名单检查文件扩展名。
        如果 expected_filename 不在白名单中，视为无验证规则，直接返回通过。

        Args:
            file_path: 待验证的文件路径
            expected_filename: 期望的文件名键（如 "download.mp4"）

        Returns:
            Tuple[bool, str]: (是否通过, 验证结果描述)
        """
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
        """将用户选择的文件复制到指定文件夹并重命名

        用于文件上传后将其整理到工作目录中。

        Args:
            src_file: 源文件路径
            dest_folder: 目标文件夹路径
            target_filename: 目标文件名（复制后重命名为此）

        Returns:
            Tuple[bool, str]: (是否成功, 结果描述)
        """
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
        """获取文件名的显示标签（带中文描述）

        Args:
            filename: 文件名键（如 "download.mp4"）

        Returns:
            str: 显示标签（如 "download.mp4 (视频文件)"）
        """
        format_info = INPUT_FILE_FORMATS.get(filename)
        if format_info:
            return f"{filename} ({format_info['description']})"
        return filename

    @staticmethod
    def create_selected_files_dict() -> Dict[str, Optional[str]]:
        """创建初始化的文件选择字典

        以 INPUT_FILE_FORMATS 的所有键为键，值全部初始化为 None。

        Returns:
            Dict[str, Optional[str]]: 文件名到路径的映射，初始全部为 None
        """
        return {filename: None for filename in INPUT_FILE_FORMATS.keys()}

    @staticmethod
    def update_file_selection(selected_files: Dict[str, Optional[str]],
                              filename: str, filepath: str) -> Dict[str, Optional[str]]:
        """更新单个文件的选中路径

        如果传入的 selected_files 为 None，则先创建新字典。

        Args:
            selected_files: 当前文件选择字典
            filename: 要更新的文件名键
            filepath: 新的文件路径

        Returns:
            Dict[str, Optional[str]]: 更新后的文件选择字典
        """
        if selected_files is None:
            selected_files = FileUtils.create_selected_files_dict()
        selected_files[filename] = filepath
        return selected_files

    @staticmethod
    def format_selected_files_status(selected_files: Dict[str, Optional[str]]) -> str:
        """格式化文件选择状态为可读文本

        用于在 UI 中显示当前各文件的选择情况。

        Args:
            selected_files: 文件选择字典

        Returns:
            str: 格式化的选择状态文本（含 ✅/❌ 标记）
        """
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

    # ──────────────────────────────────────────
    # 临时 / 输出目录解析
    # ──────────────────────────────────────────

    @staticmethod
    def _is_system_temp_dir(path: str) -> bool:
        """检测路径是否位于系统临时目录（如 Gradio 上传目录）

        Gradio 上传的文件会放在系统的临时目录中，该函数通过比较
        规范化后的路径前缀来判断。使用 os.path.normcase 确保
        大小写不敏感的 Windows 路径比较正确。

        Args:
            path: 待检测的路径

        Returns:
            bool: 在系统临时目录中返回 True
        """
        temp_dir = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
        path_norm = os.path.normcase(os.path.abspath(path))
        if path_norm == temp_dir:
            return True
        return path_norm.startswith(temp_dir + os.sep)

    @staticmethod
    def _resolve_output_dir(fpath: str, user_output_dir: Optional[str] = None) -> str:
        """解析最终输出目录

        目录优先级：
            1. 用户指定的输出目录（user_output_dir）— 直接使用，不创建子目录
            2. 原文件所在目录
            3. 如果原文件在系统临时目录中 -> 使用 PROJECT_ROOT/output/{filename}_{timestamp}

        Args:
            fpath: 输入文件路径
            user_output_dir: 用户指定的输出目录（可选）

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
        """在 .temp 下创建唯一临时工作目录

        为避免多任务并行时的目录冲突，目录名使用 prefix + 随机 UUID 前缀。
        所有临时目录统一放在项目根目录的 .temp/ 下。

        Args:
            prefix: 目录名前缀（通常使用输入文件的基础名）

        Returns:
            str: 创建的临时目录绝对路径
        """
        temp_root = os.path.join(str(PROJECT_ROOT), ".temp")
        os.makedirs(temp_root, exist_ok=True)
        dir_name = f"{prefix}_{uuid.uuid4().hex[:8]}"
        temp_dir = os.path.join(temp_root, dir_name)
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"创建临时工作目录: {temp_dir}")
        return temp_dir

    # ──────────────────────────────────────────
    # 文件复制（带重试）
    # ──────────────────────────────────────────

    @staticmethod
    def _copy_with_retry(src: str, dst: str, max_retries: int = 3) -> None:
        """带重试的文件复制

        在 Windows 环境下，文件可能被杀毒软件、Explorer 缩略图生成、
        视频播放器等进程临时锁定。本函数在被 PermissionError 阻挡时
        会等待并重试，间隔递增（1s, 2s, 3s）。

        Args:
            src: 源文件路径
            dst: 目标文件路径
            max_retries: 最大重试次数（默认 3）

        Raises:
            PermissionError: 所有重试均失败时抛出，附详细指引
        """
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

    # ──────────────────────────────────────────
    # 批量输入 / 输出文件管理
    # ──────────────────────────────────────────

    @staticmethod
    def prepare_single_input_dirs(file_paths: list, target_filename: str,
                                  output_dir: Optional[str] = None) -> List[Tuple[str, str]]:
        """为每个输入文件创建独立的临时工作目录

        对于文件列表中的每个文件：
            1. 创建唯一临时工作目录
            2. 将文件带重试复制到临时目录中（统一命名为 target_filename）
            3. 解析输出目录路径

        这种"一文件一目录"设计是为了支持批量处理时各任务互不干扰。

        Args:
            file_paths: 输入文件路径列表（支持 Gradio FileData 对象）
            target_filename: 复制到工作目录后的统一目标文件名
            output_dir: 用户指定的输出目录（可选）

        Returns:
            List[Tuple[str, str]]: 每个元素为 (working_dir, source_dir)，
                                   其中 source_dir 是后续回拷文件的目标目录
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
        """将多个输入文件复制到同一个临时工作目录

        用于"多文件输入单次处理"的场景（如同时需要视频文件和字幕文件）。
        所有文件按 file_map 的键值对复制到同一个临时目录中。

        Args:
            file_map: {目标文件名: 源文件路径} 的映射

        Returns:
            str: 创建的临时工作目录路径
        """
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
        """收集临时工作目录中的所有输出文件

        按文件名排序后返回完整路径列表。仅收集文件（不含子目录）。

        Args:
            working_dir: 临时工作目录路径

        Returns:
            List[str]: 文件路径列表（按文件名排序）
        """
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
        """将工作目录中的处理结果复制回源目录

        这是"处理-回拷"模式的关键步骤：模块在临时目录中处理，
        处理完成后将新产生的文件（排除输入文件本身）复制回源目录。

        Args:
            working_dir: 临时工作目录
            source_dir: 原始源文件目录（回拷目标）
            input_filenames: 输入文件名列表（这些文件不会被回拷，避免覆盖原文件）

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

    # ──────────────────────────────────────────
    # 文件夹路径解析
    # ──────────────────────────────────────────

    @staticmethod
    def resolve_folder_path(folder: str) -> str:
        """将文件夹路径解析为绝对路径

        如果路径已经是绝对路径或为空，直接返回。
        如果是相对路径，则相对于 PROJECT_ROOT 进行解析。

        Args:
            folder: 文件夹路径（可能为相对路径）

        Returns:
            str: 解析后的绝对路径
        """
        if not folder or os.path.isabs(folder):
            return folder
        resolved = str(PROJECT_ROOT / folder)
        return resolved

    @staticmethod
    def extract_folders_from_files(file_paths: list) -> List[str]:
        """从文件路径列表中提取唯一文件夹路径

        主要用于处理 Gradio 的文件选择器组件，提取用户所选文件
        所在目录的绝对路径。

        Args:
            file_paths: Gradio 文件组件返回的文件列表

        Returns:
            List[str]: 排序后的唯一文件夹绝对路径列表
        """
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
        """解析文本框中输入的每行一个的文件夹列表

        用户可以在 UI 文本框中手动输入多个文件夹路径（每行一个），
        此函数将其拆分为绝对路径列表。

        Args:
            folder_text: 多行文本，每行一个文件夹路径

        Returns:
            List[str]: 解析后的绝对路径列表
        """
        if not folder_text or not folder_text.strip():
            return []
        lines = [line.strip() for line in folder_text.split('\n') if line.strip()]
        return [os.path.abspath(line) for line in lines]

    @staticmethod
    def merge_folder_lists(folder_text: str, selected_files: list) -> List[str]:
        """合并文本框输入的文件夹列表和文件选择器中的文件夹

        两种文件夹来源：
            1. 用户在文本框中手动输入的多行路径
            2. 从文件选择器中提取的路径

        取并集并去重排序。

        Args:
            folder_text: 多行文本文件夹输入
            selected_files: 文件选择器选中的文件列表

        Returns:
            List[str]: 合并去重后的文件夹绝对路径列表
        """
        folders = set(FileUtils.resolve_folder_list_text(folder_text))
        folders.update(FileUtils.extract_folders_from_files(selected_files))
        return sorted(folders)
