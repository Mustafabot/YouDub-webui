"""
execution_service.py — 模块执行包装服务

本模块是 Gradio UI 与底层处理模块之间的桥梁，提供统一的执行包装：
    1. 参数验证（文件夹是否存在、文件是否选择）
    2. 工作目录自动管理（创建临时目录、复制输入文件、回拷输出文件）
    3. 异常捕获与格式化（调用 error_handler 将异常转为中文提示）
    4. 日志收集（在返回消息中包含日志缓冲区内容）
    5. 清理函数支持（失败时自动执行清理）

核心组件：
    - ExecutionService 类：提供 execute_with_wrapper、execute_with_folders_support、
      execute_batch_with_files、execute_single_with_files、wrap_with_logs 五种执行模式。

设计模式：
    - 每个 public 方法都是一个"包装器"，接收模块函数和参数，
      在统一的骨架（验证 -> 执行 -> 格式化 -> 清理）中运行。
    - 使用 log_config 的日志缓冲区机制来捕获模块执行期间的日志输出。
"""

import os
import shutil
from typing import Callable, List, Optional, Tuple
from loguru import logger
from ..log_config import get_log_buffer, clear_log_buffer
from .error_handler import ErrorHandler
from .file_utils import FileUtils


class ExecutionService:
    """模块执行包装服务

    提供多种执行模式以适配不同的 Gradio UI 交互场景。
    所有 public 方法均返回可用于 Gradio 展示的字符串结果。
    """

    def __init__(self):
        """初始化执行服务

        创建 ErrorHandler 实例，用于异常分类和中文错误格式化。
        """
        self.error_handler = ErrorHandler()

    def _format_output(self, result) -> str:
        """统一格式化模块返回值，确保始终返回可用于 UI 展示的 str

        Gradio 组件需要接收字符串作为输出。此函数负责：
            - None 结果 -> "操作已完成"
            - 已带❌前缀 -> 原样返回
            - 其他 -> 添加✅前缀

        Args:
            result: 模块函数的原始返回值（可能为 None、str 或其他类型）

        Returns:
            str: 格式化后的 UI 展示文本
        """
        if not result:
            return "✅ 操作已完成"
        result_str = str(result)
        if result_str.startswith("❌"):
            return result_str
        return f"✅ {result_str}"

    def execute_with_wrapper(
        self,
        folder: str,
        module_function: Callable,
        module_name: str,
        cleanup_function: Optional[Callable] = None,
        validate_folder: bool = True,
        **kwargs
    ) -> str:
        """适用于单文件夹处理的执行包装器

        这是最基础的执行模式，流程：
            1. 验证文件夹路径（可选）
            2. 清空日志缓冲区
            3. 执行模块函数（传入文件夹路径和额外参数）
            4. 格式化结果为 UI 展示文本
            5. 异常时：执行清理函数 + 分类错误并返回中文提示

        Args:
            folder: 文件夹路径
            module_function: 要执行的模块函数（签名：fn(folder, **kwargs) -> str）
            module_name: 模块名称（用于日志记录）
            cleanup_function: 失败时的清理函数（可选，无参数）
            validate_folder: 是否验证文件夹存在（默认 True）
            **kwargs: 传递给模块函数的额外参数

        Returns:
            str: 格式化后的结果（成功含✅标志，失败含❌和中文提示和日志）
        """
        # 参数验证阶段
        if validate_folder:
            if not folder or not folder.strip():
                return self.error_handler.format_error(
                    "未填写文件夹路径",
                    ["文件夹路径为空"],
                    ["请输入有效的文件夹路径"]
                )
            folder = FileUtils.resolve_folder_path(folder)
            if not os.path.exists(folder):
                return self.error_handler.format_error(
                    f"文件夹不存在：{folder}",
                    ["输入的文件夹路径不正确", "文件夹尚未创建"],
                    ["检查路径拼写是否正确", "确认文件夹是否已创建", "先运行前置步骤生成文件夹"]
                )

        # 清空日志缓冲区，确保之前的日志不会混入本次执行结果
        clear_log_buffer()
        try:
            # 执行实际模块函数
            result = module_function(folder, **kwargs)
            output = self._format_output(result)
            # 收集执行期间的日志内容
            logs = get_log_buffer()
            if logs:
                return f"{logs}\n\n{output}"
            return output
        except Exception as e:
            # 异常处理：记录错误 + 执行清理 + 中文错误分类
            logger.error(f"{module_name}失败: {e}")
            if cleanup_function:
                try:
                    cleanup_function()
                except Exception as cleanup_error:
                    logger.warning(f"清理失败: {cleanup_error}")

            logs = get_log_buffer()
            error_result = self.error_handler.classify_error(e)
            if logs:
                return f"{logs}\n\n{error_result}"
            return error_result

    def execute_with_folders_support(
        self,
        folder: str,
        folder_list_text: Optional[str],
        folder_select_files,
        batch_function: Callable,
        single_function: Callable,
        module_name: str,
        cleanup_function: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """支持多文件夹/单文件夹的弹性执行包装器

        如果用户通过文本框或文件选择器指定了多个文件夹，则自动切换到
        批量处理模式（batch_function）；否则退回到单文件夹模式
        （execute_with_wrapper 调用 single_function）。

        Args:
            folder: 单个文件夹路径（文本框输入）
            folder_list_text: 多行文本文件夹列表（可选）
            folder_select_files: 文件选择器选中的文件（用于提取文件夹）
            batch_function: 批量处理函数（签名：fn(folders_list, **kwargs)）
            single_function: 单文件夹处理函数
            module_name: 模块名称
            cleanup_function: 清理函数（可选）
            **kwargs: 额外参数

        Returns:
            str: 格式化后的执行结果
        """
        # 合并所有来源的文件夹列表
        selected_folders = FileUtils.merge_folder_lists(folder_list_text, folder_select_files)

        if selected_folders:
            # 有多文件夹时走批量路径
            clear_log_buffer()
            try:
                result = batch_function(selected_folders, **kwargs)
                output = self._format_output(result)
                logs = get_log_buffer()
                if logs:
                    return f"{logs}\n\n{output}"
                return output
            except Exception as e:
                logger.error(f"{module_name}失败: {e}")
                if cleanup_function:
                    try:
                        cleanup_function()
                    except Exception as cleanup_error:
                        logger.warning(f"清理失败: {cleanup_error}")

                logs = get_log_buffer()
                error_result = self.error_handler.classify_error(e)
                if logs:
                    return f"{logs}\n\n{error_result}"
                return error_result

        # 没有多文件夹时退化到单文件夹模式
        return self.execute_with_wrapper(
            folder, single_function, module_name, cleanup_function, **kwargs
        )

    def execute_batch_with_files(
        self,
        module_name: str,
        batch_function: Callable,
        file_paths: list,
        target_filename: str,
        output_dir: Optional[str] = None,
        cleanup_function: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[str, List[str]]:
        """"文件输入 + 批量处理"执行模式

        适用于"用户选择多个同类型文件进行批量处理"的场景：
            1. 为每个文件创建独立临时工作目录
            2. 将文件复制到工作目录（统一命名）
            3. 在每个目录上执行批量处理
            4. 将输出文件回拷到原始文件所在目录
            5. 清理临时目录

        Args:
            module_name: 模块名称
            batch_function: 批量处理函数（签名：fn(working_dirs_list, **kwargs)）
            file_paths: 输入文件路径列表
            target_filename: 复制到工作目录后的统一目标文件名
            output_dir: 用户指定的输出目录（可选）
            cleanup_function: 清理函数（可选）
            **kwargs: 额外参数

        Returns:
            Tuple[str, List[str]]: (格式化结果文本, 输出文件路径列表)
        """
        # 验证文件是否已选择
        if not file_paths:
            return self.error_handler.format_error(
                f"未选择文件",
                [f"{module_name}需要选择输入文件"],
                ["请选择对应的输入文件"]
            ), []

        # 准备临时工作目录（每个文件一个独立目录）
        working_dir_pairs = FileUtils.prepare_single_input_dirs(file_paths, target_filename, output_dir)
        if not working_dir_pairs:
            return self.error_handler.format_error(
                "文件准备失败",
                ["无法为选中的文件创建工作目录"],
                ["请检查文件是否存在且可读"]
            ), []

        # 提取所有工作目录路径
        working_dirs = [pair[0] for pair in working_dir_pairs]

        clear_log_buffer()
        try:
            # 执行批量处理
            result = batch_function(working_dirs, **kwargs)
            output_files = []
            for (working_dir, source_dir) in working_dir_pairs:
                # 将处理产生的输出文件回拷到原始源目录，并收集源目录中的路径
                copied = FileUtils.copy_output_files_back(
                    working_dir, source_dir, input_filenames=[target_filename]
                )
                output_files.extend(copied)
            output = self._format_output(result)
            logs = get_log_buffer()
            if logs:
                output = f"{logs}\n\n{output}"
            return output, output_files
        except Exception as e:
            logger.error(f"{module_name}失败: {e}")
            if cleanup_function:
                try:
                    cleanup_function()
                except Exception as cleanup_error:
                    logger.warning(f"清理失败: {cleanup_error}")
            logs = get_log_buffer()
            error_result = self.error_handler.classify_error(e)
            if logs:
                return f"{logs}\n\n{error_result}", []
            return error_result, []
        finally:
            # 无论成功或失败，清理所有临时工作目录
            for working_dir, _ in working_dir_pairs:
                try:
                    shutil.rmtree(working_dir, ignore_errors=True)
                except Exception:
                    pass

    def execute_single_with_files(
        self,
        module_name: str,
        single_function: Callable,
        file_map: dict,
        output_dir: Optional[str] = None,
        cleanup_function: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[str, List[str]]:
        """"文件输入 + 单次处理"执行模式

        适用于"用户选择多个不同类型的文件进行单次处理"的场景（如TTS需要
        同时输入 transcript.json 和 speaker 参考音频）：
            1. 检查所有必要输入文件是否已选择
            2. 将所有文件复制到同一个临时工作目录
            3. 执行单次处理
            4. 将输出文件回拷（优先用户指定目录，否则取公共父目录）
            5. 清理临时目录

        Args:
            module_name: 模块名称
            single_function: 单次处理函数（签名：fn(working_dir, **kwargs)）
            file_map: {目标文件名: 源文件路径} 的映射
            output_dir: 用户指定的输出目录（可选）
            cleanup_function: 清理函数（可选）
            **kwargs: 额外参数

        Returns:
            Tuple[str, List[str]]: (格式化结果文本, 输出文件路径列表)
        """
        # 检查是否有未选择的必要输入文件
        required_keys = [k for k, v in file_map.items() if v is None]
        if required_keys:
            return self.error_handler.format_error(
                "未选择必要的输入文件",
                [f"缺少以下输入文件: {', '.join(required_keys)}"],
                ["请选择所有必要的输入文件"]
            ), []

        # 确定输出目录
        # 用户指定输出目录优先
        if output_dir and output_dir.strip():
            source_dir = FileUtils.resolve_folder_path(output_dir)
            os.makedirs(source_dir, exist_ok=True)
            logger.info(f"使用用户指定的输出目录: {source_dir}")
        else:
            # 确定原始源目录：取所有输入文件的公共父目录
            source_paths = []
            for filepath in file_map.values():
                if filepath:
                    fpath = filepath.name if hasattr(filepath, 'name') else filepath
                    if fpath and os.path.exists(fpath):
                        source_paths.append(os.path.dirname(os.path.abspath(fpath)))
            source_dir = None
            if source_paths:
                try:
                    source_dir = os.path.commonpath(source_paths)
                except ValueError:
                    logger.warning("输入文件跨盘符，输出将回拷到第一个文件所在目录")
                    source_dir = source_paths[0]

        # 将所有输入文件复制到同一个临时工作目录
        working_dir = FileUtils.prepare_multi_input_dir(file_map)

        clear_log_buffer()
        try:
            # 执行单次处理
            result = single_function(working_dir, **kwargs)
            # 将处理产生的输出文件回拷到原始源目录，并收集源目录中的路径
            if source_dir:
                output_files = FileUtils.copy_output_files_back(
                    working_dir, source_dir, input_filenames=list(file_map.keys())
                )
            else:
                output_files = FileUtils.collect_output_files(working_dir)
            output = self._format_output(result)
            logs = get_log_buffer()
            if logs:
                output = f"{logs}\n\n{output}"
            return output, output_files
        except Exception as e:
            logger.error(f"{module_name}失败: {e}")
            if cleanup_function:
                try:
                    cleanup_function()
                except Exception as cleanup_error:
                    logger.warning(f"清理失败: {cleanup_error}")
            logs = get_log_buffer()
            error_result = self.error_handler.classify_error(e)
            if logs:
                return f"{logs}\n\n{error_result}", []
            return error_result, []
        finally:
            # 无论成功或失败，清理临时工作目录
            try:
                shutil.rmtree(working_dir, ignore_errors=True)
            except Exception:
                pass

    def wrap_with_logs(self, func: Callable, *args, **kwargs) -> str:
        """最简单的执行包装器：捕获函数执行期间的所有日志和异常

        适用于不需要文件管理、不需要工作目录管理的简单函数调用。
        本质是 try-except + 日志缓冲区的标准模板。

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            str: 格式化后的执行结果（含日志）
        """
        clear_log_buffer()
        try:
            result = func(*args, **kwargs)
            output = self._format_output(result)
            logs = get_log_buffer()
            if logs:
                return f"{logs}\n\n{output}"
            return output
        except Exception as e:
            logs = get_log_buffer()
            logger.error(f"执行失败: {e}")
            error_result = self.error_handler.classify_error(e)
            if logs:
                return f"{logs}\n\n{error_result}"
            return error_result
