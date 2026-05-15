import os
import shutil
from typing import Callable, List, Optional, Tuple
from loguru import logger
from ..log_config import get_log_buffer, clear_log_buffer
from .error_handler import ErrorHandler
from .file_utils import FileUtils


class ExecutionService:

    def __init__(self):
        self.error_handler = ErrorHandler()

    def _format_output(self, result) -> str:
        """统一格式化模块返回值，确保始终返回 str"""
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

        clear_log_buffer()
        try:
            result = module_function(folder, **kwargs)
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
        selected_folders = FileUtils.merge_folder_lists(folder_list_text, folder_select_files)

        if selected_folders:
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
        if not file_paths:
            return self.error_handler.format_error(
                f"未选择文件",
                [f"{module_name}需要选择输入文件"],
                ["请选择对应的输入文件"]
            ), []

        working_dir_pairs = FileUtils.prepare_single_input_dirs(file_paths, target_filename, output_dir)
        if not working_dir_pairs:
            return self.error_handler.format_error(
                "文件准备失败",
                ["无法为选中的文件创建工作目录"],
                ["请检查文件是否存在且可读"]
            ), []

        working_dirs = [pair[0] for pair in working_dir_pairs]

        clear_log_buffer()
        try:
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
        required_keys = [k for k, v in file_map.items() if v is None]
        if required_keys:
            return self.error_handler.format_error(
                "未选择必要的输入文件",
                [f"缺少以下输入文件: {', '.join(required_keys)}"],
                ["请选择所有必要的输入文件"]
            ), []

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

        working_dir = FileUtils.prepare_multi_input_dir(file_map)

        clear_log_buffer()
        try:
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
            try:
                shutil.rmtree(working_dir, ignore_errors=True)
            except Exception:
                pass

    def wrap_with_logs(self, func: Callable, *args, **kwargs) -> str:
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
