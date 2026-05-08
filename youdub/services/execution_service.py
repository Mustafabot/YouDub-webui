import os
from typing import Callable, List, Optional
from loguru import logger
from ..log_config import get_log_buffer, clear_log_buffer
from .error_handler import ErrorHandler
from .file_utils import FileUtils


class ExecutionService:

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.file_utils = FileUtils()

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
            folder = self.file_utils.resolve_folder_path(folder)
            if not os.path.exists(folder):
                return self.error_handler.format_error(
                    f"文件夹不存在：{folder}",
                    ["输入的文件夹路径不正确", "文件夹尚未创建"],
                    ["检查路径拼写是否正确", "确认文件夹是否已创建", "先运行前置步骤生成文件夹"]
                )

        clear_log_buffer()
        try:
            result = module_function(folder, **kwargs)
            output = f"✅ {result}" if result and not str(result).startswith("❌") else result
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
        selected_folders = self.file_utils.merge_folder_lists(folder_list_text, folder_select_files)

        if selected_folders:
            clear_log_buffer()
            try:
                result = batch_function(selected_folders, **kwargs)
                output = f"✅ {result}" if result and not str(result).startswith("❌") else result
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

    def wrap_with_logs(self, func: Callable, *args, **kwargs) -> str:
        clear_log_buffer()
        try:
            result = func(*args, **kwargs)
            logs = get_log_buffer()
            if logs:
                return f"{logs}\n\n{result}"
            return result
        except Exception as e:
            logs = get_log_buffer()
            logger.error(f"执行失败: {e}")
            error_result = self.error_handler.classify_error(e)
            if logs:
                return f"{logs}\n\n{error_result}"
            return error_result
