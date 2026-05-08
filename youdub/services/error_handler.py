import re
from typing import List, Tuple
from loguru import logger


class ErrorHandler:

    def classify_error(self, error: Exception) -> str:
        msg = str(error).lower()
        if self._is_network_error(msg):
            return self._format_network_error()
        if self._is_cuda_oom_error(msg):
            return self._format_cuda_oom_error()
        if self._is_memory_error(msg):
            return self._format_memory_error()
        if self._is_api_key_error(msg):
            return self._format_api_key_error()
        if self._is_file_not_found_error(msg):
            return self._format_file_not_found_error()
        if self._is_ffmpeg_error(msg):
            return self._format_ffmpeg_error()
        if self._is_dependency_error(msg):
            return self._format_dependency_error()
        return self._format_unknown_error(error)

    def format_error(self, desc: str, causes: List[str], suggestions: List[str]) -> str:
        lines = [f"❌ 操作失败：{desc}", "", "可能的原因："]
        for c in causes:
            lines.append(f"- {c}")
        lines.append("")
        lines.append("建议：")
        for s in suggestions:
            lines.append(f"- {s}")
        return "\n".join(lines)

    def _is_network_error(self, msg: str) -> bool:
        return any(kw in msg for kw in ['connection', 'network', 'timeout', 'urlopen', 'http', 'ssl'])

    def _is_cuda_oom_error(self, msg: str) -> bool:
        cuda_oom_patterns = [
            r'cuda.*out of memory',
            r'out of memory.*cuda',
            r'cudnn_status_alloc_failed',
            r'cuda.*alloc.*failed',
            r'alloc.*failed.*cuda'
        ]
        return any(re.search(p, msg) for p in cuda_oom_patterns) or ('oom' in msg and 'cuda' in msg)

    def _is_memory_error(self, msg: str) -> bool:
        return 'out of memory' in msg or 'oom' in msg

    def _is_api_key_error(self, msg: str) -> bool:
        return any(kw in msg for kw in ['api key', 'api_key', 'unauthorized', 'invalid api', 'authentication', '401', '403'])

    def _is_file_not_found_error(self, msg: str) -> bool:
        return any(kw in msg for kw in ['no such file', 'not found', 'filenotfound', 'does not exist'])

    def _is_ffmpeg_error(self, msg: str) -> bool:
        return 'winerror 2' in msg

    def _is_dependency_error(self, msg: str) -> bool:
        return any(kw in msg for kw in ['numba needs numpy', 'numba.*numpy', 'numpy.*numba'])

    def _format_network_error(self) -> str:
        from ..config import check_network
        if not check_network():
            return self.format_error(
                "离线模式下模型加载失败",
                ["当前处于离线模式，无法从网络下载模型或检查更新", "模型可能未完整下载到本地缓存", "模型缓存路径与加载路径不匹配"],
                ["请先连接网络并在「模型管理」中下载所有必要模型", "确认模型状态显示为全部已下载", "如问题持续，尝试删除模型缓存后重新下载"]
            )
        return self.format_error(
            "网络连接错误",
            ["网络连接不稳定或无法访问目标服务器", "代理设置不正确", "目标服务器暂时不可用"],
            ["检查网络连接是否正常", "如使用代理，请确认代理设置正确", "稍后重试"]
        )

    def _format_cuda_oom_error(self) -> str:
        return self.format_error(
            "CUDA 显存不足",
            ["GPU 显存不足以运行当前模型", "模型过大或批处理大小过大", "显存碎片化导致即使显存足够也可能分配失败", "上一个步骤的模型未释放显存"],
            ["在设置中选择更小的模型（如 medium 或 small）", "减小批处理大小（Batch Size），建议从 8 开始测试", "将计算设备切换为 CPU 模式", "关闭其他占用 GPU 程序（如浏览器、游戏）", "重启程序以完全重置显存状态"]
        )

    def _format_memory_error(self) -> str:
        return self.format_error(
            "系统内存不足",
            ["系统内存不足以运行当前任务", "同时运行的程序过多"],
            ["关闭其他占用内存的程序", "减小模型或批处理大小", "增加系统虚拟内存"]
        )

    def _format_api_key_error(self) -> str:
        return self.format_error(
            "API Key 无效或未设置",
            ["API Key 未填写或已过期", "API Key 格式不正确", "API Base URL 配置错误"],
            ["前往设置页面检查并填写正确的 API Key", "确认 API Key 是否仍然有效", "检查 API Base URL 是否正确"]
        )

    def _format_file_not_found_error(self) -> str:
        return self.format_error(
            "文件或路径不存在",
            ["指定的文件夹路径不正确", "文件已被移动或删除"],
            ["检查输入的路径是否正确", "确认文件是否存在于指定位置"]
        )

    def _format_ffmpeg_error(self) -> str:
        return self.format_error(
            "系统找不到指定的可执行文件",
            ["FFmpeg 可能未安装或不在系统 PATH 中", "依赖的外部工具（如 ffmpeg）未找到"],
            ["运行 python scripts/download_ffmpeg.py 自动下载安装 FFmpeg", "检查配置中的 FFMPEG_PATH 是否正确设置", "查看控制台日志获取详细错误信息"]
        )

    def _format_dependency_error(self) -> str:
        return self.format_error(
            "依赖库版本不兼容",
            ["Numba 与 NumPy 版本不兼容", "NumPy 版本过高，超出 Numba 支持范围", "通常由 pip 自动升级导致"],
            [
                "在终端执行修复命令：pip install \"numpy<2.4\" \"numba>=0.63\"",
                "或升级 Numba：pip install \"numba>=0.64\"",
                "如问题持续，删除 venv 后重新运行 setup_windows.bat"
            ]
        )

    def _format_unknown_error(self, error: Exception) -> str:
        return self.format_error(
            f"发生未知错误：{str(error)}",
            ["程序运行过程中遇到了未预期的错误"],
            ["查看控制台日志获取详细错误信息", "检查所有配置是否正确", "如问题持续，请提交 Issue 反馈"]
        )
