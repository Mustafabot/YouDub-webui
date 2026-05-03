import sys
import os
from collections import deque
from loguru import logger

from .config import PROJECT_ROOT

LOG_FILE = PROJECT_ROOT / "youdub.log"

FORMAT_FILE = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
FORMAT_CONSOLE = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

_log_buffer = deque(maxlen=1000)

_INITIALIZED = False


def _buffer_sink(message):
    record = message.record
    formatted = (
        f"{record['time'].strftime('%H:%M:%S.%f')[:-3]} | "
        f"{str(record['level']): <8} | "
        f"{record['name']}:{record['line']} - "
        f"{record['message']}"
    )
    _log_buffer.append(formatted)


def init_logging():
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    logger.remove()

    logger.add(
        sys.stderr,
        format=FORMAT_CONSOLE,
        level="DEBUG",
        colorize=True,
    )

    logger.add(
        LOG_FILE,
        format=FORMAT_FILE,
        level="DEBUG",
        rotation="10 MB",
        encoding="utf-8",
    )

    logger.add(
        _buffer_sink,
        level="INFO",
    )

    logger.debug(f"日志系统初始化完成，日志文件: {LOG_FILE}")


def get_log_buffer(clear=True):
    if not _log_buffer:
        return ""
    lines = list(_log_buffer)
    if clear:
        _log_buffer.clear()
    return "\n".join(lines)


def clear_log_buffer():
    _log_buffer.clear()
