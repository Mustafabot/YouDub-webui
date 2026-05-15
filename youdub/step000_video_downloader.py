# -*- coding: utf-8 -*-
"""
step000_video_downloader.py — 步骤0：视频下载/导入模块

本模块是 YouDub 流水线的第一个步骤，负责：
1. 从 YouTube URL 获取视频信息列表（支持单个视频和播放列表）
2. 根据视频信息下载视频和音频，并使用 FFmpeg 合并为 MP4
3. 导入本地视频文件到统一的目录结构中
4. 提供批量扫描目录进行下载的功能

输出目录结构：
    {folder_path}/{uploader}/{upload_date} {title}/
        download.mp4          # 下载/导入的视频文件
        download.info.json    # 视频元数据信息
        download.thumbnail    # 视频缩略图（可选）

输出供步骤01（Demucs 人声分离）使用。
"""

import os          # 文件和路径操作
import re          # 正则表达式，用于清理标题中的非法字符
import json        # JSON 读写，用于保存/读取视频元数据
import shutil      # 文件复制，用于导入本地视频
from datetime import datetime    # 日期时间处理，用于获取文件修改日期
from pathlib import Path         # 面向对象的文件路径操作
from loguru import logger        # 结构化日志记录
import yt_dlp      # YouTube 下载库，支持视频信息提取和下载

# 导入项目配置模块：确保 FFmpeg 可用、获取 FFmpeg 路径、项目根目录
from .config import ensure_ffmpeg_available, get_ffmpeg_path, PROJECT_ROOT


def sanitize_title(title):
    """
    清理标题字符串，移除不合法的文件名字符。

    只保留中文字符、字母、数字、空格、下划线和连字符，
    并将连续多个空格替换为单个空格。

    Args:
        title: 原始标题字符串

    Returns:
        清理后的安全标题字符串
    """
    # 只保留中文字符(一-鿿)、字母数字(\w)、空格和横线
    title = re.sub(r'[^\w一-鿿 \d_-]', '', title)
    # 将连续多个空白字符替换为单个空格
    title = re.sub(r'\s+', ' ', title)
    return title


def get_target_folder(info, folder_path):
    """
    根据视频信息计算目标输出文件夹路径。

    路径格式：{folder_path}/{uploader}/{upload_date} {title}

    Args:
        info: 视频信息字典（来自 yt_dlp）
        folder_path: 基础存储目录

    Returns:
        完整的目标文件夹路径，若 upload_date 缺失则返回 None
    """
    # 清理标题和上传者名称中的非法字符
    sanitized_title = sanitize_title(info['title'])
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    # 获取上传日期（格式：YYYYMMDD）
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        # 没有上传日期时无法创建有序目录，返回 None
        return None

    # 构建输出文件夹路径
    output_folder = os.path.join(
        folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}')

    return output_folder


def download_single_video(info, folder_path, resolution='1080p'):
    """
    下载单个视频并合并为 MP4 文件。

    使用 yt_dlp 分别下载最佳视频和音频流，然后通过 FFmpeg 合并。
    如果目标文件已存在则跳过下载。

    Args:
        info: 视频信息字典（包含 webpage_url 等字段）
        folder_path: 基础存储目录
        resolution: 目标视频分辨率，如 '1080p'、'720p'

    Returns:
        输出文件夹路径，若 upload_date 缺失则返回 None

    Raises:
        RuntimeError: FFmpeg 不可用时抛出
        FileNotFoundError: 下载后未找到 MP4 文件时抛出
    """
    # 检查 FFmpeg 是否可用，不可用时尝试自动下载
    ffmpeg_available, ffmpeg_msg = ensure_ffmpeg_available(auto_download=True)
    if not ffmpeg_available:
        # FFmpeg 是视频合并的必须依赖，不可用时给出详细安装指引
        raise RuntimeError(
            f'FFmpeg 不可用，无法进行视频下载和合并。{ffmpeg_msg}\n'
            f'请按以下方式之一安装 FFmpeg：\n'
            f'1. 运行 python scripts/download_ffmpeg.py 自动下载\n'
            f'2. Windows: 从 https://ffmpeg.org/download.html 下载，解压后将 bin 目录添加到系统 PATH，或在配置中设置 FFMPEG_PATH\n'
            f'3. macOS: brew install ffmpeg\n'
            f'4. Linux: sudo apt install ffmpeg'
        )

    # 清理视频标题和上传者名称
    sanitized_title = sanitize_title(info['title'])
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    # 获取上传日期
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        # 缺少上传日期时无法确定目录名，返回 None
        return None

    # 构建输出文件夹路径
    output_folder = os.path.join(folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}')
    # 如果目标 MP4 文件已存在，跳过下载（断点续传支持）
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already downloaded in {output_folder}')
        return output_folder

    # 移除分辨率后缀中的 'p'（如 '1080p' -> '1080'）
    resolution = resolution.replace('p', '')
    # 获取 FFmpeg 可执行文件路径
    ffmpeg_path = get_ffmpeg_path()
    # yt_dlp 下载选项配置
    ydl_opts = {
        # 视频格式选择：优先最佳视频(MP4,≤指定分辨率)+最佳音频(M4A)，其次最佳MP4，最后最佳任意格式
        'format': f'bestvideo[ext=mp4][height<={resolution}]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'writeinfojson': True,           # 保存视频信息为 JSON 文件
        'writethumbnail': True,          # 下载视频缩略图
        'outtmpl': os.path.join(folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}', 'download'),  # 输出文件名模板（不含扩展名）
        'ignoreerrors': True,            # 忽略下载过程中的错误，继续处理
        'merge_output_format': 'mp4',    # 视频和音频合并输出为 MP4 格式
        'ffmpeg_location': ffmpeg_path,  # 指定 FFmpeg 路径
    }

    # 使用 yt_dlp 执行下载
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([info['webpage_url']])

    # 验证下载是否成功：检查 MP4 文件是否存在
    if not os.path.exists(os.path.join(output_folder, 'download.mp4')):
        raise FileNotFoundError(
            f'下载完成但未找到 download.mp4，请检查 ffmpeg 是否已安装且可用。'
            f'目录内容: {os.listdir(output_folder) if os.path.exists(output_folder) else "目录不存在"}'
        )

    logger.info(f'Video downloaded in {output_folder}')
    return output_folder


def download_videos(info_list, folder_path, resolution='1080p'):
    """
    批量下载多个视频。

    遍历视频信息列表，逐个调用 download_single_video 下载。

    Args:
        info_list: 视频信息字典列表
        folder_path: 基础存储目录
        resolution: 目标视频分辨率
    """
    for info in info_list:
        download_single_video(info, folder_path, resolution)


def get_info_list_from_url(url, num_videos):
    """
    从 YouTube URL 获取视频信息列表（生成器模式）。

    支持单个视频和播放列表，使用 yield 逐条返回视频信息，
    避免一次性加载所有视频信息占用过多内存。

    Args:
        url: YouTube URL 字符串或 URL 列表
        num_videos: 播放列表最大提取视频数

    Yields:
        视频信息字典，每条包含标题、上传者、上传日期等字段
    """
    # 如果传入的是单个 URL 字符串，转为列表统一处理
    if isinstance(url, str):
        url = [url]

    # 配置 yt_dlp 仅提取信息不下载
    ydl_opts = {
        'format': 'best',
        'dumpjson': True,             # 仅导出 JSON 信息，不下载
        'playlistend': num_videos,    # 限制播放列表提取数量
        'ignoreerrors': True          # 跳过出错的视频
    }

    # 逐个处理 URL，用生成器 yield 返回结果
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            # extract_info(download=False) 只获取元数据不下载
            result = ydl.extract_info(u, download=False)
            if 'entries' in result:
                # 'entries' 字段存在说明是播放列表
                for video_info in result['entries']:
                    yield video_info
            else:
                # 单个视频直接返回
                yield result


def download_from_url(url, folder_path, resolution='1080p', num_videos=5):
    """
    从 YouTube URL 下载视频的完整流程入口。

    先获取视频信息列表，再逐个下载。适合从单个 URL 或播放列表下载。

    Args:
        url: YouTube URL 字符串或 URL 列表
        folder_path: 基础存储目录
        resolution: 目标视频分辨率，如 '1080p'
        num_videos: 播放列表最大下载视频数，默认 5
    """
    # 移除分辨率后缀中的 'p'
    resolution = resolution.replace('p', '')
    # 统一 URL 为列表格式
    if isinstance(url, str):
        url = [url]

    # 第一步：获取所有视频的元数据信息
    ydl_opts = {
        'format': 'best',
        'dumpjson': True,             # 仅提取 JSON 信息
        'playlistend': num_videos,    # 限制播放列表长度
        'ignoreerrors': True          # 跳过错误条目
    }

    video_info_list = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            result = ydl.extract_info(u, download=False)
            if 'entries' in result:
                # 播放列表：提取所有条目
                video_info_list.extend(result['entries'])
            else:
                # 单个视频
                video_info_list.append(result)

    # 第二步：批量下载视频（已清洗标题）
    download_videos(video_info_list, folder_path, resolution)


def get_file_modification_date(file_path):
    """
    获取文件的修改日期，返回 YYYYMMDD 格式字符串。

    用于本地视频导入时，用文件修改日期代替上传日期。

    Args:
        file_path: 文件路径

    Returns:
        格式化的日期字符串，如 '20240115'
    """
    # 获取文件最后修改时间的时间戳
    timestamp = os.path.getmtime(file_path)
    # 将时间戳转换为 datetime 对象
    date = datetime.fromtimestamp(timestamp)
    # 格式化为 YYYYMMDD 字符串
    return date.strftime('%Y%m%d')


def validate_video_file(video_path):
    """
    验证视频文件的有效性。

    检查文件是否存在、是否为目录、是否为空以及文件格式。

    Args:
        video_path: 视频文件路径

    Returns:
        (is_valid: bool, message: str) 元组
    """
    # 检查文件是否存在
    if not os.path.exists(video_path):
        logger.error(f'Video file not found: {video_path}')
        return False, '文件不存在'

    # 检查路径是否为目录（预期是文件）
    if os.path.isdir(video_path):
        logger.error(f'Expected file but got directory: {video_path}')
        return False, '路径指向目录而非文件'

    # 检查文件是否为空
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        logger.error(f'Video file is empty: {video_path}')
        return False, '文件为空'

    # 检查文件是否过小（可能是损坏的文件）
    min_size = 1024  # 最小 1KB
    if file_size < min_size:
        logger.warning(f'Video file is very small ({file_size} bytes): {video_path}')

    # 检查视频格式是否常见
    video_ext = os.path.splitext(video_path)[1].lower()
    supported_formats = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v']
    if video_ext not in supported_formats:
        logger.warning(f'Uncommon video format: {video_ext}, proceeding anyway')

    return True, 'OK'


def import_local_video(video_path, folder_path, title=None, uploader=None, upload_date=None):
    """
    导入本地视频文件到统一的处理目录结构。

    将本地视频复制到标准目录 {folder_path}/{uploader}/{upload_date} {title}/download.mp4，
    并生成对应的 download.info.json 元数据文件。

    Args:
        video_path: 本地视频文件路径
        folder_path: 基础存储目录
        title: 视频标题（可选，默认为文件名不含扩展名）
        uploader: 上传者名称（可选，默认为 'Local'）
        upload_date: 上传日期 YYYYMMDD（可选，默认为文件修改日期）

    Returns:
        输出文件夹路径，失败返回 None
    """
    # 验证本地视频文件有效性
    is_valid, error_msg = validate_video_file(video_path)
    if not is_valid:
        logger.error(f'Cannot import video {video_path}: {error_msg}')
        return None

    # 如果未提供标题，使用文件名（不含扩展名）作为标题
    if title is None:
        title = os.path.splitext(os.path.basename(video_path))[0]
    title = sanitize_title(title)

    # 如果未提供上传者，默认为 'Local'
    if uploader is None:
        uploader = 'Local'
    uploader = sanitize_title(uploader)

    # 处理上传日期：未提供时使用文件修改日期
    if upload_date is None:
        upload_date = get_file_modification_date(video_path)
    else:
        # 确保日期格式正确（YYYYMMDD，8位数字）
        upload_date = str(upload_date).strip()
        if len(upload_date) != 8 or not upload_date.isdigit():
            logger.warning(f'Invalid upload date format: {upload_date}, using file modification date instead')
            upload_date = get_file_modification_date(video_path)

    # 构建输出文件夹路径
    output_folder = os.path.join(folder_path, uploader, f'{upload_date} {title}')

    # 如果目标文件已存在，跳过导入
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already imported in {output_folder}')
        return output_folder

    # 创建输出文件夹（若不存在）
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        logger.error(f'Failed to create output folder {output_folder}: {e}')
        return None

    # 复制视频文件到目标路径
    target_video_path = os.path.join(output_folder, 'download.mp4')
    logger.info(f'Copying video file to {target_video_path}')
    try:
        # shutil.copy2 保留文件元数据（如修改时间）
        shutil.copy2(video_path, target_video_path)
    except Exception as e:
        logger.error(f'Failed to copy video file: {e}')
        return None

    # 验证复制是否成功
    if not os.path.exists(target_video_path):
        logger.error(f'Video file was not copied to target: {target_video_path}')
        return None

    # 生成元数据 JSON 文件（与 yt_dlp 下载时生成的格式一致）
    info_json_path = os.path.join(output_folder, 'download.info.json')
    if not os.path.exists(info_json_path):
        # 构建视频元数据字典
        video_info = {
            'title': title,
            'uploader': uploader,
            'upload_date': upload_date,
            'webpage_url': f'local://{os.path.basename(video_path)}',
            'extractor': 'local',       # 标识为本地导入
            'format': 'local',
            'duration': 0,
            'is_local': True,           # 自定义标记：本地视频
            'original_path': video_path, # 原始路径，便于追溯
            'original_size': os.path.getsize(video_path)
        }
        try:
            with open(info_json_path, 'w', encoding='utf-8') as f:
                json.dump(video_info, f, ensure_ascii=False, indent=2)
            logger.info(f'Created metadata file: {info_json_path}')
        except Exception as e:
            logger.warning(f'Failed to create metadata file: {e}')

    logger.info(f'Video imported successfully to {output_folder}')
    return output_folder


def get_info_list_from_local(video_paths, folder_path):
    """
    从本地视频文件路径列表生成视频信息（生成器模式）。

    用于将本地文件统一为与 YouTube 下载一致的信息格式，
    供后续流水线步骤使用。

    Args:
        video_paths: 视频文件路径字符串或路径列表
        folder_path: 基础存储目录

    Yields:
        视频信息字典，包含 is_local=True 标记
    """
    # 统一为列表格式
    if isinstance(video_paths, str):
        video_paths = [video_paths]

    # 逐个处理视频路径
    for video_path in video_paths:
        # 跳过不存在的文件
        if not os.path.exists(video_path):
            logger.warning(f'Skipping non-existent file: {video_path}')
            continue

        # 从文件名推断标题
        title = os.path.splitext(os.path.basename(video_path))[0]
        title = sanitize_title(title)
        uploader = 'Local'  # 本地视频统一标记上传者为 Local
        upload_date = get_file_modification_date(video_path)

        # 先创建输出目录
        output_folder = os.path.join(folder_path, uploader, f'{upload_date} {title}')
        os.makedirs(output_folder, exist_ok=True)

        # 构造兼容的信息字典
        video_info = {
            'title': title,
            'uploader': uploader,
            'upload_date': upload_date,
            'webpage_url': f'local://{os.path.basename(video_path)}',
            'extractor': 'local',
            'local_video_path': video_path,  # 保存原始路径供后续复制
            'is_local': True
        }
        yield video_info


def process_local_video(video_info, folder_path):
    """
    根据视频信息字典处理单个本地视频的导入。

    从 video_info 中读取 local_video_path，复制到标准目录结构。

    Args:
        video_info: 视频信息字典（来自 get_info_list_from_local）
        folder_path: 基础存储目录

    Returns:
        输出文件夹路径，失败返回 None
    """
    # 获取本地视频路径
    local_path = video_info.get('local_video_path')
    if not local_path or not os.path.exists(local_path):
        logger.error(f'Local video path not valid: {local_path}')
        return None

    # 清理并获取各字段
    title = sanitize_title(video_info.get('title', 'Untitled'))
    uploader = sanitize_title(video_info.get('uploader', 'Local'))
    upload_date = video_info.get('upload_date', get_file_modification_date(local_path))

    # 构建输出目录
    output_folder = os.path.join(folder_path, uploader, f'{upload_date} {title}')

    # 跳过已导入的视频
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already imported in {output_folder}')
        return output_folder

    # 创建目录并复制文件
    os.makedirs(output_folder, exist_ok=True)

    target_video_path = os.path.join(output_folder, 'download.mp4')
    shutil.copy2(local_path, target_video_path)

    # 保存元数据
    info_json_path = os.path.join(output_folder, 'download.info.json')
    with open(info_json_path, 'w', encoding='utf-8') as f:
        json.dump(video_info, f, ensure_ascii=False, indent=2)

    logger.info(f'Local video processed: {output_folder}')
    return output_folder


def _download_video_into_dir(info, target_dir, resolution='1080p'):
    """
    直接将视频下载到指定目录，不创建上传者/日期子文件夹。

    与 download_single_video 的区别在于：不使用标准分级目录结构，
    而是直接下载到给定的 target_dir 中。用于已经存在 info.json 的场景。

    Args:
        info: 视频信息字典
        target_dir: 目标目录（视频将下载到该目录下的 download.mp4）
        resolution: 目标视频分辨率

    Raises:
        RuntimeError: FFmpeg 不可用时抛出
        FileNotFoundError: 下载后未找到文件时抛出
    """
    # 检查 FFmpeg 可用性
    ffmpeg_available, ffmpeg_msg = ensure_ffmpeg_available(auto_download=True)
    if not ffmpeg_available:
        raise RuntimeError(
            f'FFmpeg 不可用，无法进行视频下载和合并。{ffmpeg_msg}\n'
            f'请按以下方式之一安装 FFmpeg：\n'
            f'1. 运行 python scripts/download_ffmpeg.py 自动下载\n'
            f'2. Windows: 从 https://ffmpeg.org/download.html 下载，解压后将 bin 目录添加到系统 PATH，或在配置中设置 FFMPEG_PATH\n'
            f'3. macOS: brew install ffmpeg\n'
            f'4. Linux: sudo apt install ffmpeg'
        )

    # 移除分辨率后缀
    resolution_num = resolution.replace('p', '')
    # 获取 FFmpeg 路径
    ffmpeg_path = get_ffmpeg_path()
    # yt_dlp 配置：直接输出到目标目录
    ydl_opts = {
        'format': f'bestvideo[ext=mp4][height<={resolution_num}]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(target_dir, 'download'),  # 直接使用传入的目录
        'ignoreerrors': True,
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_path,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([info['webpage_url']])

    # 验证下载结果
    if not os.path.exists(os.path.join(target_dir, 'download.mp4')):
        raise FileNotFoundError(
            f'下载完成但未找到 download.mp4，请检查 ffmpeg 是否已安装且可用。'
            f'目录内容: {os.listdir(target_dir) if os.path.exists(target_dir) else "目录不存在"}'
        )

    logger.info(f'Video downloaded directly into {target_dir}')
    return target_dir


def download_all_videos_under_folder(root_folder, url=None, video_path=None, resolution="1080p", num_videos=5):
    """
    扫描文件夹，为所有存在 download.info.json 但缺少 download.mp4 的目录执行下载。

    用于断点续传或批量补下载场景。遍历指定根目录下的所有子目录，
    找到标记了 info.json 但尚未下载视频的目录进行下载。

    Args:
        root_folder: 要扫描的根文件夹路径（相对或绝对）
        url: 保留参数，未使用
        video_path: 保留参数，未使用
        resolution: 视频分辨率
        num_videos: 保留参数，未使用

    Returns:
        执行结果消息字符串
    """
    # 如果是相对路径，基于项目根目录转为绝对路径
    if not os.path.isabs(root_folder):
        root_folder = str(PROJECT_ROOT / root_folder)
    found_video_dir = False
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(root_folder):
        # 需要存在 info.json 但缺少 mp4 的目录才需要下载
        if 'download.info.json' not in files:
            continue
        if 'download.mp4' in files:
            continue
        found_video_dir = True
        # 读取已有元数据信息
        info_path = os.path.join(root, 'download.info.json')
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        # 直接下载到 info.json 所在目录（不重新创建层级结构）
        _download_video_into_dir(info, root, resolution)
    if not found_video_dir:
        logger.info(f'No videos to download under {root_folder}')
    return f'Downloaded all videos under {root_folder}'


def download_videos_in_folders(folder_list, resolution="1080p"):
    """
    处理指定目录列表中的视频下载。

    接收一个目录列表，对每个目录检查其中是否存在 download.info.json，
    若存在且缺少 download.mp4，则执行下载。

    Args:
        folder_list: 需要处理的目录路径列表（单个字符串或列表）
        resolution: 视频分辨率

    Returns:
        格式化的执行结果字符串，包含成功和失败计数
    """
    # 统一为列表格式
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    success_list = []  # 成功下载的目录列表
    fail_list = []     # 失败下载的目录列表

    for folder in folder_list:
        folder = os.path.abspath(folder)
        info_path = os.path.join(folder, 'download.info.json')

        # 检查元数据文件是否存在
        if not os.path.exists(info_path):
            fail_list.append(f"{folder}: 缺少 download.info.json")
            continue

        # 如果已存在视频文件，跳过
        if os.path.exists(os.path.join(folder, 'download.mp4')):
            logger.info(f'Video already downloaded in {folder}')
            success_list.append(folder)
            continue

        # 执行下载
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            result = download_single_video(info, os.path.dirname(folder), resolution)
            if result:
                success_list.append(folder)
            else:
                fail_list.append(f"{folder}: 下载失败")
        except Exception as e:
            logger.error(f'Error downloading video in {folder}: {e}')
            fail_list.append(f"{folder}: {e}")

    logger.info(f'Downloaded {len(success_list)}/{len(folder_list)} videos; {len(fail_list)} failed')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'


if __name__ == '__main__':
    # 模块直接运行时的示例用法
    url = 'https://www.youtube.com/watch?v=3LPJfIKxwWc'
    folder_path = 'videos'
    download_from_url(url, folder_path)
