import os
import re
import json
import shutil
from datetime import datetime
from pathlib import Path
from loguru import logger
import yt_dlp


def sanitize_title(title):
    # Only keep numbers, letters, Chinese characters, and spaces
    title = re.sub(r'[^\w\u4e00-\u9fff \d_-]', '', title)
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title)
    return title


def get_target_folder(info, folder_path):
    sanitized_title = sanitize_title(info['title'])
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        return None

    output_folder = os.path.join(
        folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}')

    return output_folder

def download_single_video(info, folder_path, resolution='1080p'):
    sanitized_title = sanitize_title(info['title'])
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        return None
    
    output_folder = os.path.join(folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}')
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already downloaded in {output_folder}')
        return output_folder
    
    resolution = resolution.replace('p', '')
    ydl_opts = {
        # 'res': '1080',
        'format': f'bestvideo[ext=mp4][height<={resolution}]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'writeinfojson': True,
        'writethumbnail': True,
        'outtmpl': os.path.join(folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}', 'download'),
        'ignoreerrors': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([info['webpage_url']])
    logger.info(f'Video downloaded in {output_folder}')
    return output_folder

def download_videos(info_list, folder_path, resolution='1080p'):
    for info in info_list:
        download_single_video(info, folder_path, resolution)

def get_info_list_from_url(url, num_videos):
    if isinstance(url, str):
        url = [url]

    # Download JSON information first
    ydl_opts = {
        'format': 'best',
        'dumpjson': True,
        'playlistend': num_videos,
        'ignoreerrors': True
    }

    # video_info_list = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            result = ydl.extract_info(u, download=False)
            if 'entries' in result:
                # Playlist
                # video_info_list.extend(result['entries'])
                for video_info in result['entries']:
                    yield video_info
            else:
                # Single video
                # video_info_list.append(result)
                yield result
    
    # return video_info_list

def download_from_url(url, folder_path, resolution='1080p', num_videos=5):
    resolution = resolution.replace('p', '')
    if isinstance(url, str):
        url = [url]

    # Download JSON information first
    ydl_opts = {
        'format': 'best',
        'dumpjson': True,
        'playlistend': num_videos,
        'ignoreerrors': True
    }

    video_info_list = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            result = ydl.extract_info(u, download=False)
            if 'entries' in result:
                # Playlist
                video_info_list.extend(result['entries'])
            else:
                # Single video
                video_info_list.append(result)

    # Now download videos with sanitized titles
    download_videos(video_info_list, folder_path, resolution)


def get_file_modification_date(file_path):
    timestamp = os.path.getmtime(file_path)
    date = datetime.fromtimestamp(timestamp)
    return date.strftime('%Y%m%d')

def validate_video_file(video_path):
    if not os.path.exists(video_path):
        logger.error(f'Video file not found: {video_path}')
        return False, '文件不存在'
    
    if os.path.isdir(video_path):
        logger.error(f'Expected file but got directory: {video_path}')
        return False, '路径指向目录而非文件'
    
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        logger.error(f'Video file is empty: {video_path}')
        return False, '文件为空'
    
    min_size = 1024
    if file_size < min_size:
        logger.warning(f'Video file is very small ({file_size} bytes): {video_path}')
    
    video_ext = os.path.splitext(video_path)[1].lower()
    supported_formats = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v']
    if video_ext not in supported_formats:
        logger.warning(f'Uncommon video format: {video_ext}, proceeding anyway')
    
    return True, 'OK'

def import_local_video(video_path, folder_path, title=None, uploader=None, upload_date=None):
    is_valid, error_msg = validate_video_file(video_path)
    if not is_valid:
        logger.error(f'Cannot import video {video_path}: {error_msg}')
        return None
    
    if title is None:
        title = os.path.splitext(os.path.basename(video_path))[0]
    title = sanitize_title(title)
    
    if uploader is None:
        uploader = 'Local'
    uploader = sanitize_title(uploader)
    
    if upload_date is None:
        upload_date = get_file_modification_date(video_path)
    else:
        upload_date = str(upload_date).strip()
        if len(upload_date) != 8 or not upload_date.isdigit():
            logger.warning(f'Invalid upload date format: {upload_date}, using file modification date instead')
            upload_date = get_file_modification_date(video_path)
    
    output_folder = os.path.join(folder_path, uploader, f'{upload_date} {title}')
    
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already imported in {output_folder}')
        return output_folder
    
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        logger.error(f'Failed to create output folder {output_folder}: {e}')
        return None
    
    target_video_path = os.path.join(output_folder, 'download.mp4')
    logger.info(f'Copying video file to {target_video_path}')
    try:
        shutil.copy2(video_path, target_video_path)
    except Exception as e:
        logger.error(f'Failed to copy video file: {e}')
        return None
    
    if not os.path.exists(target_video_path):
        logger.error(f'Video file was not copied to target: {target_video_path}')
        return None
    
    info_json_path = os.path.join(output_folder, 'download.info.json')
    if not os.path.exists(info_json_path):
        video_info = {
            'title': title,
            'uploader': uploader,
            'upload_date': upload_date,
            'webpage_url': f'local://{os.path.basename(video_path)}',
            'extractor': 'local',
            'format': 'local',
            'duration': 0,
            'is_local': True,
            'original_path': video_path,
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
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    
    for video_path in video_paths:
        if not os.path.exists(video_path):
            logger.warning(f'Skipping non-existent file: {video_path}')
            continue
        
        title = os.path.splitext(os.path.basename(video_path))[0]
        title = sanitize_title(title)
        uploader = 'Local'
        upload_date = get_file_modification_date(video_path)
        
        output_folder = os.path.join(folder_path, uploader, f'{upload_date} {title}')
        os.makedirs(output_folder, exist_ok=True)
        
        video_info = {
            'title': title,
            'uploader': uploader,
            'upload_date': upload_date,
            'webpage_url': f'local://{os.path.basename(video_path)}',
            'extractor': 'local',
            'local_video_path': video_path,
            'is_local': True
        }
        yield video_info

def process_local_video(video_info, folder_path):
    local_path = video_info.get('local_video_path')
    if not local_path or not os.path.exists(local_path):
        logger.error(f'Local video path not valid: {local_path}')
        return None
    
    title = sanitize_title(video_info.get('title', 'Untitled'))
    uploader = sanitize_title(video_info.get('uploader', 'Local'))
    upload_date = video_info.get('upload_date', get_file_modification_date(local_path))
    
    output_folder = os.path.join(folder_path, uploader, f'{upload_date} {title}')
    
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already imported in {output_folder}')
        return output_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    target_video_path = os.path.join(output_folder, 'download.mp4')
    shutil.copy2(local_path, target_video_path)
    
    info_json_path = os.path.join(output_folder, 'download.info.json')
    with open(info_json_path, 'w', encoding='utf-8') as f:
        json.dump(video_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f'Local video processed: {output_folder}')
    return output_folder


if __name__ == '__main__':
    # Example usage
    url = 'https://www.youtube.com/watch?v=3LPJfIKxwWc'
    folder_path = 'videos'
    download_from_url(url, folder_path)
