import json
import os
from PIL import Image

from .config import PROJECT_ROOT


def resize_thumbnail(folder, size=(1280, 960)):
    image_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_path = None
    for suffix in image_suffix:
        candidate_path = os.path.join(folder, f'download{suffix}')
        if os.path.exists(candidate_path):
            image_path = candidate_path
            break
    if image_path is None:
        raise FileNotFoundError(f'在 {folder} 中未找到任何图片文件（支持格式：{", ".join(image_suffix)}）')
    with Image.open(image_path) as img:
        img_ratio = img.width / img.height
        target_ratio = size[0] / size[1]

        if img_ratio < target_ratio:
            new_height = size[1]
            new_width = int(new_height * img_ratio)
        else:
            new_width = size[0]
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_img = Image.new('RGB', size, "black")

        x_offset = (size[0] - new_width) // 2
        y_offset = (size[1] - new_height) // 2
        new_img.paste(img, (x_offset, y_offset))

        new_img_path = os.path.join(folder, 'video.png')
        new_img.save(new_img_path)
        return new_img_path

def generate_summary_txt(folder):
    summary_path = os.path.join(folder, 'summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f'摘要文件不存在: {summary_path}，请确认翻译步骤已正确执行')
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    title = f'{summary["title"]} - {summary["author"]}'
    summary = summary['summary']
    txt = f'{title}\n\n{summary}'
    with open(os.path.join(folder, 'video.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)

def generate_info(folder):
    generate_summary_txt(folder)
    resize_thumbnail(folder)
    
def generate_all_info_under_folder(root_folder):
    if not os.path.isabs(root_folder):
        root_folder = str(PROJECT_ROOT / root_folder)
    found_video_dir = False
    for root, dirs, files in os.walk(root_folder):
        if 'download.info.json' not in files and 'video.txt' not in files:
            continue
        found_video_dir = True
        if 'download.info.json' not in files:
            raise FileNotFoundError(
                f'发现视频目录 {root} 但缺少 download.info.json，请确认下载步骤已正确执行。目录内容: {files}'
            )
        if 'video.txt' in files and 'video.png' in files:
            continue
        generate_info(root)
    if not found_video_dir:
        raise FileNotFoundError(f'在 {root_folder} 下未找到任何视频处理目录')
    return f'Generated all info under {root_folder}'
def generate_info_in_folders(folder_list):
    """处理指定目录列表中的信息生成

    Args:
        folder_list: 需要处理的目录路径列表
    """
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    success_list = []
    fail_list = []
    for subdir in folder_list:
        subdir = os.path.abspath(subdir)
        files = os.listdir(subdir) if os.path.exists(subdir) else []
        if 'download.info.json' not in files:
            fail_list.append(f"{subdir}: 缺少 download.info.json")
            continue
        if 'video.txt' in files and 'video.png' in files:
            logger.info(f'信息已生成，跳过: {subdir}')
            success_list.append(subdir)
            continue
        try:
            generate_info(subdir)
            success_list.append(subdir)
        except Exception as e:
            logger.error(f'Error generating info in {subdir}: {e}')
            fail_list.append(f"{subdir}: {e}")
    logger.info(f'信息生成完成: 成功 {len(success_list)}/{len(folder_list)}, 失败 {len(fail_list)}')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'


if __name__ == '__main__':
    generate_all_info_under_folder('videos')
