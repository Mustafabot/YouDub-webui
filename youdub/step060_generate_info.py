# -*- coding: utf-8 -*-
"""
step060_generate_info.py — 生成视频信息（步骤6）

所属流水线位置：
  步骤6/7：视频信息生成模块
  上游依赖：步骤2（WhisperX 语音识别）生成的 summary.json、步骤1（下载）提供的 download.jpg/png
  下游产出：video.txt（标题+摘要文本）、video.png（标准化封面图）
  下游消费：step070 上传 B 站时作为封面和描述

功能说明：
  1. resize_thumbnail：将视频封面图统一缩放到指定尺寸（默认 1280x960），
     保持原始宽高比并用黑边填充空白区域（letterbox 模式）
  2. generate_summary_txt：从 summary.json 中提取标题、作者、摘要内容，
     生成简洁的文本文件用于 B 站描述
  3. generate_info：组合上述两个功能，一步生成所有信息文件

输入文件：
  - {folder}/summary.json          : LLM 生成的摘要（含 title/author/summary/tags）
  - {folder}/download.{jpg|png|...}: 下载的视频封面原图

输出文件：
  - {folder}/video.txt             : 包含标题和摘要的纯文本文件
  - {folder}/video.png             : 标准化尺寸（1280x960）的封面图片
"""

import json       # 读写 summary.json 等 JSON 文件
import os         # 文件和路径操作
from PIL import Image         # Python Imaging Library，用于图片缩放、裁剪、合成

from .config import PROJECT_ROOT  # 项目根路径


def resize_thumbnail(folder, size=(1280, 960)):
    """
    将视频封面图缩放并填充至目标尺寸。

    处理策略（letterbox 模式）：
      1. 保持原始图片的宽高比进行等比缩放（长边适配目标尺寸）
      2. 在黑色背景上居中放置缩放后的图片
      3. 使用 LANCZOS 重采样算法保证缩放质量

    支持的输入格式：jpg、jpeg、png、bmp、webp

    Args:
        folder (str):    视频处理目录路径
        size (tuple):    目标尺寸 (width, height)，默认 (1280, 960)

    Returns:
        str: 生成的封面图文件路径（{folder}/video.png）

    Raises:
        FileNotFoundError: 在指定目录中未找到任何支持的图片文件
    """
    # 支持的图片格式后缀列表（按常见度排序）
    image_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_path = None

    # 遍历查找存在的封面图片（优先使用 download.jpg 或 download.png）
    for suffix in image_suffix:
        candidate_path = os.path.join(folder, f'download{suffix}')
        if os.path.exists(candidate_path):
            image_path = candidate_path
            break

    # 未找到任何图片文件时抛错
    if image_path is None:
        raise FileNotFoundError(
            f'在 {folder} 中未找到任何图片文件（支持格式：{", ".join(image_suffix)}）'
        )

    # 打开图片并计算宽高比
    with Image.open(image_path) as img:
        img_ratio = img.width / img.height          # 原始宽高比
        target_ratio = size[0] / size[1]            # 目标宽高比

        # 根据宽高比决定适配策略（等比缩放，保持图片完整）
        if img_ratio < target_ratio:
            # 原图更"瘦高"：以目标高度为基准缩放
            new_height = size[1]
            new_width = int(new_height * img_ratio)
        else:
            # 原图更"宽扁"：以目标宽度为基准缩放
            new_width = size[0]
            new_height = int(new_width / img_ratio)

        # 使用 LANCZOS 重采样算法进行高质量缩放
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 创建黑色背景画布
        new_img = Image.new('RGB', size, "black")

        # 计算居中位置偏移量
        x_offset = (size[0] - new_width) // 2
        y_offset = (size[1] - new_height) // 2

        # 将缩放后的图片粘贴到黑色背景的居中位置
        new_img.paste(img, (x_offset, y_offset))

        # 保存标准化后的封面图
        new_img_path = os.path.join(folder, 'video.png')
        new_img.save(new_img_path)
        return new_img_path


def generate_summary_txt(folder):
    """
    从 summary.json 生成摘要文本文件 (video.txt)。

    从 summary.json 中读取 LLM 生成的标题、作者信息和内容摘要，
    组合成格式化的纯文本文件，供 B 站上传使用。

    文件格式：
      第一行：标题 - 作者
      空行
      后续行：摘要内容

    Args:
        folder (str): 视频处理目录路径

    Returns:
        None（结果写入 {folder}/video.txt）

    Raises:
        FileNotFoundError: summary.json 不存在时抛出
    """
    # summary.json 路径（由步骤2 的 LLM 翻译生成）
    summary_path = os.path.join(folder, 'summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f'摘要文件不存在: {summary_path}，请确认翻译步骤已正确执行'
        )

    # 读取摘要数据
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # 拼接标题行
    title = f'{summary["title"]} - {summary["author"]}'
    # 提取摘要内容
    summary_text = summary['summary']

    # 写入视频信息文本文件
    txt = f'{title}\n\n{summary_text}'
    with open(os.path.join(folder, 'video.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)


def generate_info(folder):
    """
    为指定文件夹生成所有视频信息（摘要文本 + 封面图片）。

    组合调用 generate_summary_txt 和 resize_thumbnail，
    一步完成 B 站上传所需的文本和图片信息。

    Args:
        folder (str): 视频处理目录路径

    Returns:
        None（结果写入 {folder}/video.txt 和 {folder}/video.png）
    """
    generate_summary_txt(folder)   # 生成摘要文本
    resize_thumbnail(folder)       # 生成标准化封面


def generate_all_info_under_folder(root_folder):
    """
    扫描指定目录下所有视频处理目录，逐个生成视频信息。

    通过 os.walk 递归查找包含 download.info.json 的子目录，
    跳过已生成（video.txt + video.png 同时存在）的目录。

    Args:
        root_folder (str): 根目录路径（相对或绝对路径）

    Returns:
        str: 提示信息，表示所有目录信息生成完成

    Raises:
        FileNotFoundError: 未找到任何视频处理目录，或目录缺少 download.info.json
    """
    # 如果是相对路径，基于项目根目录拼接
    if not os.path.isabs(root_folder):
        root_folder = str(PROJECT_ROOT / root_folder)
    found_video_dir = False

    # 递归遍历所有子目录
    for root, dirs, files in os.walk(root_folder):
        # 只处理包含 download.info.json 的目录
        if 'download.info.json' not in files and 'video.txt' not in files:
            continue
        found_video_dir = True
        # 有 video.txt 但无 download.info.json 说明步骤顺序异常
        if 'download.info.json' not in files:
            raise FileNotFoundError(
                f'发现视频目录 {root} 但缺少 download.info.json，'
                f'请确认下载步骤已正确执行。目录内容: {files}'
            )
        # 跳过已生成的目录（同时存在 video.txt 和 video.png）
        if 'video.txt' in files and 'video.png' in files:
            continue
        # 执行信息生成
        generate_info(root)

    if not found_video_dir:
        raise FileNotFoundError(f'在 {root_folder} 下未找到任何视频处理目录')
    return f'Generated all info under {root_folder}'


def generate_info_in_folders(folder_list):
    """处理指定目录列表中的信息生成（批量入口）

    支持传入单个目录路径字符串或多个路径的列表。逐一调用 generate_info，
    分别统计成功和失败的个数。

    Args:
        folder_list (str|list): 需要处理的目录路径列表（或单个路径字符串）

    Returns:
        str: 包含成功/失败数量的提示信息
    """
    # 如果传入的是单个字符串，包装为列表
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
    # 命令行直接运行时的测试入口：处理整个 videos 目录
    generate_all_info_under_folder('videos')
