#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除空目录的工具脚本

功能：
  遍历 'videos' 目录下的所有子目录，如果子目录中包含 download.mp4 文件，
  则对该子目录的名称进行清理（移除特殊字符），并将清理后的目录重命名为新名称。
  此脚本主要用于修复视频下载后因标题包含特殊字符而导致的目录名问题。

使用方式：
  python remove_empty.py
  （直接运行，无需参数）
"""

import os  # 操作系统接口，用于文件路径操作、目录遍历和重命名
import re  # 正则表达式库，用于清理目录名称中的特殊字符


def sanitize_title(title):
    """
    清理标题字符串，移除非法字符并规范化空白

    参数:
        title (str): 原始标题字符串

    返回:
        str: 清理后的标题字符串

    清理规则：
        1. 移除所有非中文字符、非单词字符（字母、数字、下划线）、非空格和短横线的字符
        2. 将连续的空白字符压缩为单个空格
    """
    # 匹配并保留：中文字符、单词字符（\w 包括字母数字下划线）、空格、短横线和下划线
    # 移除所有不在上述集合中的字符
    title = re.sub(r'[^\w一-鿿 \d_-]', '', title)
    # 将连续的空白字符（空格、制表符、换行等）替换为单个空格
    title = re.sub(r'\s+', ' ', title)
    return title


# 遍历 'videos' 目录及其所有子目录
# os.walk 返回三元组：(当前目录路径, 子目录列表, 文件列表)
for root, dirs, files in os.walk('videos'):
    # 检查当前目录中是否包含 download.mp4 文件
    # download.mp4 是视频下载后的标准文件名，表示该目录包含有效视频
    if 'download.mp4' in files:
        # 获取父目录路径（root 的上一级目录）
        par_folder = os.path.dirname(root)
        # 获取当前目录的原始名称（即视频标题）
        old_title = os.path.basename(root)
        # 对目录名称进行清理，移除特殊字符
        new_title = sanitize_title(old_title)
        # 仅在清理后的名称与原名称不同时执行重命名
        if old_title != new_title:
            # 打印原始名称和清理后的名称，方便用户查看修改内容
            print(old_title)
            print(new_title)
            print()
            # 执行目录重命名操作
            # 将 old_title 替换为 new_title，保持父目录路径不变
            os.rename(os.path.join(par_folder, old_title),
                      os.path.join(par_folder, new_title))
