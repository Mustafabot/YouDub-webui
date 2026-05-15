# -*- coding: utf-8 -*-
"""
step070_upload_bilibili.py — 上传处理结果到 Bilibili（步骤7，流水线最终步骤）

所属流水线位置：
  步骤7/7：B 站上传模块（最终步骤）
  上游依赖：步骤5（视频合成）生成的 video.mp4、步骤6（信息生成）生成的 video.png/video.txt
  下游产出：B 站投稿记录文件 bilibili.json（记录上传结果）

功能说明：
  1. 使用 bilibili_toolman 库的 BiliSession 实现登录和投稿
  2. 自动检测已上传的视频（通过 bilibili.json 缓存），避免重复上传
  3. 构建 B 站投稿信息：标题（中配前缀）、描述（含原视频链接和项目地址）、标签、分区
  4. 支持最多 12 个标签（每个标签不超过 20 字符）
  5. 投稿失败时自动重试最多 5 次，每次间隔 10 秒

输入文件：
  - {folder}/video.mp4               : 合成完成的视频
  - {folder}/video.png               : 标准化封面图片
  - {folder}/summary.json            : LLM 生成的摘要（title/author/summary/tags）
  - {folder}/download.info.json      : 原始视频元数据（原标题和原视频链接）

输出文件：
  - {folder}/bilibili.json           : B 站 API 返回的投稿结果 JSON

配置文件依赖：
  - BILI_SESSDATA     : B 站登录会话数据（Cookie 的一部分）
  - BILI_BILI_JCT     : B 站 CSRF Token（bili_jct Cookie）
"""

import subprocess    # 子进程管理（当前模块未使用，保留以备扩展）
import time          # 延时控制（上传重试间隔）
import json          # JSON 读写（bilibili.json、summary.json、download.info.json）
import os            # 文件和路径操作
from bilibili_toolman.bilisession.web import BiliSession       # B 站 Web 会话管理
from bilibili_toolman.bilisession.common.submission import Submission  # B 站投稿数据模型
from loguru import logger  # 结构化日志

from .config import get_config, PROJECT_ROOT  # 配置管理


def bili_login():
    """
    使用 SESSDATA 和 bili_jct 登录 B 站并返回会话对象。

    需要预先在配置中设置以下项：
      - BILI_SESSDATA: 浏览器 Cookie 中的 SESSDATA 值
      - BILI_BILI_JCT: 浏览器 Cookie 中的 bili_jct 值

    Returns:
        BiliSession: 已登录的 B 站会话实例

    Raises:
        Exception: 登录失败时抛出（通常是因为 Cookie 过期或无效）
    """
    # 从配置中读取 B 站登录凭证
    SESSDATA = get_config('BILI_SESSDATA')
    BILI_JCT = get_config('BILI_BILI_JCT')

    try:
        # 使用 Cookie 字符串初始化 BiliSession
        session = BiliSession(f'SESSDATA={SESSDATA};bili_jct={BILI_JCT}')
        logger.info(f"bilibili登陆成功。")
        return session
    except Exception as e:
        logger.error(e)
        raise Exception('bilibili登陆失败，请更换SESSDATA和bili_jct。')


def upload_video(folder):
    """
    将指定文件夹中的合成视频上传到 B 站。

    处理流程：
      1. 检查 bilibili.json 缓存，确认是否已上传
      2. 验证输入文件（video.mp4、video.png、summary.json、download.info.json）存在
      3. 从 summary.json 读取标题、作者、摘要、标签信息
      4. 从 download.info.json 读取原视频标题和 URL
      5. 构建 B 站投稿信息并上传
      6. 上传成功后将结果缓存到 bilibili.json

    Args:
        folder (str): 视频处理目录路径

    Returns:
        bool: 上传成功返回 True

    Raises:
        FileNotFoundError: 缺少必要的输入文件
        Exception: 上传失败（重试 5 次后仍失败）
    """
    # ---- 检查是否已经上传成功 ----
    submission_result_path = os.path.join(folder, 'bilibili.json')
    if os.path.exists(submission_result_path):
        with open(submission_result_path, 'r', encoding='utf-8') as f:
            submission_result = json.load(f)
        # B 站 API 返回码 code=0 表示成功
        if submission_result['results'][0]['code'] == 0:
            logger.info('Video already uploaded.')
            return True

    # ---- 输入文件路径定义 ----
    video_path = os.path.join(folder, 'video.mp4')         # 合成视频（来自步骤5）
    cover_path = os.path.join(folder, 'video.png')         # 封面图片（来自步骤6）

    # ---- 输入文件存在性检查 ----
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f'合成视频不存在: {video_path}，请确认视频合成步骤已正确执行'
        )
    if not os.path.exists(cover_path):
        raise FileNotFoundError(
            f'封面图片不存在: {cover_path}，请确认信息生成步骤已正确执行'
        )
    if not os.path.exists(os.path.join(folder, 'summary.json')):
        raise FileNotFoundError(
            f'摘要文件不存在: {os.path.join(folder, "summary.json")}，请确认翻译步骤已正确执行'
        )
    if not os.path.exists(os.path.join(folder, 'download.info.json')):
        raise FileNotFoundError(
            f'元数据文件不存在: {os.path.join(folder, "download.info.json")}，请确认下载步骤已正确执行'
        )

    # ---- 读取摘要信息（标题、摘要、标签） ----
    with open(os.path.join(folder, 'summary.json'), 'r', encoding='utf-8') as f:
        summary = json.load(f)
    # 清理标题中的预设前缀（如果 LLM 添加了"视频标题："字样）
    summary['title'] = summary['title'].replace('视频标题：', '').strip()
    # 清理摘要中的预设前缀
    summary['summary'] = summary['summary'].replace(
        '视频摘要：', '').replace('视频简介：', '').strip()
    # 获取标签列表
    tags = summary.get('tags', [])
    if not isinstance(tags, list):
        tags = []

    # ---- 构建 B 站视频标题 ----
    # 添加【中配】前缀以标识这是中文配音版
    title = f'【中配】{summary["title"]} - {summary["author"]}'

    # ---- 读取原视频元数据 ----
    with open(os.path.join(folder, 'download.info.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    title_English = data['title']           # 原视频英文标题
    webpage_url = data['webpage_url']       # 原视频链接

    # ---- 构建 B 站视频描述 ----
    description = (
        f'{title_English}\n'                         # 原视频标题
        f'{summary["summary"]}\n\n'                   # 中文摘要
        f'项目地址：https://github.com/liuzhao1225/YouDub-webui\n'
        f'YouDub 是一个开创性的开源工具，旨在将 YouTube 和其他平台上的高质量视频'
        f'翻译和配音成中文版本。该工具结合了最新的 AI 技术，包括语音识别、'
        f'大型语言模型翻译，以及 AI 声音克隆技术，提供与原视频相似的中文配音，'
        f'为中文用户提供卓越的观看体验。'
    )

    # ---- 登录 B 站 ----
    session = bili_login()
    # time.sleep(5)  # 注释掉的登录后等待

    # ---- 上传和投稿（最多重试 5 次） ----
    for retry in range(5):
        try:
            # 上传视频文件，获取视频 endpoint
            video_endpoint, _ = session.UploadVideo(video_path)

            # 创建主投稿对象
            submission = Submission(
                title=title,
                desc=description
            )

            # 添加视频分 P（此处只有 1P）
            submission.videos.append(
                Submission(
                    title=title,
                    video_endpoint=video_endpoint
                )
            )

            # 上传封面图片
            submission.cover_url = session.UploadCover(cover_path)

            # ---- 设置标签 ----
            # 合并默认标签和摘要中的标签，最多保留 12 个
            tags = ['YouDub', summary["author"], 'AI',
                    'ChatGPT'] + tags + ['中文配音', '科学', '科普']
            for tag in tags[:12]:
                # B 站标签长度限制为 20 字符
                if len(tag) > 20:
                    tag = tag[:20]
                submission.tags.append(tag)

            # 设置分区（201 = 科普）
            submission.thread = 201
            # 声明为转载（非原创），标注原始来源
            submission.copyright = submission.COPYRIGHT_REUPLOAD
            submission.source = webpage_url

            # 提交投稿
            response = session.SubmitSubmission(submission, seperate_parts=False)

            # 检查投稿结果
            if response['results'][0]['code'] != 0:
                logger.error(response)
                raise Exception(response)

            logger.info(f"Submission successful: {response}")

            # 将投稿结果缓存到 bilibili.json
            with open(os.path.join(folder, 'bilibili.json'), 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=4)

            return True

        except Exception as e:
            logger.error(f"Error submitting:\n{e}")
            # 等待 10 秒后重试
            time.sleep(10)

    # 重试 5 次仍失败
    raise Exception('上传失败')


def upload_all_videos_under_folder(folder):
    """
    扫描指定目录下所有视频处理目录，逐个上传到 B 站。

    通过 os.walk 递归查找包含 video.mp4 的子目录。

    Args:
        folder (str): 根目录路径（相对或绝对路径）

    Returns:
        str: 提示信息，表示所有上传操作完成

    Raises:
        FileNotFoundError: 未找到任何包含 video.mp4 的视频目录
    """
    # 如果是相对路径，基于项目根目录拼接
    if not os.path.isabs(folder):
        folder = str(PROJECT_ROOT / folder)
    found_video_dir = False

    # 递归遍历所有子目录
    for dir, _, files in os.walk(folder):
        # 只处理包含 video.mp4 的目录
        if 'video.mp4' not in files and 'bilibili.json' not in files:
            continue
        found_video_dir = True
        # 有 bilibili.json 但无 video.mp4 说明步骤顺序异常
        if 'video.mp4' not in files:
            raise FileNotFoundError(
                f'发现视频目录 {dir} 但缺少 video.mp4，'
                f'请确认视频合成步骤已正确执行。目录内容: {files}'
            )
        # 执行上传
        upload_video(dir)

    if not found_video_dir:
        raise FileNotFoundError(f'在 {folder} 下未找到任何视频处理目录')
    return f'All videos under {folder} uploaded.'


def upload_videos_in_folders(folder_list):
    """处理指定目录列表中的视频上传（批量入口）

    支持传入单个目录路径字符串或多个路径的列表。逐一调用 upload_video，
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
        if 'video.mp4' not in files:
            fail_list.append(f"{subdir}: 缺少 video.mp4")
            continue
        # 检查是否已经上传成功
        if os.path.exists(os.path.join(subdir, 'bilibili.json')):
            with open(os.path.join(subdir, 'bilibili.json'), 'r', encoding='utf-8') as f:
                bilibili_info = json.load(f)
            if bilibili_info['results'][0]['code'] == 0:
                logger.info(f'Video already uploaded: {subdir}')
                success_list.append(subdir)
                continue
        try:
            upload_video(subdir)
            success_list.append(subdir)
        except Exception as e:
            logger.error(f'Error uploading video in {subdir}: {e}')
            fail_list.append(f"{subdir}: {e}")
    logger.info(f'上传完成: 成功 {len(success_list)}/{len(folder_list)}, 失败 {len(fail_list)}')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'


if __name__ == '__main__':
    # 命令行直接运行时的测试入口
    # 示例路径（注释状态）：
    # folder = r'F:\YouDub-webui\videos\DigiDigger\20200824 How do non-euclidean games work Bitwise'
    folder = r'videos\The Game Theorists\20210522 Game Theory What Level is Ashs Pikachu Pokemon'
    upload_all_videos_under_folder(folder)
