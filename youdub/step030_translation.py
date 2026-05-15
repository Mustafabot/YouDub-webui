# -*- coding: utf-8 -*-
"""
step030_translation.py — 步骤3：LLM 字幕翻译模块

本模块是 YouDub 流水线的第四个步骤（流水线编号 03），负责：
1. 读取步骤2输出的 transcript.json（带时间戳和说话人的转录结果）
2. 使用 LLM（OpenAI 兼容 API）生成视频摘要和标题翻译
3. 使用 LLM 逐行翻译字幕文本，保持时间戳和说话人信息
4. 对翻译结果进行后处理和有效性验证
5. 按标点符号将翻译后的长句拆分为适合 TTS 的短句

输入：
    {folder}/download.info.json       # 步骤0生成的视频元数据
    {folder}/transcript.json          # 步骤2生成的转录结果

输出：
    {folder}/summary.json             # 视频摘要（标题、简介、标签，已翻译为目标语言）
    {folder}/translation.json         # 逐行翻译结果（含拆分后的短句和时间戳）

翻译结果格式：
    [{"start": float, "end": float, "text": str, "speaker": str, "translation": str}, ...]

输出供步骤04（TTS 语音合成）使用。
"""

import json          # JSON 读写，用于加载和保存转录/翻译/摘要数据
import os            # 文件和路径操作
import re            # 正则表达式，用于句子拆分和翻译结果解析
from collections import deque  # 双端队列，用于维护翻译历史上下文
from openai import OpenAI      # OpenAI API 客户端（兼容任何 OpenAI 格式 API）
import time          # 延迟等待，用于 API 重试间的退避
from loguru import logger      # 结构化日志记录

from .config import get_config, PROJECT_ROOT  # 配置管理和项目根目录


def get_model_name():
    """
    获取 LLM 模型名称配置。

    从配置文件中读取 MODEL_NAME，默认使用 'gpt-3.5-turbo'。

    Returns:
        模型名称字符串
    """
    return get_config('MODEL_NAME', 'gpt-3.5-turbo')


# 模块加载时打印当前使用的模型名称，便于调试
logger.info(f'using model {get_model_name()}')


# ============================================================
# 默认提示词模板
# 用户可通过配置文件自定义这些提示词
# ============================================================

# 视频摘要生成的系统提示词：要求以 JSON 格式输出标题和摘要
DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    'You are a expert in the field of this video. '
    'Please detailedly summarize the video in JSON format.\n'
    '```json\n'
    '{{"title": "the title of the video", "summary", "the summary of the video"}}\n'
    '```'
)

# 摘要翻译提示词：将视频标题、摘要和标签翻译为目标语言
DEFAULT_SUMMARY_TRANSLATION_PROMPT = (
    'You are a native speaker of {target_language}. '
    'Please translate the title and summary into {target_language} in JSON format. '
    '```json\n'
    '{{"title": "the {target_language} title of the video", '
    '"summary", "the {target_language} summary of the video", '
    '"tags": [list of tags in {target_language}]}}\n'
    '```.'
)

# 行级翻译的系统提示词：要求自然、流畅、地道的翻译
DEFAULT_TRANSLATION_SYSTEM_PROMPT = (
    'You are a expert in the field of this video.\n'
    '{info}\n'
    'Translate the sentence into {target_language}.'
    '下面我让你来充当翻译家，你的目标是把任何语言翻译成中文，'
    '请翻译时不要带翻译腔，而是要翻译得自然、流畅和地道，使用优美和高雅的表达方式。'
    '请将人工智能的"agent"翻译为"智能体"，强化学习中是`Q-Learning`而不是`Queue Learning`。'
    '数学公式写成plain text，不要使用latex。'
    '确保翻译正确和简洁。注意信达雅。'
)

# 默认的 few-shot 示例：给出翻译输入/输出的格式示范
DEFAULT_TRANSLATION_FEWSHOT = [
    {'role': 'user', 'content': '使用地道的中文Translate:"Knowledge is power."'},
    {'role': 'assistant', 'content': '翻译："知识就是力量。"'},
    {'role': 'user', 'content': '使用地道的中文Translate:"To be or not to be, that is the question."'},
    {'role': 'assistant', 'content': '翻译："生存还是毁灭，这是一个值得考虑的问题。"'},
]


def get_api_params():
    """
    从配置中读取 OpenAI API 调用参数。

    支持 temperature、top_p、max_tokens 三个可选参数，
    只在配置中有有效值时才会添加。

    Returns:
        包含 API 调用参数的字典
    """
    params = {}

    # 读取 temperature 参数（控制输出随机性，0-2）
    temperature = get_config('OPENAI_API_TEMPERATURE', 1.0)
    if temperature:
        try:
            params['temperature'] = float(temperature)
        except (ValueError, TypeError):
            pass

    # 读取 top_p 参数（核采样，0-1）
    top_p = get_config('OPENAI_API_TOP_P', 1.0)
    if top_p:
        try:
            params['top_p'] = float(top_p)
        except (ValueError, TypeError):
            pass

    # 读取 max_tokens 参数（最大生成长度）
    max_tokens = get_config('OPENAI_API_MAX_TOKENS', '')
    if max_tokens and str(max_tokens).strip():
        try:
            params['max_tokens'] = int(max_tokens)
        except (ValueError, TypeError):
            pass

    return params


def build_extra_body():
    """
    构建 API 调用中的 extra_body 参数。

    用于传递特定模型（如 Yi-34B）所需的额外参数，
    同时支持用户自定义 extra_body（从 JSON 配置读取）。

    Returns:
        extra_body 字典
    """
    extra_body = {
        'repetition_penalty': 1.1,  # 默认重复惩罚系数
    }

    # 特定模型需要额外的停止 token
    if get_model_name() == "01ai/Yi-34B-Chat-4bits":
        extra_body['stop_token_ids'] = [7]

    # 用户自定义 extra_body（从配置 JSON 字符串解析）
    custom_extra = get_config('OPENAI_API_EXTRA_BODY', '')
    if custom_extra and str(custom_extra).strip():
        try:
            import json
            custom_dict = json.loads(custom_extra)
            if isinstance(custom_dict, dict):
                extra_body.update(custom_dict)
                logger.info(f'使用自定义 extra_body: {custom_dict}')
        except json.JSONDecodeError as e:
            logger.warning(f'自定义 extra_body JSON 解析失败: {e}，使用默认配置')
        except Exception as e:
            logger.warning(f'处理自定义 extra_body 时出错: {e}')

    return extra_body


def get_extra_body():
    """
    获取 extra_body 参数的封装函数（供外部调用）。

    Returns:
        extra_body 字典
    """
    return build_extra_body()


def get_custom_prompt(key, default_value):
    """
    从配置中读取自定义提示词，如果未配置则返回默认值。

    Args:
        key: 配置键名
        default_value: 默认提示词

    Returns:
        提示词字符串
    """
    value = get_config(key, '')
    if value and str(value).strip():
        return value
    return default_value


def get_custom_fewshot():
    """
    从配置中读取自定义 few-shot 示例。

    如果用户配置了 TRANSLATION_FEWSHOT_EXAMPLES（JSON 格式），
    则使用用户自定义的示例；否则使用默认示例。

    Returns:
        few-shot 示例列表
    """
    value = get_config('TRANSLATION_FEWSHOT_EXAMPLES', '')
    if value and str(value).strip():
        try:
            examples = json.loads(value)
            if isinstance(examples, list) and len(examples) > 0:
                # 验证每个示例的格式
                for ex in examples:
                    if 'role' not in ex or 'content' not in ex:
                        logger.warning(f'自定义 few-shot 示例格式无效，使用默认')
                        return DEFAULT_TRANSLATION_FEWSHOT
                return examples
        except json.JSONDecodeError as e:
            logger.warning(f'自定义 few-shot 示例 JSON 解析失败: {e}，使用默认')
    return DEFAULT_TRANSLATION_FEWSHOT


def get_necessary_info(info: dict):
    """
    从完整的视频元数据中提取翻译所需的关键信息。

    Args:
        info: 完整的视频信息字典（来自 download.info.json）

    Returns:
        包含标题、上传者、描述、日期、分类、标签的字典
    """
    return {
        'title': info['title'],
        'uploader': info['uploader'],
        'description': info['description'],
        'upload_date': info['upload_date'],
        'categories': info['categories'],
        'tags': info['tags'],
    }


def ensure_transcript_length(transcript, max_length=4000):
    """
    确保转录文本长度不超过 LLM 的上下文窗口限制。

    如果转录文本过长，从中间截断，保留开头和结尾部分，
    因为通常开头和结尾包含最有价值的信息。

    Args:
        transcript: 转录文本列表（每项包含 'text' 字段）
        max_length: 最大字符数

    Returns:
        截断后的转录文本列表
    """
    mid = len(transcript) // 2              # 中间位置
    before, after = transcript[:mid], transcript[mid:]  # 分为前半和后半
    length = max_length // 2                # 每半最大长度
    return before[:length] + after[-length:]  # 保留开头和结尾


def summarize(info, transcript, target_language='简体中文'):
    """
    使用 LLM 生成视频摘要，然后将摘要翻译为目标语言。

    两步流程：
    1. 摘要生成：LLM 根据视频元数据+转录文本生成英文摘要
    2. 摘要翻译：将生成的摘要翻译为中文（或其他目标语言）

    每步都有多次重试机制，确保成功。

    Args:
        info: 视频元数据字典
        transcript: 转录结果列表
        target_language: 目标语言，默认 '简体中文'

    Returns:
        包含翻译后标题、作者、摘要、标签、语言的字典

    Raises:
        RuntimeError: 摘要生成或翻译在多次重试后仍然失败
    """
    # 初始化 OpenAI 客户端（支持自定义 base_url 用于兼容 API）
    client = OpenAI(
        base_url=get_config('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        api_key=get_config('OPENAI_API_KEY')
    )

    # 拼接转录文本为单个字符串
    transcript = ' '.join(line['text'] for line in transcript)
    # 如果转录文本过长，截断以节省 token
    transcript = ensure_transcript_length(transcript, max_length=2000)

    # 构建视频信息前缀
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '

    # 构建完整的摘要请求内容
    full_description = (
        f'The following is the full content of the video:\n'
        f'{info_message}\n{transcript}\n{info_message}\n'
        f'According to the above content, '
        f'detailedly Summarize the video in JSON format:\n'
        f'```json\n{{"title": "", "summary": ""}}\n```'
    )

    # ========== 第一步：生成摘要 ==========
    summary_prompt = get_custom_prompt('SUMMARY_SYSTEM_PROMPT', DEFAULT_SUMMARY_SYSTEM_PROMPT)
    messages = [
        {'role': 'system', 'content': summary_prompt},
        {'role': 'user', 'content': full_description},
    ]
    retry_message = ''
    success = False
    # 摘要生成最多重试 5 次
    for retry in range(5):
        try:
            messages = [
                {'role': 'system', 'content': summary_prompt},
                {'role': 'user', 'content': full_description + retry_message},
            ]
            api_params = get_api_params()
            response = client.chat.completions.create(
                model=get_model_name(),
                messages=messages,
                timeout=240,            # 超时时间 4 分钟
                extra_body=get_extra_body(),
                **api_params
            )
            summary = response.choices[0].message.content.replace('\n', '')
            # 检查结果是否包含占位符文本（表明 LLM 未正确执行任务）
            if '视频标题' in summary:
                raise Exception('包含"视频标题"')
            logger.info(summary)
            # 从响应中提取 JSON 对象
            summary = re.findall(r'\{.*?\}', summary)[0]
            summary = json.loads(summary)
            summary = {
                'title': summary['title'].replace('title:', '').strip(),
                'summary': summary['summary'].replace('summary:', '').strip()
            }
            if 'title' in summary['title']:
                raise Exception('Invalid summary')
            success = True
            break
        except Exception as e:
            # 失败时追加提示，引导 LLM 输出正确格式
            retry_message += '\nSummarize the video in JSON format:\n```json\n{"title": "", "summary": ""}\n```'
            logger.warning(f'总结失败\n{e}')
            time.sleep(1)  # 重试前等待 1 秒

    if not success:
        raise Exception(f'总结失败')

    # 提取摘要结果
    title = summary['title']
    summary_text = summary['summary']
    tags = info['tags']

    # ========== 第二步：翻译摘要 ==========
    # 读取并格式化摘要翻译提示词
    summary_trans_prompt = get_custom_prompt('SUMMARY_TRANSLATION_PROMPT', DEFAULT_SUMMARY_TRANSLATION_PROMPT)
    summary_trans_prompt = summary_trans_prompt.format(target_language=target_language)
    messages = [
        {'role': 'system', 'content': summary_trans_prompt},
        {'role': 'user',
         'content': (
             f'The title of the video is "{title}". '
             f'The summary of the video is "{summary_text}". Tags: {tags}.\n'
             f'Please translate the above title and summary and tags into '
             f'{target_language} in JSON format. '
             f'```json\n{{"title": "", "summary", ""， "tags": []}}\n```. '
             f'Remember to tranlate the title and the summary and tags '
             f'into {target_language} in JSON.'
         )},
    ]
    max_retries = 10  # 翻译最多重试 10 次
    for retry in range(max_retries):
        try:
            api_params = get_api_params()
            response = client.chat.completions.create(
                model=get_model_name(),
                messages=messages,
                timeout=240,
                extra_body=get_extra_body(),
                **api_params
            )
            summary = response.choices[0].message.content.replace('\n', '')
            logger.info(summary)
            # 提取 JSON 响应
            summary = re.findall(r'\{.*?\}', summary)[0]
            summary = json.loads(summary)
            # 验证翻译结果不包含占位符
            if target_language in summary['title'] or target_language in summary['summary']:
                raise Exception('Invalid translation')

            # 清理标题中可能的多余引号
            title = summary['title'].strip()
            if (title.startswith('"') and title.endswith('"')) or \
               (title.startswith('"') and title.endswith('"')) or \
               (title.startswith("'") and title.endswith("'")) or \
               (title.startswith("'") and title.endswith("'")) or \
               (title.startswith('《') and title.endswith('》')):  # 《》
                title = title[1:-1]

            # 组装最终结果
            result = {
                'title': title,
                'author': info['uploader'],
                'summary': summary['summary'],
                'tags': summary['tags'],
                'language': target_language
            }
            return result
        except Exception as e:
            logger.warning(f'总结翻译失败 (重试 {retry+1}/{max_retries})\n{e}')
            time.sleep(1)

    raise RuntimeError(f"摘要翻译在 {max_retries} 次重试后仍然失败")


def translation_postprocess(result):
    """
    对 LLM 翻译结果进行后处理，清理和规范化文本。

    处理项包括：
    - 删除括号内的注释内容
    - 替换省略号
    - 保留数字中的逗号
    - 替换特殊字符
    - 术语统一（如 AI -> 人工智能）

    Args:
        result: 原始翻译结果字符串

    Returns:
        清理后的翻译字符串
    """
    # 删除中文全角圆括号内的补充说明内容
    # 注意：正则中用了半角 [^)]*，实际匹配的是全角左括号到半角右括号之间的内容
    result = re.sub(r'\（[^)]*\）', '', result)
    # 将英文省略号替换为中文逗号
    result = result.replace('...', '，')
    # 保留数字中的千位分隔逗号
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    # 替换特殊字符
    result = result.replace('²', '的平方')  # ² -> 的平方
    result = result.replace('————', '：')  # ---- -> ：
    result = result.replace('——', '：')              # —— -> ：
    result = result.replace('°', '度')                    # ° -> 度
    # 术语统一
    result = result.replace("AI", '人工智能')      # AI -> 人工智能
    result = result.replace('变压器', "Transformer")    # 变压器 -> Transformer
    return result


def valid_translation(text, translation):
    """
    验证 LLM 翻译结果的有效性和格式。

    检查翻译结果是否包含了多余的提示文本、是否过长或过短，
    以及是否含有不应出现的禁用词。

    Args:
        text: 原文
        translation: LLM 返回的翻译

    Returns:
        (is_valid: bool, processed_translation: str) 元组
        is_valid=False 时，processed_translation 包含错误提示消息
        is_valid=True 时，processed_translation 是后处理后的翻译
    """
    # 处理 Markdown 代码块包裹的翻译
    if translation.startswith('```') and translation.endswith('```'):
        translation = translation[3:-3]
        return True, translation_postprocess(translation)

    # 处理引号包裹的翻译
    if (translation.startswith('"') and translation.endswith('"')) or \
       (translation.startswith('"') and translation.endswith('"')):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)

    # 处理包含"翻译："前缀的格式
    if '翻译' in translation and '："' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and '："' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and ':"' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    # 对短原文，检查翻译是否过长（可能是包含了解释而非纯翻译）
    if len(text) <= 10:
        if len(translation) > 15:
            return False, 'Only translate the following sentence and give me the result.'
    # 检查翻译长度是否合理（翻译不应比原文短太多）
    elif len(translation) > len(text) * 0.75:
        return False, 'The translation is too long. Only translate the following sentence and give me the result.'

    # 检查翻译中是否包含禁用词（表明 LLM 添加了多余的解释）
    forbidden = [
        '翻译', '这句', '\n',
        '简体中文', '中文',
        'translate', 'Translate', 'translation', 'Translation'
    ]
    translation = translation.strip()
    for word in forbidden:
        if word in translation:
            return False, f"Don't include `{word}` in the translation. Only translate the following sentence and give me the result."

    return True, translation_postprocess(translation)


def split_text_into_sentences(para):
    """
    将段落文本按中文标点符号拆分为句子列表。

    支持句号、问号、感叹号、省略号等作为分句依据，
    同时保持引号的完整性（将终止符后的引号归入当前句）。

    Args:
        para: 待拆分的段落文本

    Returns:
        拆分后的句子列表
    """
    # 在句尾标点后插入换行符作为分句标记
    para = re.sub(r'([。！？?])([，。！？?"\'》])', r"\1\n\2", para)
    # 处理英文省略号（6个点）后的分句
    para = re.sub(r'(\.{6})([，。！？?"\'》])', r"\1\n\2", para)
    # 处理中文省略号（2个...）后的分句
    para = re.sub(r'(…{2})([，。！？?"\'》])', r"\1\n\2", para)
    # 处理引号：如果引号前有终止符，将分句符放到引号后
    para = re.sub(r'([。！？?]["\'])([，。！？?"\'》])', r'\1\n\2', para)
    # 去除段尾多余的换行
    para = para.rstrip()
    # 按换行分割为句子列表
    return para.split("\n")


def split_sentences(translation):
    """
    将翻译结果中的每个条目按句子拆分为更小的单元。

    用于 TTS 合成的句子切分：将长翻译按标点拆分为短句，
    并为每个短句分配精确的时间戳。

    Args:
        translation: 翻译结果列表，每项包含 start、end、text、speaker、translation

    Returns:
        拆分后的翻译结果列表，每项对应一个短句
    """
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item['translation']

        # 将翻译文本拆分为句子
        sentences = split_text_into_sentences(translation_text)

        # 估算每个字符的时间长度（基于当前条目的总时长）
        duration_per_char = (item['end'] - item['start']) / len(translation_text)

        for sentence in sentences:
            # 根据句子长度分配结束时间
            sentence_end = start + duration_per_char * len(sentence)

            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": text,
                "speaker": speaker,
                "translation": sentence
            })

            # 更新下一句的开始时间
            start = sentence_end

    return output_data


def _translate(summary, transcript, target_language='简体中文'):
    """
    内部函数：逐行翻译转录文本（带历史上下文的 LLM 调用）。

    翻译策略：
    - 为每个用户设置系统提示词和 few-shot 示例
    - 维护最近 30 条翻译历史作为上下文（帮助 LLM 保持翻译风格一致）
    - 每条翻译最多重试 30 次
    - 遇到服务器错误时重建客户端连接

    Args:
        summary: 视频摘要字典（包含标题和简介）
        transcript: 转录结果列表
        target_language: 目标语言

    Returns:
        翻译结果字符串列表，与 transcript 顺序一一对应

    Raises:
        RuntimeError: 某行翻译在 30 次重试后仍然失败
    """
    # 初始化 OpenAI 客户端
    client = OpenAI(
        base_url=get_config('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        api_key=get_config('OPENAI_API_KEY')
    )

    # 构建视频背景信息（帮助 LLM 理解上下文）
    info = (f'This is a video called "{summary["title"]}". '
            f'{summary["summary"]}.')

    full_translation = []  # 存储所有翻译结果

    # 构建系统提示词
    system_prompt = get_custom_prompt('TRANSLATION_SYSTEM_PROMPT', DEFAULT_TRANSLATION_SYSTEM_PROMPT)
    system_prompt = system_prompt.format(info=info, target_language=target_language)

    # 获取 few-shot 示例
    fewshot = get_custom_fewshot()

    # 固定的系统消息和 few-shot 示例（每次请求都会包含）
    fixed_message = [
        {'role': 'system', 'content': system_prompt},
    ] + fewshot

    # 历史上下文队列：保留最近 30 条翻译对
    history = deque(maxlen=30)

    # 逐行翻译
    for line in transcript:
        text = line['text']

        translation = ''
        ok = False
        retry_message = 'Only translate the quoted sentence and give me the final translation.'

        # 每行最多重试 30 次
        for retry in range(30):
            # 构建消息：固定部分 + 历史上下文 + 当前翻译请求
            messages = fixed_message + \
                list(history) + [{'role': 'user',
                                  'content': f'使用地道的中文Translate:"{text}"'}]

            try:
                api_params = get_api_params()
                response = client.chat.completions.create(
                    model=get_model_name(),
                    messages=messages,
                    timeout=240,
                    extra_body=get_extra_body(),
                    **api_params
                )
                translation = response.choices[0].message.content.replace('\n', '')
                logger.info(f'原文：{text}')
                logger.info(f'译文：{translation}')

                # 验证翻译结果
                success, translation = valid_translation(text, translation)
                if not success:
                    retry_message += translation
                    raise Exception('Invalid translation')

                ok = True
                break
            except Exception as e:
                logger.error(e)
                error_msg = str(e)
                # 遇到服务器内部错误，重建客户端连接
                if 'Internal Server Error' in error_msg:
                    client = OpenAI(
                        base_url=get_config('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                        api_key=get_config('OPENAI_API_KEY')
                    )
                time.sleep(1)  # 重试前等待

        if not ok:
            raise RuntimeError(
                f'翻译行在 30 次重试后仍然失败: {text[:50]}')

        # 保存当前翻译
        full_translation.append(translation)

        # 更新历史上下文
        history.append({'role': 'user', 'content': f'Translate:"{text}"'})
        history.append({'role': 'assistant', 'content': f'翻译："{translation}"'})

        # 小延迟避免 API 限流
        time.sleep(0.1)

    return full_translation


def translate(folder, target_language='简体中文'):
    """
    主入口：对指定目录执行完整的字幕翻译流程。

    完整流程：
    1. 读取视频元数据（download.info.json）
    2. 读取转录结果（transcript.json）
    3. 生成/读取视频摘要（summary.json）
    4. 执行逐行翻译
    5. 将翻译结果按句子拆分并添加时间戳
    6. 保存为 translation.json

    Args:
        folder: 视频处理目录
        target_language: 目标语言

    Returns:
        True 表示翻译成功

    Raises:
        FileNotFoundError: 元数据或转录文件不存在时抛出
    """
    # 检查翻译结果是否已存在（跳过已处理的目录）
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True

    # 读取视频元数据
    info_path = os.path.join(folder, 'download.info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f'元数据文件不存在: {info_path}'
            f'，请确认下载步骤已正确执行')

    # 读取转录结果
    transcript_path = os.path.join(folder, 'transcript.json')
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(
            f'转录文件不存在: {transcript_path}'
            f'，请确认语音识别步骤已正确执行')

    # 加载并提取视频信息
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    info = get_necessary_info(info)

    # 加载转录数据
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    # 生成或读取视频摘要
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path, 'r', encoding='utf-8'))
    else:
        summary = summarize(info, transcript, target_language)
        if summary is None:
            logger.error(f'Failed to summarize {folder}')
            return False
        # 保存摘要到文件
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # 执行逐行翻译
    translation_path = os.path.join(folder, 'translation.json')
    translation = _translate(summary, transcript, target_language)

    # 将翻译结果与转录数据合并
    for i, line in enumerate(transcript):
        line['translation'] = translation[i]

    # 按句子拆分翻译结果，便于 TTS 处理
    transcript = split_sentences(transcript)

    # 保存最终翻译结果
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    return True


def translate_all_transcript_under_folder(folder, target_language):
    """
    递归扫描目录，对所有包含转录结果的目录执行翻译。

    用于批量处理场景：自动遍历目录树，找到所有需要翻译的视频目录。

    Args:
        folder: 根目录路径（相对或绝对）
        target_language: 目标语言

    Returns:
        执行结果消息字符串

    Raises:
        FileNotFoundError: 未找到任何视频处理目录时抛出
    """
    # 相对路径转为绝对路径
    if not os.path.isabs(folder):
        folder = str(PROJECT_ROOT / folder)

    logger.info(f'开始翻译: {folder}')
    found_video_dir = False

    # 递归遍历所有子目录
    for root, dirs, files in os.walk(folder):
        # 跳过不包含视频数据的目录
        if ('download.info.json' not in files
                and 'transcript.json' not in files
                and 'translation.json' not in files):
            continue
        found_video_dir = True

        # 检查转录文件是否存在
        if 'transcript.json' not in files:
            raise FileNotFoundError(
                f'发现视频目录 {root} '
                f'但缺少 transcript.json，'
                f'请确认语音识别步骤已正确执行'
                f'。目录内容: {files}'
            )

        # 跳过已翻译的目录
        if 'translation.json' in files:
            continue

        # 执行翻译
        translate(root, target_language)

    if not found_video_dir:
        parent_contents = (
            os.listdir(os.path.dirname(folder))
            if os.path.exists(os.path.dirname(folder))
            else []
        )
        raise FileNotFoundError(
            f'在 {folder} 下未找到任何视频处理目录\n'
            f'请确认该目录下存在包含 '
            f'download.info.json 或 transcript.json 的子目录。\n'
            f'父目录内容: {parent_contents}'
        )

    return f'Translated all videos under {folder}'


def translate_transcripts_in_folders(folder_list, target_language='简体中文'):
    """
    处理指定目录列表中的字幕翻译。

    与 translate_all_transcript_under_folder 的区别：
    接收明确的目录列表而非根目录，不递归遍历。

    Args:
        folder_list: 需要处理的目录路径列表（单个字符串或列表）
        target_language: 目标语言

    Returns:
        格式化的执行结果字符串
    """
    # 统一为列表格式
    if isinstance(folder_list, str):
        folder_list = [folder_list]

    success_list = []
    fail_list = []

    for subdir in folder_list:
        subdir = os.path.abspath(subdir)
        files = os.listdir(subdir) if os.path.exists(subdir) else []

        # 检查转录文件是否存在
        if 'transcript.json' not in files:
            fail_list.append(f"{subdir}: 缺少 transcript.json")
            continue

        # 跳过已翻译的目录
        if 'translation.json' in files:
            logger.info(f'Translation already exists in {subdir}')
            success_list.append(subdir)
            continue

        # 执行翻译
        try:
            translate(subdir, target_language)
            success_list.append(subdir)
        except Exception as e:
            logger.error(f'Error translating in {subdir}: {e}')
            fail_list.append(f"{subdir}: {e}")

    logger.info(
        f'翻译完成: 成功 {len(success_list)}/{len(folder_list)}, '
        f'失败 {len(fail_list)}')
    return f'成功: {len(success_list)}\n失败: {len(fail_list)}'


if __name__ == '__main__':
    # 模块直接运行时的示例用法
    translate_all_transcript_under_folder(
        r'videos\TED-Ed\20240227 Can you solve the magical maze riddle - Alex Rosenthal',
        '简体中文'
    )
