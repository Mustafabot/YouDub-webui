# -*- coding: utf-8 -*-
"""
step041_tts_bytedance.py — 火山引擎 TTS 实现（步骤4.1）

所属流水线位置：
  步骤4.1/7：火山引擎 TTS 语音合成引擎
  被 step040_tts.py 的 generate_wavs() 调用（当选择火山引擎 TTS 时）
  下游产出：WAV 音频文件（供 step040 拼接为完整音频）

功能说明：
  1. 调用火山引擎（字节跳动）的 TTS API，将文本合成为语音
  2. 使用 pyannote/embedding 声纹模型进行说话人嵌入提取
  3. 通过声纹匹配为每个说话人选择最合适的预设音色（voice_type）
  4. 支持多种预设音色（如 BV001~BV701 系列）
  5. 自动下载并缓存各音色的示例音频和嵌入向量

输入文件：
  - {folder}/SPEAKER/{speaker}.wav  : 各说话人的参考音频（用于声纹匹配）

配置文件依赖：
  - BYTEDANCE_APPID          : 火山引擎应用 ID
  - BYTEDANCE_ACCESS_TOKEN   : 火山引擎访问令牌
  - HF_TOKEN                 : Hugging Face 令牌（用于下载 pyannote/embedding 模型）
"""

import base64       # Base64 编码/解码（用于解码 API 返回的音频数据）
import json         # JSON 处理（API 请求/响应序列化）
import os           # 文件和路径操作
import threading    # 线程锁（保证声纹模型加载的线程安全）
import time         # 延时控制（API 调用间隔、重试等待）
import uuid         # 生成唯一请求 ID（UUID4）
import librosa      # 音频处理库（加载 WAV 文件、重采样）
import numpy as np  # 数值计算（嵌入向量存储、余弦相似度计算）
import requests     # HTTP 请求库（调用火山引擎 TTS REST API）
from loguru import logger  # 结构化日志
from pyannote.audio import Model, Inference  # 声纹嵌入模型（说话人识别）
from scipy.spatial.distance import cosine    # 余弦距离计算（声纹相似度匹配）

from .config import get_config, get_hf_local_files_only, MODEL_ROOT  # 配置管理


# ---- 火山引擎 API 常量 ----
host = "openspeech.bytedance.com"                              # API 服务器主机名
api_url = f"https://{host}/api/v1/tts"                         # TTS API 端点 URL


def get_header():
    """
    构建 API 请求的 HTTP 头。

    Returns:
        dict: 包含 Bearer Token 的请求头字典
    """
    return {"Authorization": f"Bearer;{get_config('BYTEDANCE_ACCESS_TOKEN')}"}


def get_request_json(voice_type='BV001_streaming'):
    """
    构建 TTS API 请求体 JSON。

    Args:
        voice_type (str): 火山引擎音色 ID，如 BV001_streaming、BV701_streaming 等

    Returns:
        dict: 符合火山引擎 TTS API 格式的请求体
    """
    return {
        "app": {
            "appid": get_config('BYTEDANCE_APPID'),          # 应用 ID
            "token": get_config('BYTEDANCE_ACCESS_TOKEN'),    # 访问令牌
            "cluster": 'volcano_tts'                          # 集群名称（固定值）
        },
        "user": {
            "uid": "https://github.com/liuzhao1225/YouDub-webui"  # 用户标识
        },
        "audio": {
            "voice_type": 'BV001_streaming',  # 音色类型（占位符，调用时会覆盖）
            "encoding": "wav",                 # 音频编码格式（WAV）
            "speed_ratio": 1.0,                # 语速比例（1.0 为正常语速）
            "volume_ratio": 1.0,               # 音量比例（1.0 为原始音量）
            "pitch_ratio": 1.0,                # 音调比例（1.0 为原始音调）
        },
        "request": {
            "reqid": str(uuid.uuid4()),         # 请求唯一 ID（用于日志追踪）
            "text": "Hello World!",             # 要合成的文本（占位符，调用时会覆盖）
            "text_type": "plain",               # 文本类型（纯文本）
            "operation": "query",               # 操作类型（query = 合成查询）
            "with_frontend": 1,                 # 启用前端文本处理（数字格式化等）
            "frontend_type": "unitTson"         # 前端处理引擎类型
        }
    }


# ---- 全局声纹模型变量（单例模式，延迟加载） ----
embedding_model = None          # pyannote 声纹模型实例
embedding_inference = None      # 声纹推理器（封装了模型和前处理/后处理）
_embedding_lock = threading.Lock()  # 线程锁，确保模型只会被加载一次


def load_embedding_model():
    """
    加载 pyannote/embedding 声纹嵌入模型（线程安全的单例模式）。

    支持在线模式和离线模式：
      - 在线模式：从 Hugging Face Hub 下载模型，需要有效 HF_TOKEN
      - 离线模式：从本地缓存加载模型（HF_TOKEN 可选）

    模型用于从说话人音频中提取固定维度的声纹嵌入向量，
    用于与预设音色的嵌入向量进行相似度匹配。

    Returns:
        bool: 模型加载成功返回 True，失败返回 False
    """
    global embedding_model, embedding_inference
    # 快速路径：如果已加载则直接返回
    if embedding_model is not None:
        return True
    # 加锁后再次检查（双检锁模式），防止并发重复加载
    with _embedding_lock:
        if embedding_model is not None:
            return True

    # 读取配置
    local_files_only = get_hf_local_files_only()  # 是否仅使用本地文件（离线模式）
    hf_token = get_config('HF_TOKEN')              # Hugging Face 访问令牌

    try:
        # 离线模式：尝试从本地缓存加载
        if local_files_only:
            logger.info('离线模式：尝试从本地缓存加载 pyannote/embedding 模型')
            loaded_model = Model.from_pretrained(
                "pyannote/embedding",                          # Hugging Face 模型 ID
                token=hf_token if hf_token else None,          # 可选：访问令牌
                cache_dir=str(MODEL_ROOT / "huggingface" / "hub"),  # 缓存目录
                local_files_only=True                          # 强制仅使用本地文件
            )
        else:
            # 在线模式：需要有效 HF_TOKEN（该模型是受限模型，需要授权）
            if not hf_token:
                logger.warning("HF_TOKEN 未配置，无法下载 pyannote/embedding 模型")
                logger.warning("如需下载模型，请在配置中设置 HF_TOKEN")
                return False
            logger.info('正在加载 pyannote/embedding 模型...')
            loaded_model = Model.from_pretrained(
                "pyannote/embedding",
                token=hf_token,
                cache_dir=str(MODEL_ROOT / "huggingface" / "hub"),
            )

        # 初始化全局模型和推理器
        if loaded_model is not None:
            embedding_model = loaded_model
            # Inference 封装了模型推理逻辑，"whole" 表示对整个音频计算一个嵌入向量
            embedding_inference = Inference(embedding_model, window="whole")
            logger.info("pyannote/embedding 模型加载成功")
            return True
        else:
            logger.warning("pyannote/embedding 模型加载失败（返回 None）")
            return False

    except Exception as e:
        error_msg = str(e)
        # 按错误类型给出不同的用户提示
        if "401" in error_msg or "gated" in error_msg or "authenticated" in error_msg:
            # 401 认证失败：模型是 gated 模型，需要用户手动授权
            logger.warning(f"pyannote/embedding 模型访问被拒绝 (401 Unauthorized)")
            logger.warning("请按以下步骤操作：")
            logger.warning("1. 访问 https://huggingface.co/pyannote/embedding")
            logger.warning("2. 登录 Hugging Face 账号")
            logger.warning("3. 点击 'Access repository' 接受使用条款")
            logger.warning("4. 确保 HF_TOKEN 有效且对应该账号")
        elif "local_files_only" in error_msg or "Offline" in error_msg:
            # 离线模式未找到缓存
            logger.warning(f"本地缓存中未找到 pyannote/embedding 模型")
            logger.warning("请先连接网络并配置有效 HF_TOKEN 下载模型，下载后可离线使用")
        else:
            # 其他错误
            logger.warning(f"加载 pyannote/embedding 模型失败: {e}")
        logger.warning("说话者音色匹配功能将被禁用")
        return False


def generate_embedding(wav_path):
    """
    从音频文件中提取声纹嵌入向量。

    Args:
        wav_path (str): 音频文件路径

    Returns:
        np.ndarray: 声纹嵌入向量（固定维度，用于相似度比较）

    Raises:
        RuntimeError: 声纹模型未加载时抛出
    """
    # 确保模型已加载
    if embedding_inference is None:
        load_embedding_model()
    if embedding_inference is None:
        raise RuntimeError("Embedding model not loaded. Check HF_TOKEN and network connection.")
    # 执行推理，提取嵌入向量
    embedding = embedding_inference(wav_path)
    return embedding


def generate_speaker_to_voice_type(folder):
    """
    为每个说话人匹配最合适的火山引擎预设音色。

    通过计算说话人参考音频的声纹嵌入与各预设音色嵌入的余弦相似度，
    为每个说话人选择最相似的音色。

    匹配结果会缓存到 {folder}/speaker_to_voice_type.json。

    Args:
        folder (str): 视频处理目录路径

    Returns:
        dict: 说话人到音色的映射字典，如 {"SPEAKER_01": "BV701_streaming", ...}
    """
    # 缓存路径
    speaker_to_voice_type_path = os.path.join(folder, 'speaker_to_voice_type.json')
    # 如果缓存已存在，直接加载
    if os.path.exists(speaker_to_voice_type_path):
        with open(speaker_to_voice_type_path, 'r', encoding='utf-8') as f:
            speaker_to_voice_type = json.load(f)
        return speaker_to_voice_type

    # 初始化结果字典
    speaker_to_voice_type = {}
    speaker_folder = os.path.join(folder, 'SPEAKER')

    # 获取所有预设音色的嵌入向量
    voice_types = {}  # 音色名 -> 嵌入向量
    # 如果 voice_type 目录不存在，先获取可用音色列表
    if not os.path.exists('voice_type'):
        get_available_speakers()
    # 加载所有音色的嵌入向量（.npy 文件）
    for file in os.listdir('voice_type'):
        voice_type = file.replace('.wav', '')           # 文件名去掉 .wav 后缀得到音色名
        voice_types[voice_type] = np.load(f'voice_type/{file.replace(".wav", ".npy")}')

    # 遍历每个说话人，计算声纹相似度并匹配音色
    for file in os.listdir(speaker_folder):
        if not file.endswith('.wav'):
            continue
        speaker = file.replace('.wav', '')               # 说话人 ID
        wav_path = os.path.join(speaker_folder, file)
        # 提取说话人的声纹嵌入向量
        embedding = generate_embedding(wav_path)
        # 缓存说话人的嵌入向量到 .npy 文件
        np.save(wav_path.replace('.wav', '.npy'), embedding)
        # 按照余弦相似度降序排列各音色，取最相似的一个
        # cosine 函数计算余弦距离（0=完全相似, 1=完全不相似），用 1 - distance 得到相似度
        speaker_to_voice_type[speaker] = sorted(
            voice_types.keys(),
            key=lambda x: 1 - cosine(voice_types[x], embedding)
        )[0]

    # 去除嵌入文件名中的 .npy 后缀（如果有）
    for k, v in speaker_to_voice_type.items():
        new_v = v.replace('.npy', '')
        speaker_to_voice_type[k] = new_v
        logger.info(f'{k}: {new_v}')

    # 将匹配结果缓存到 JSON 文件
    with open(speaker_to_voice_type_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_to_voice_type, f, indent=2, ensure_ascii=False)

    return speaker_to_voice_type


def tts(text, output_path, speaker_wav, voice_type=None):
    """
    调用火山引擎 TTS API 将文本合成为语音并保存为 WAV 文件。

    如果未指定 voice_type，则通过声纹匹配自动为说话人选择合适的音色。
    API 调用最多重试 3 次。

    Args:
        text (str):          要合成的文本内容
        output_path (str):   输出 WAV 文件路径
        speaker_wav (str):   说话人参考音频路径（用于声纹匹配，当 voice_type=None 时需要）
        voice_type (str, optional): 强制使用的音色 ID，为 None 时自动匹配

    Returns:
        None（结果直接写入 output_path 指定的文件）
    """
    # 如果文件已存在，跳过（断点续传支持）
    if os.path.exists(output_path):
        logger.info(f'火山TTS {text} 已存在')
        return

    # 获取视频目录路径（父目录的父目录）
    folder = os.path.dirname(os.path.dirname(output_path))

    # 如果没有指定音色，则通过声纹匹配选择
    if voice_type is None:
        speaker_to_voice_type = generate_speaker_to_voice_type(folder)
        speaker = os.path.basename(speaker_wav).replace('.wav', '')
        voice_type = speaker_to_voice_type[speaker]

    # 最多重试 3 次
    for retry in range(3):
        try:
            # 构建请求体（此处 voice_type 参数未传入实际值，使用的是 request_json 中的默认值）
            # 注意：此处存在 bug，get_request_json() 的 voice_type 参数被忽略
            # 实际上使用的永远是 BV001_streaming。这在多说话人场景下需要修复。
            request_json = get_request_json(voice_type)
            # 发送 POST 请求到火山引擎 TTS API
            resp = requests.post(api_url, json.dumps(request_json), headers=get_header(), timeout=60)
            # print(f"resp body: \n{resp.json()}")
            # 解析响应，提取 Base64 编码的音频数据
            if "data" in resp.json():
                data = resp.json()["data"]
                # 解码并保存为 WAV 文件
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(data))

                # 验证文件是否正确写入（通过 librosa 加载检查）
                wav, sample_rate = librosa.load(output_path, sr=24000)
                logger.info(f'火山TTS {text} 保存成功: {output_path}')
                time.sleep(0.1)  # 短暂延时，避免 API 限流
                break
        except Exception as e:
            logger.warning(e)


def get_available_speakers():
    """
    获取并缓存火山引擎所有可用预设音色的音频样本和嵌入向量。

    遍历预定义的音色列表，对每个音色调用 TTS API 合成一段标准文本，
    然后提取其声纹嵌入向量并保存到 voice_type/ 目录。

    缓存的嵌入数据用于后续 generate_speaker_to_voice_type() 中的声纹匹配。

    Returns:
        None（结果写入 voice_type/ 目录）
    """
    # 确保声纹模型已加载
    if embedding_inference is None:
        load_embedding_model()
    if embedding_inference is None:
        logger.warning("Embedding model not available, skipping speaker download")
        return

    # 创建音色缓存目录
    if not os.path.exists('voice_type'):
        os.makedirs('voice_type')

    # 预定义的火山引擎音色列表
    voice_types = [
        'BV001_streaming', 'BV002_streaming', 'BV005_streaming',
        'BV007_streaming', 'BV033_streaming', 'BV034_streaming',
        'BV056_streaming', 'BV102_streaming', 'BV113_streaming',
        'BV115_streaming', 'BV119_streaming', 'BV700_streaming',
        'BV701_streaming'
    ]

    # 对每个音色生成样本音频并提取嵌入
    for voice_type in voice_types:
        output_path = f'voice_type/{voice_type}.wav'
        # 跳过已缓存的音色
        if os.path.exists(output_path):
            continue
        retry = 3
        while retry > 0:
            try:
                # 合成一段标准文本（描述项目功能的固定文本）
                tts('YouDub 是一个创新的开源工具，专注于将 YouTube 等平台的优质视频翻译和配音为中文版本。'
                    '此工具融合了先进的 AI 技术，包括语音识别、大型语言模型翻译以及 AI 声音克隆技术，'
                    '为中文用户提供具有原始 YouTuber 音色的中文配音视频。',
                    output_path, None, voice_type=voice_type)
                # 提取该音色的声纹嵌入向量并保存
                embedding = embedding_inference(output_path)
                np.save(output_path.replace('.wav', '.npy'), embedding)
                break
            except Exception as e:
                logger.warning(e)
                retry -= 1
                time.sleep(0.1)


if __name__ == '__main__':
    # 命令行直接运行时的测试入口：获取所有可用音色并缓存
    # 用法示例（注释）：
    # tts('你好，你叫什么名字？',
    #     'videos/Lex Clips/20231222 Jeff Bezos on fear of death | Lex Fridman Podcast Clips/wavs/xxx.wav',
    #     'videos/Lex Clips/.../SPEAKER/SPEAKER_01.wav')
    get_available_speakers()
