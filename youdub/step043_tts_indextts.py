# -*- coding: utf-8 -*-
"""
step043_tts_indextts.py — IndexTTS-2 零样本 TTS 实现（步骤4.3）

所属流水线位置：
  步骤4.3/7：IndexTTS-2 语音合成引擎（声音克隆）
  被 step040_tts.py 的 generate_wavs() 调用（当选择 IndexTTS 时）
  下游产出：WAV 音频文件（供 step040 拼接为完整音频）

功能说明：
  1. 封装 IndexTTS-2 模型（IndexTeam/IndexTTS-2），实现零样本语音克隆
  2. 支持 CUDA GPU 加速，根据显存大小自动选择 fp16 精度和 DeepSpeed 配置
  3. 提供模型生命周期管理：延迟加载、线程安全、显存清理
  4. 支持自动设备和精度检测（CPU / GPU，fp16 开关）
  5. 每段 TTS 合成最多重试 3 次，确保鲁棒性

模型依赖（约 3GB）：
  - IndexTeam/IndexTTS-2 from Hugging Face
  - 模型文件存放在 {PROJECT_ROOT}/models/index-tts/ 目录

输入文件：
  - {folder}/SPEAKER/{speaker}.wav  : 说话人参考音频（用于声音克隆）

配置文件依赖：
  - INDEXTTS_MODEL_DIR         : 模型目录路径（默认 models/index-tts）
  - INDEXTTS_USE_FP16          : 是否启用半精度推理（默认 true）
  - INDEXTTS_USE_DEEPSPEED     : 是否启用 DeepSpeed 加速（默认 false）
"""

import os          # 路径操作、文件存在性检查
import gc          # Python 垃圾回收器，用于手动清理内存
import contextlib  # 上下文管理工具，用于临时重定向标准输出
import io          # 内存中的 I/O 流，用于捕获模型加载时的控制台输出
from loguru import logger      # 结构化日志
import torch       # PyTorch：GPU 内存管理、设备检测
import time        # 计时与延迟

from .config import get_config, PROJECT_ROOT, MODEL_ROOT  # 配置管理


# ---- IndexTTS 库的条件导入 ----
# 该库可能未安装，因此使用 try/except 处理
try:
    # 从 indextts 包导入 IndexTTS2 推理类
    from indextts.infer_v2 import IndexTTS2
    INDEXTTS_AVAILABLE = True          # 库可用标志
except ImportError:
    INDEXTTS_AVAILABLE = False         # 库不可用标志
    logger.warning(
        "indextts library not available. "
        "Install from GitHub: pip install git+https://github.com/index-tts/index-tts.git"
    )


# ---- 全局模型变量（单例模式） ----
model = None           # IndexTTS2 模型实例（全局共享）
model_config = None    # 模型配置字典（记录加载时的参数）


def _detect_tts_config(device='auto'):
    """
    自动检测并返回 IndexTTS-2 的最佳运行配置。

    当 device='auto' 时，根据 GPU 显存大小自动决策：
      - 显存 >= 8GB  : 使用 CUDA，按配置决定 fp16 和 DeepSpeed
      - 显存 < 8GB   : 使用 CUDA，强制 fp16（节约显存），不使用 DeepSpeed
      - 无 GPU       : 使用 CPU，禁用 fp16 和 DeepSpeed

    Args:
        device (str): 设备策略，'auto' 为自动检测，'cpu' 强制 CPU，或其他设备名

    Returns:
        dict: 包含 device、use_fp16、use_deepspeed、model_dir 的配置字典
    """
    # 读取模型目录配置（支持相对路径和绝对路径）
    model_dir = get_config('INDEXTTS_MODEL_DIR', 'models/index-tts')
    if not os.path.isabs(model_dir):
        model_dir = str(PROJECT_ROOT / model_dir)

    # 读取精度和加速配置（字符串转布尔值）
    use_fp16 = get_config('INDEXTTS_USE_FP16', 'true').lower() in ('true', '1', 'yes')
    use_deepspeed = get_config('INDEXTTS_USE_DEEPSPEED', 'false').lower() in ('true', '1', 'yes')

    if device == 'auto':
        # 自动检测是否有可用 GPU
        if not torch.cuda.is_available():
            logger.info('未检测到 GPU，IndexTTS 将使用 CPU 推理')
            return {'device': 'cpu', 'use_fp16': False, 'use_deepspeed': False, 'model_dir': model_dir}

        # 获取 GPU 显存大小
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8:
            # 大显存：使用用户配置的精度和加速设置
            config = {
                'device': 'cuda',
                'use_fp16': use_fp16,
                'use_deepspeed': use_deepspeed,
                'model_dir': model_dir
            }
        elif vram_gb >= 4:
            # 中显存：强制 fp16，禁用 DeepSpeed
            config = {
                'device': 'cuda',
                'use_fp16': True,
                'use_deepspeed': False,
                'model_dir': model_dir
            }
        else:
            # 小显存：强制 fp16，禁用 DeepSpeed
            config = {
                'device': 'cuda',
                'use_fp16': True,
                'use_deepspeed': False,
                'model_dir': model_dir
            }
        logger.info(
            f'GPU 显存: {vram_gb:.1f}GB，IndexTTS 配置: '
            f'fp16={config["use_fp16"]}, deepspeed={config["use_deepspeed"]}'
        )
        return config

    elif device == 'cpu':
        # 强制 CPU 模式
        return {'device': 'cpu', 'use_fp16': False, 'use_deepspeed': False, 'model_dir': model_dir}
    else:
        # 使用指定的设备
        return {'device': device, 'use_fp16': use_fp16, 'use_deepspeed': use_deepspeed, 'model_dir': model_dir}


def init_indextts():
    """
    初始化 IndexTTS-2 模型（加载入口）。

    由外部调用（如 model_manager），触发模型的延迟加载。
    等同于调用 load_model() 使用默认的自动检测配置。
    """
    load_model()


def load_model(device='auto', use_fp16=None, use_deepspeed=None):
    """
    加载 IndexTTS-2 模型（线程安全的单例模式）。

    支持通过参数覆盖自动检测的配置。模型加载后全局共享，
    避免每个 TTS 请求都重新加载。

    Args:
        device (str):       设备策略，'auto' 为自动检测
        use_fp16 (bool, optional): 是否使用半精度，覆盖自动检测结果
        use_deepspeed (bool, optional): 是否使用 DeepSpeed，覆盖自动检测结果

    Raises:
        RuntimeError: IndexTTS 库未安装，或模型文件不存在，或模型加载失败
    """
    global model, model_config
    # 快速路径：模型已加载则直接返回
    if model is not None:
        return

    # 检查 IndexTTS 库是否可导入
    if not INDEXTTS_AVAILABLE:
        raise RuntimeError(
            "IndexTTS 未安装。请在 WebUI 设置页「模型管理」中点击「下载全部缺失模型」自动安装，\n"
            "或手动执行: pip install git+https://github.com/index-tts/index-tts.git"
        )

    # 自动检测运行配置
    config = _detect_tts_config(device)
    # 允许参数覆盖检测结果
    if use_fp16 is not None:
        config['use_fp16'] = use_fp16
    if use_deepspeed is not None:
        config['use_deepspeed'] = use_deepspeed
    # 保存配置到全局变量（供后续 tts() 函数使用）
    model_config = config

    model_dir = config['model_dir']
    # 模型配置文件路径（IndexTTS-2 使用 config.yaml 描述模型结构）
    cfg_path = os.path.join(model_dir, 'config.yaml')

    # 检查模型文件是否存在
    if not os.path.exists(cfg_path):
        raise RuntimeError(
            f'IndexTTS 模型配置文件不存在: {cfg_path}\n'
            f'请先在设置页「模型管理」中下载 IndexTTS-2 模型，'
            f'或手动下载到 {model_dir} 目录。\n'
            f'下载命令: huggingface-cli download IndexTeam/IndexTTS-2 --local-dir={model_dir}'
        )

    # 设置 PyTorch CUDA 内存分配策略（减少内存碎片）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    logger.info(
        f'加载 IndexTTS v2 模型 (fp16={config["use_fp16"]}, '
        f'deepspeed={config["use_deepspeed"]}) 从 {model_dir}'
    )
    t_start = time.time()

    # 模型加载
    load_error = None
    try:
        model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=config['use_fp16'],
            use_cuda_kernel=False,                  # 禁用 CUDA 内核（兼容性更好）
            use_deepspeed=config['use_deepspeed'],
        )
    except Exception as e:
        load_error = e
        logger.warning(f'加载 IndexTTS v2 模型失败: {e}')

    # 如果模型加载失败，抛出详细错误信息
    if model is None and load_error:
        raise RuntimeError(
            f'IndexTTS v2 模型加载失败。\n'
            f'错误详情: {load_error}'
        )

    t_end = time.time()
    logger.info(f'IndexTTS v2 模型加载完成，耗时 {t_end - t_start:.2f}s')
    # 输出加载后的 GPU 显存使用情况
    _log_cuda_memory()


def _log_cuda_memory():
    """
    记录当前 CUDA 显存使用情况到日志。

    输出已分配显存、已预留显存和总显存大小（GB 为单位）。
    仅在 CUDA 可用时执行。
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9          # 已分配（实际使用）
        reserved = torch.cuda.memory_reserved() / 1e9            # 已预留（包括缓存）
        total = torch.cuda.get_device_properties(0).total_memory / 1e9  # 总显存
        logger.info(
            f'CUDA 显存: 已分配={allocated:.2f}GB, '
            f'已预留={reserved:.2f}GB, 总计={total:.2f}GB'
        )


def tts(text, output_path, speaker_wav, device='auto'):
    """
    使用 IndexTTS-2 模型进行零样本语音合成（声音克隆）。

    根据说话人参考音频的音色，将指定文本合成为该音色的语音。
    最多重试 3 次，每次失败时清理 GPU 缓存并重试。

    Args:
        text (str):           要合成的文本内容
        output_path (str):    输出 WAV 文件路径
        speaker_wav (str):    说话人参考音频路径（提供音色特征）
        device (str):         设备策略，默认 'auto'

    Raises:
        RuntimeError: IndexTTS 未安装、参考音频不存在、模型未加载或合成失败
        FileNotFoundError: 参考音频文件不存在
    """
    global model

    # 检查 IndexTTS 库是否可用
    if not INDEXTTS_AVAILABLE:
        raise RuntimeError(
            "IndexTTS 未安装。请在 WebUI 设置页「模型管理」中点击「下载全部缺失模型」自动安装，\n"
            "或手动执行: pip install git+https://github.com/index-tts/index-tts.git"
        )

    # 检查说话人参考音频是否存在
    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(
            f'参考音频不存在: {speaker_wav}，请确认说话人分离步骤已正确执行'
        )

    # 如果输出文件已存在，跳过合成（断点续传支持）
    if os.path.exists(output_path):
        logger.info(f'IndexTTS 音频已存在，跳过: {output_path}')
        return

    # 模型尚未加载时执行加载
    if model is None:
        load_model(device)

    # 加载仍然失败时报错
    if model is None:
        raise RuntimeError('IndexTTS v2 模型加载失败，无法进行语音合成')

    # 判断是否使用 GPU（根据模型配置中的设备信息）
    use_cuda = model_config and model_config.get('device') != 'cpu'

    # 最多重试 3 次
    last_error = None
    for retry in range(3):
        try:
            # 合成前清理 GPU 缓存，释放显存
            if use_cuda:
                torch.cuda.empty_cache()

            # 重定向 IndexTTS 的控制台输出到内存缓冲区（避免干扰日志）
            with contextlib.redirect_stdout(io.StringIO()):
                # 禁用梯度计算（推理模式下不需要）
                with torch.no_grad():
                    # 执行模型推理生成语音
                    model.infer(
                        spk_audio_prompt=speaker_wav,  # 说话人参考音频
                        text=text,                     # 要合成的文本
                        output_path=output_path,       # 输出文件路径
                        verbose=False,                 # 不输出详细信息
                    )

            # 验证输出文件是否成功生成
            if not os.path.exists(output_path):
                raise RuntimeError(f'音频文件保存失败: {output_path}')

            # 合成成功后清理 GPU 缓存
            if use_cuda:
                torch.cuda.empty_cache()
            break  # 成功退出重试循环

        except Exception as e:
            last_error = e
            logger.warning(f'IndexTTS 生成失败 (尝试 {retry+1}/3): {e}')
            # 失败后清理 GPU 缓存和 Python 垃圾
            if use_cuda:
                torch.cuda.empty_cache()
                gc.collect()
    else:
        # 当 for 循环未 break（即 3 次全部失败）时执行
        text_display = f'{text[:50]}...' if len(text) > 50 else text
        raise RuntimeError(
            f'IndexTTS 生成失败，已重试 3 次。最后一次错误: {last_error}\n'
            f'文本: {text_display}'
        )


def cleanup_indextts():
    """
    清理 IndexTTS-2 模型，释放 GPU 显存。

    删除模型对象、清空配置、触发垃圾回收、清空 CUDA 缓存。
    在需要重新加载模型或释放资源时调用。
    """
    global model, model_config
    if model is not None:
        del model          # 删除模型引用
        model = None
    model_config = None    # 清空配置
    gc.collect()           # 强制 Python 垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
    logger.info('IndexTTS v2 模型已清理，显存已释放')


def get_model_info():
    """
    获取当前 IndexTTS 模型的加载状态和配置信息。

    Returns:
        dict: 包含 device、use_fp16、use_deepspeed、model_dir 的配置字典。
              若模型未加载，device 为 'not_loaded'。
    """
    if model_config is None:
        return {
            'device': 'not_loaded',
            'use_fp16': False,
            'use_deepspeed': False,
            'model_dir': 'unknown'
        }
    return dict(model_config)  # 返回副本，防止外部修改


if __name__ == '__main__':
    # 命令行直接运行时的测试入口：交互式文本合成
    speaker_wav = r'videos\TED-Ed\20231121 Why did the US try to kill all the bison？ - Andrew C. Isenberg\audio_vocals.wav'
    while True:
        text = input('请输入：')
        tts(text, f'playground/{text}.wav', speaker_wav)
