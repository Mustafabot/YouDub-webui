# YouDub-webui: 优质视频中文化工具

## 🚀 快速开始

只需数步，即可开始使用 YouDub-webui：

### 第一步：安装依赖

```bash
# Windows 用户直接双击运行
setup_windows.bat
```

<br />

### 第二步：启动程序

```bash
# Windows 用户直接双击运行
run_windows.bat
```

完成！浏览器会自动打开 YouDub-webui 界面，粘贴视频链接即可开始使用。

***

## 目录

- [YouDub-webui: 优质视频中文化工具](#youdub-webui-优质视频中文化工具)
  - [🚀 快速开始](#-快速开始)
  - [目录](#目录)
  - [简介](#简介)
  - [主要特点](#主要特点)
  - [📦 安装指南](#-安装指南)
    - [环境要求](#环境要求)
    - [安装步骤](#安装步骤)
  - [⚙️ 环境变量配置](#️-环境变量配置)
    - [配置文件说明](#配置文件说明)
    - [必填配置项](#必填配置项)
    - [可选配置项](#可选配置项)
  - [🚀 运行程序](#-运行程序)
  - [使用步骤](#使用步骤)
    - [1.](#1-全自动-do-everything) **[全自动 (Do Everything)](#1-全自动-do-everything)**
    - [2.](#2-下载视频-download-video) **[下载视频 (Download Video)](#2-下载视频-download-video)**
    - [3.](#3-人声分离-demucs-interface) **[人声分离 (Demucs Interface)](#3-人声分离-demucs-interface)**
    - [4.](#4-语音识别-whisper-inference) **[语音识别 (Whisper Inference)](#4-语音识别-whisper-inference)**
    - [5.](#5-字幕翻译-translation-interface) **[字幕翻译 (Translation Interface)](#5-字幕翻译-translation-interface)**
    - [6.](#6-语音合成-tts-interface) **[语音合成 (TTS Interface)](#6-语音合成-tts-interface)**
    - [7.](#7-视频合成-synthesize-video-interface) **[视频合成 (Synthesize Video Interface)](#7-视频合成-synthesize-video-interface)**
  - [❓ 常见问题解答 (FAQ)](#-常见问题解答-faq)
  - [🔧 故障排查指南](#-故障排查指南)
  - [技术细节](#技术细节)
    - [AI 语音识别](#ai-语音识别)
    - [大型语言模型翻译](#大型语言模型翻译)
    - [视频处理](#视频处理)

## 简介

`YouDub-webui` 是 `YouDub`项目的网页交互版本，基于 `Gradio` 构建，为用户提供简易操作界面来访问和使用 `YouDub`的强大功能。`YouDub`是一个开创性的开源工具，旨在将 YouTube 和其他平台上的高质量视频翻译和配音成中文版本。该工具结合了最新的 AI 技术，包括语音识别、大型语言模型翻译，以及 AI 声音克隆技术，提供与原视频相似的中文配音，为中文用户提供卓越的观看体验。

`YouDub-webui` 适用于多种场景，包括教育、娱乐和专业翻译，特别适合那些希望将国外优秀视频内容本地化的用户。此工具的简洁界面使得即使是非技术用户也能轻松上手，实现视频的快速中文化处理。

***

## 主要特点

`YouDub-webui` 融合了多项先进技术，提供了一套完整的视频中文化工具包，其主要特点包括：

- **视频下载**: 支持通过链接直接下载 YouTube 视频。无论是单个视频、播放列表还是频道内的多个视频，均能轻松下载。
- **AI 语音识别**: 利用先进的 AI 技术，将视频中的语音高效转换为文字。不仅提供精确的语音到文本转换，还能自动对齐时间并识别不同说话者，极大地增强了信息的丰富性和准确性。
- **大型语言模型翻译**: 结合大型语言模型如 GPT，实现快速且精准的中文翻译。无论是俚语还是专业术语，均能得到恰当的翻译，确保内容的准确性与地道性。
- **视频处理**: 综合了音视频同步处理、字幕添加、视频播放速度调整和帧率设置等多项功能。用户可以根据需要生成高质量的最终视频，实现无缝的观看体验。

`YouDub-webui` 的这些特点使其成为一个强大且易于使用的视频中文化工具，无论是个人用户还是专业团队，都能从中受益。

## 📦 安装指南

### 环境要求

| 项目        | 最低要求 | 推荐配置                      |
| --------- | ---- | ------------------------- |
| Python 版本 | 3.9  | 3.10+                     |
| 内存        | 8GB  | 16GB 以上                   |
| 显卡（可选）    | -    | NVIDIA GPU（支持 CUDA 11.8+） |
| 硬盘空间      | 10GB | 20GB 以上                   |

### 安装步骤

#### Windows 用户（推荐）

**一键安装**

双击运行 `setup_windows.bat`，脚本会自动完成：

- 创建虚拟环境
- 安装所有依赖
- 配置 CUDA 支持（如有 NVIDIA 显卡）

#### 手动安装（进阶用户）

如果你需要自定义安装，可以按以下步骤操作：

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
venv\Scripts\activate

# 3. 安装基础依赖
pip install -r requirements.txt

# 4. 安装 CUDA 版本的 PyTorch（可选，如有 NVIDIA 显卡，可大幅提升速度）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ⚙️ 环境变量配置

### 配置文件说明

YouDub-webui 使用 `.env` 文件管理所有配置。首次使用前，请：

1. 复制 `.env.example` 文件，重命名为 `.env`
2. 根据你的需求填写配置项
3. 保存文件后重启程序

### 必填配置项

| 配置项              | 说明                   | 获取方式                                              |
| ---------------- | -------------------- | ------------------------------------------------- |
| `OPENAI_API_KEY` | OpenAI API 密钥，用于翻译功能 | [OpenAI 官网](https://platform.openai.com/api-keys) |

### 可选配置项

| 配置项               | 说明                            | 默认值                                                    |
| ----------------- | ----------------------------- | ------------------------------------------------------ |
| `MODEL_NAME`      | 使用的 GPT 模型名称                  | `gpt-3.5-turbo`                                        |
| `OPENAI_API_BASE` | 自定义 API 地址（用于第三方 API 或本地部署模型） | -                                                      |
| `HF_TOKEN`        | Hugging Face Token，用于说话人识别    | [Hugging Face](https://huggingface.co/settings/tokens) |
| `HF_ENDPOINT`     | Hugging Face 镜像地址（解决国内下载问题）   | `https://hf-mirror.com`                                |
| `APPID`           | 火山引擎 APPID，用于高质量 TTS          | [火山引擎](https://console.volcengine.com/)                |
| `ACCESS_TOKEN`    | 火山引擎 Access Token             | 同上                                                     |
| `BILI_SESSDATA`   | Bilibili 登录凭证（用于上传）           | 浏览器 Cookie                                             |
| `BILI_BILI_JCT`   | Bilibili CSRF Token（用于上传）     | 浏览器 Cookie                                             |

> **提示**: 配置项说明不区分大小写，`BILI_BASE64` 为旧版配置，现已改用 `BILI_SESSDATA` 和 `BILI_BILI_JCT`。

## 🚀 运行程序

### Windows 用户（推荐）

直接双击运行 `run_windows.bat`，程序会自动：

- 激活虚拟环境
- 启动 Web 服务
- 打开浏览器界面

### 手动运行

```bash
# 激活虚拟环境
venv\Scripts\activate

# 启动程序
python app.py
```

启动成功后，在浏览器中访问 `http://localhost:7860` 即可使用。

## 使用步骤

### 1. **全自动 (Do Everything)**

此界面是一个一站式的解决方案，它将执行从视频下载到视频合成的所有步骤。

- **Root Folder**: 设置视频文件的根目录。
- **Video URL**: 输入视频或播放列表或频道的URL。
- **Number of videos to download**: 设置要下载的视频数量。
- **Resolution**: 选择下载视频的分辨率。
- **Demucs Model**: 选择用于音频分离的Demucs模型。
- **Demucs Device**: 选择音频分离的处理设备。
- **Number of shifts**: 设置音频分离时的移位数。
- **Whisper Model**: 选择用于语音识别的Whisper模型。
- **Whisper Download Root**: 设置Whisper模型的下载根目录。
- **Whisper Batch Size**: 设置Whisper处理的批量大小。
- **Whisper Diarization**: 选择是否进行说话者分离。
- **Translation Target Language**: 选择字幕的目标翻译语言。
- **Force Bytedance**: 选择是否强制使用Bytedance语音合成。
- **Subtitles**: 选择是否在视频中包含字幕。
- **Speed Up**: 设置视频播放速度。
- **FPS**: 设置视频的帧率。
- **Max Workers**: 设置处理任务的最大工作线程数。
- **Max Retries**: 设置任务失败后的最大重试次数。
- **Auto Upload Video**: 选择是否自动上传视频到Bilibili。

### 2. **下载视频 (Download Video)**

此界面用于单独下载视频。

- **Video URL**: 输入视频或播放列表或频道的URL。
- **Output Folder**: 设置视频下载后的输出文件夹。
- **Resolution**: 选择下载视频的分辨率。
- **Number of videos to download**: 设置要下载的视频数量。

### 3. **人声分离 (Demucs Interface)**

此界面用于从视频中分离人声。

- **Folder**: 设置包含视频的文件夹。
- **Model**: 选择用于音频分离的Demucs模型。
- **Device**: 选择音频分离的处理设备。
- **Progress Bar in Console**: 选择是否在控制台显示进度条。
- **Number of shifts**: 设置音频分离时的移位数。

### 4. **语音识别 (Whisper Inference)**

此界面用于从视频音频中进行语音识别。

- **Folder**: 设置包含视频的文件夹。
- **Model**: 选择用于语音识别的Whisper模型。
- **Download Root**: 设置Whisper模型的下载根目录。
- **Device**: 选择语音识别的处理设备。
- **Batch Size**: 设置Whisper处理的批量大小。
- **Diarization**: 选择是否进行说话者分离。

### 5. **字幕翻译 (Translation Interface)**

此界面用于将识别出的语音转换为字幕并翻译。

- **Folder**: 设置包含视频的文件夹。
- **Target Language**: 选择字幕的目标翻译语言。

### 6. **语音合成 (TTS Interface)**

此界面用于将翻译后的文字转换为语音。

- **Folder**: 设置包含视频的文件夹。
- **Force Bytedance**: 选择是否强制使用Bytedance语音合成。

### 7. **视频合成 (Synthesize Video Interface)**

此界面用于将视频、字幕和语音合成为最终视频。

- **Folder**: 设置包含视频的文件夹。
- **Subtitles**: 选择是否在视频中包含字幕。
- **Speed Up**: 设置视频播放速度。
- **FPS**: 设置视频的帧率。
- **Resolution**: 选择视频的分辨率。

## ❓ 常见问题解答 (FAQ)

### Q1: 支持哪些 Python 版本？

**A**: 推荐使用 Python 3.10及以上版本。

### Q2: 必须使用虚拟环境吗？

**A**: 强烈建议使用虚拟环境！虚拟环境可以避免依赖冲突，保持系统环境干净。我们提供的 `setup_windows.bat` 会自动创建虚拟环境。

### Q3: 依赖安装失败怎么办？

**A**:

1. 检查 Python 版本是否为 3.9+
2. 尝试升级 pip：`python -m pip install --upgrade pip`
3. 使用国内镜像源：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
4. 如果是 TTS 安装失败，可以先跳过，使用火山引擎 TTS 替代

### Q4: 如何启用 CUDA 加速？

**A**:

1. 确保安装了 NVIDIA 显卡驱动
2. 安装 CUDA Toolkit 11.8 或 12.x
3. 运行：`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
4. 验证：在 Python 中执行 `import torch; print(torch.cuda.is_available())` 应返回 `True`

### Q5: OPENAI\_API\_KEY 如何获取？

**A**:以OpenAI官方为例：

1. 访问 [OpenAI Platform](https://platform.openai.com/) 注册账号
2. 进入 [API Keys](https://platform.openai.com/api-keys) 页面
3. 点击 "Create new secret key" 生成密钥
4. 注意保存好密钥，只显示一次！

### Q6: Hugging Face 模型下载很慢怎么办？

**A**: 设置中添加国内镜像地址，加速模型下载。

### Q7: 翻译效果不好怎么办？

**A**:

1. 尝试更换模型：修改 `MODEL_NAME` 
2. 检查原视频语音识别是否准确，口音重可能影响识别
3. 可以手动编辑 `translation.json` 文件修正翻译

### Q8: 语音合成没有声音怎么办？

**A**:

1. 检查是否配置了火山引擎 `APPID` 和 `ACCESS_TOKEN`
2. 确保网络连接正常，火山引擎 TTS 需要联网
3. 尝试勾选 "Force Bytedance" 强制使用火山引擎
4. 如果使用 F5-TTS，首次运行会下载模型，请耐心等待

### Q9: 处理视频时内存不足怎么办？

**A**:

1. 降低 Whisper 模型大小（使用 `base` 或 `small` 而非 `large`）
2. 减小 Batch Size
3. 关闭其他占用内存的程序
4. 增加虚拟内存（页面文件）大小

### Q10: 可以处理本地视频文件吗？

**A**: 可以！将视频文件放入工作目录，然后跳过"下载视频"步骤，直接从"人声分离"开始处理即可。

***

## 🔧 故障排查指南

### 第一步：环境检查

#### 检查 Python 版本

```bash
python --version
```

✅ 正常：显示 `Python 3.10.x` 或 `3.11.x`
❌ 异常：版本过高或过低 → 安装正确的 Python 版本

#### 检查虚拟环境

运行 `setup_windows.bat` 后，检查是否存在 `venv` 文件夹
✅ 正常：`venv` 文件夹存在且内含多个子文件夹
❌ 异常：文件夹不存在 → 重新运行安装脚本

#### 检查依赖安装

```bash
venv\Scripts\activate
pip list
```

✅ 正常：能看到 `torch`, `gradio`, `openai` 等包
❌ 异常：缺少关键包 → 重新运行 `pip install -r requirements.txt`

### 第二步：网络问题排查

#### 检查网络连接

```bash
ping api.openai.com
```

✅ 正常：能收到响应
❌ 异常：请求超时 → 检查网络设置或代理配置

#### 测试 OpenAI API

在 `.env` 配置好 API Key 后，运行以下测试：

```python
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

✅ 正常：输出 "Hello!" 或类似回复
❌ 异常：报错 → 检查 API Key 是否正确，账户是否有余额

#### 检查 Hugging Face 连接

如果模型下载失败：

1. 验证 `HF_TOKEN` 是否正确配置
2. 尝试设置 `HF_ENDPOINT=https://hf-mirror.com`
3. 检查防火墙是否阻止下载

### 第三步：CUDA 问题排查

#### 验证 CUDA 是否可用

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
```

✅ 正常：`CUDA available: True`，显示正确的版本号
❌ 异常：返回 False → 重新安装 CUDA 版本的 PyTorch

#### 常见 CUDA 错误解决方案

1. **"CUDA out of memory"**：减小模型大小或 batch size
2. **"CUDA kernel errors"**：升级显卡驱动到最新版本
3. **"No CUDA GPUs are available"**：确认显卡支持 CUDA 且驱动正常

### 第四步：常见错误信息及解决方案

| 错误信息                                         | 可能原因          | 解决方案                          |
| -------------------------------------------- | ------------- | ----------------------------- |
| `ModuleNotFoundError: No module named 'xxx'` | 缺少依赖包         | 运行 `pip install xxx`          |
| `AuthenticationError`                        | API Key 错误或过期 | 检查 `.env` 中的 `OPENAI_API_KEY` |
| `RateLimitError`                             | API 调用频率超限    | 稍后重试                          |
| `ConnectionError`                            | 网络连接问题        | 检查网络、代理、防火墙设置                 |
| `RuntimeError: CUDA error`                   | GPU 内存不足或驱动问题 | 减小模型大小、升级驱动                   |

## 技术细节

### AI 语音识别

我们的 AI 语音识别功能现在基于 [WhisperX](https://github.com/m-bain/whisperX) 实现。WhisperX 是一个高效的语音识别系统，建立在 OpenAI 开发的 Whisper 系统之上。它不仅能够精确地将语音转换为文本，还能自动对齐时间，并识别每句话的说话人物。这种先进的处理方式不仅提高了处理速度和准确度，还为用户提供了更丰富的信息，例如说话者的识别。

### 大型语言模型翻译

我们的翻译功能继续使用 OpenAI API 提供的各种模型，包括官方的 GPT 模型。
