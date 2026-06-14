# 本地 ASR 语音转写方案 — OpenLess + 本地大模型模拟 OpenAI Whisper API

## 一、背景与目标

**硬件环境：** Windows 11 / RTX 4070 12GB / 16GB RAM / i5-13400F

**需求：** 在 OpenLess 中使用本地 ASR 模型替代 OpenAI Whisper 云服务，对外暴露兼容 OpenAI Whisper 的 API 接口，主要识别中文（优先）和英文。

**核心诉求：** 速度快、质量好、中文识别准确率高。

---

## 二、可行性分析 — 结论：完全可行

| 评估维度 | 结论 |
|---------|------|
| 硬件能否跑动 | RTX 4070 12GB 绰绰有余，所有主流 ASR 模型 VRAM 占用均 < 5GB |
| API 兼容性 | 多个开源项目已提供 OpenAI Whisper 兼容的 `/v1/audio/transcriptions` 端点 |
| OpenLess 对接 | OpenLess 发送 `POST /v1/audio/transcriptions`（multipart/form-data），返回 `{"text": "..."}`，主流本地方案均支持 |
| 中文识别质量 | 专用中文模型（Paraformer-zh）CER ~1.95%，远优于 Whisper large-v3 的 ~5-8% |
| 响应速度 | 本地 GPU 推理，实时因子 RTF < 0.1，即 10 秒音频约 1 秒内返回 |

---

## 三、方案对比

| 方案 | 中文 CER | 英文质量 | 速度 | OpenAI API 兼容 | 部署难度 | 推荐度 |
|------|---------|---------|------|-----------------|---------|--------|
| **FunASR + Paraformer-zh** | **~1.95%** | 良好 | 极快（非自回归） | 支持 | 中等 | ★★★★★ |
| FunASR + SenseVoice-Small | ~3-4% | 良好 | 极快（5-15x Whisper） | 支持 | 中等 | ★★★★ |
| faster-whisper-server + large-v3 | ~5-8% | 优秀 | 快 | 完美（drop-in） | 简单 | ★★★ |
| whisper.cpp | ~5-8% | 优秀 | 快 | 需额外包装 | 较难 | ★★ |

---

## 四、推荐方案：FunASR + Paraformer-zh

### 为什么选这个

1. **中文识别质量碾压 Whisper** — Paraformer-zh 在 AISHELL-1 测试集上 CER ~1.95%，而 Whisper large-v3 约 5-8%，差距 3-4 倍
2. **速度极快** — 非自回归架构，比 Whisper 的自回归解码快一个数量级
3. **VRAM 占用低** — 全套模型（ASR + VAD + 标点 + ITN）总 VRAM < 3GB，RTX 4070 12GB 富余大量空间
4. **自带标点和逆文本正则化** — 输出带标点的自然文本，不需要额外后处理
5. **已支持 OpenAI 兼容 API** — FunASR 服务端已提供 `/v1/audio/transcriptions` 端点

### 备选方案：SenseVoice-Small

如果追求极致速度且可接受中文 CER 略升至 ~3-4%，可用 SenseVoice-Small：
- VRAM 仅 ~1-2GB
- 速度比 Whisper 快 5-15 倍
- 额外支持情感识别、音频事件检测

---

## 五、实施步骤

### Step 1：环境准备

```bash
# 1. 安装 CUDA Toolkit（如果还没装）
#    下载：https://developer.nvidia.com/cuda-downloads
#    选择 Windows → x86_64 → 本地安装包

# 2. 创建 Python 虚拟环境（推荐 Python 3.10）
python -m venv funasr-env
funasr-env\Scripts\activate

# 3. 安装 PyTorch（CUDA 版本）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装 FunASR
pip install funasr
```

### Step 2：下载模型

首次运行时模型会自动下载到 `C:\Users\<用户名>\.cache\modelscope\` 下，或手动下载：

- **ASR 模型：** `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- **VAD 模型：** `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- **标点模型：** `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`
- **ITN 模型：** `itn_zh-cn`

总下载大小约 2-3GB。

### Step 3：启动 OpenAI 兼容 API 服务

FunASR 提供了内置的 WebSocket/HTTP 服务。有两种方式：

**方式 A：使用 FunASR 自带服务（推荐）**

```bash
# 启动 FunASR 服务端（包含 OpenAI 兼容端点）
python -m funasr.bin.asr_server \
  --model-dir iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --vad-dir iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --punc-dir iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch \
  --itn-dir itn_zh-cn \
  --host 0.0.0.0 \
  --port 10095
```

> 注意：FunASR 的 OpenAI 兼容端点路径和端口可能因版本而异，启动后需确认实际端点。

**方式 B：使用 faster-whisper-server 作为 API 层（最简单的 API 兼容方案）**

如果 FunASR 的 API 兼容性在实际对接 OpenLess 时遇到问题，可以用更快的方式：

```bash
pip install faster-whisper-server
faster-whisper-server --host 0.0.0.0 --port 8000 --model large-v3 --compute-type int8
```

这是**零配置**的 OpenAI API 兼容方案，OpenLess 直接可用。

### Step 4：配置 OpenLess

在 OpenLess 的 ASR 设置中：

| 配置项 | 值 |
|-------|-----|
| ASR 供应商 | OpenAI Whisper |
| Base URL | `http://localhost:8000/v1`（faster-whisper-server）或 `http://localhost:10095/v1`（FunASR） |
| API Key | 任意非空字符串，如 `sk-local-dummy` |
| Model | `whisper-1` 或对应模型名 |

OpenLess 的 URL 规则会自动将 `http://localhost:8000/v1` 拼接为 `http://localhost:8000/v1/audio/transcriptions`。

### Step 5：验证

```bash
# 用 curl 测试 API 是否正常
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-local-dummy" \
  -F "file=@test_audio.wav" \
  -F "model=whisper-1"

# 期望返回
# {"text": "这是一段测试音频的转写结果"}
```

---

## 六、性能预期（RTX 4070 12GB）

| 指标 | Paraformer-zh | SenseVoice-Small | faster-whisper large-v3 int8 |
|------|--------------|------------------|------------------------------|
| 中文 CER | ~1.95% | ~3-4% | ~5-8% |
| 英文质量 | 良好 | 良好 | 优秀 |
| 10 秒音频推理时间 | < 0.5 秒 | < 0.3 秒 | < 1 秒 |
| VRAM 占用 | ~2GB | ~1.5GB | ~3GB |
| 启动加载时间 | ~10 秒 | ~5 秒 | ~15 秒 |

---

## 七、注意事项

1. **CUDA 版本匹配** — PyTorch CUDA 版本需与已安装的 NVIDIA 驱动匹配。RTX 4070 一般用 CUDA 12.x
2. **首次启动** — 模型下载需要网络，首次可能需要等待几分钟
3. **Windows 路径** — FunASR 的模型缓存路径含中文用户名时可能出问题，建议用户名用英文
4. **OpenLess 音频格式** — OpenLess 发送 16kHz 单声道 16bit PCM WAV，本地服务端需支持此格式（上述方案均支持）
5. **如遇 API 兼容问题** — FunASR 的 OpenAI 兼容端点是较新功能，如不稳定可随时切换到 faster-whisper-server 作为兜底

---

## 八、最终建议

```
首选：FunASR + Paraformer-zh   → 中文质量最好，适合以中文为主的场景
备选：faster-whisper-server     → API 兼容性最完美，零配置即用，中文质量尚可
```

建议先用 **faster-whisper-server** 快速跑通 OpenLess 对接（10 分钟内搞定），确认链路没问题后再切换到 **FunASR + Paraformer-zh** 获得更好的中文识别效果。
