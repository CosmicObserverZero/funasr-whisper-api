# FunASR 本地 ASR 语音转写服务

基于 FunASR + Paraformer-zh 的本地语音转写 API，兼容 OpenAI Whisper 接口，可对接 OpenLess。

## 硬件要求

- NVIDIA 显卡（推荐 RTX 3060 及以上，显存 >= 4GB）
- 已安装 CUDA Toolkit

## OpenLess 配置

| 配置项 | 值 |
|-------|-----|
| ASR 供应商 | OpenAI Whisper |
| Base URL | `http://localhost:8000/v1` |
| API Key | `sk-local-dummy`（任意非空字符串） |
| Model | `whisper-1` |

## 运行

### 首次运行（需要安装环境）

```bash
# 1. 进入项目目录
cd d:\ai_code\本地openai whisper api

# 2. 创建虚拟环境
python -m venv funasr-env

# 3. 激活虚拟环境
funasr-env\Scripts\activate

# 4. 安装 PyTorch（CUDA 版本）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 安装 FunASR 及依赖
pip install funasr modelscope fastapi uvicorn python-multipart

# 6. 启动服务（首次会自动下载模型，约 2-3GB）
python server.py
```

### 日常运行

```bash
cd d:\ai_code\本地openai whisper api
funasr-env\Scripts\python.exe server.py
```

看到 `Uvicorn running on http://0.0.0.0:8000` 即表示启动成功。

### 验证

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-local-dummy" \
  -F "file=@你的音频文件.wav" \
  -F "model=whisper-1"
```

返回 `{"text": "转写结果"}` 即正常。

## 清理删除

如不再使用，删除以下内容即可：

### 1. 关闭服务

在运行服务的终端按 `Ctrl+C` 停止。

### 2. 删除虚拟环境

```bash
rmdir /s /q "d:\ai_code\本地openai whisper api\funasr-env"
```

### 3. 删除模型缓存（约 2-3GB）

```bash
rmdir /s /q "%USERPROFILE%\.cache\modelscope"
```

### 4. 删除项目文件

```bash
del "d:\ai_code\本地openai whisper api\server.py"
del "d:\ai_code\本地openai whisper api\README.md"
```

保留 `本地ASR语音转写方案.md` 即可。

## 文件说明

| 文件 | 说明 |
|------|------|
| `server.py` | API 服务脚本 |
| `funasr-env/` | Python 虚拟环境 |
| `本地ASR语音转写方案.md` | 方案分析文档 |
