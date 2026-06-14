import tempfile
import os
from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.responses import JSONResponse
import uvicorn
from funasr import AutoModel

app = FastAPI()

# 加载模型（启动时一次性加载）
print("正在加载 FunASR 模型，请稍候...")
asr_model = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    device="cuda",
    disable_update=True,
)
print("模型加载完成！")

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: str = Form(None),
    authorization: str = Header(None),
):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = asr_model.generate(input=tmp_path)
        text = result[0]["text"] if result else ""
        return JSONResponse(content={"text": text})
    finally:
        os.unlink(tmp_path)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
