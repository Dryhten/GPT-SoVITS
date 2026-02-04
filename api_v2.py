"""
# WebAPI文档

` python api_v2.py -a 0.0.0.0 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"0.0.0.0"（对外访问）`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 15,                  # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
    "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
    "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
    "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
    "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 获取可用音色列表 (v1)

endpoint: `/v1/voices`

GET:
```
http://127.0.0.1:9880/v1/voices
```

RESP:
成功: 返回 JSON，如 `{"voices": ["xiaofeng", ...]}`，为当前 wav/voices.json 中且对应 wav 文件存在的音色名称列表。可在 /v1/tts 的 voice 参数中使用。

参考音频要求（wav/ 下各音色对应的 .wav）：
- 服务启动时会自动检查时长；**超过 10 秒** 的参考音频会被自动裁剪为前 10 秒并写回原文件，**小于 3 秒** 仅打日志（TTS 可能仍会报错）。
- 若使用 SoVITS V3，须为每个参考音频提供提示文本：`wav/<basename>.txt` + `wav/<basename>.lang`，或 `wav/<basename>.json`（含 prompt_text、prompt_lang）。

### 流式 TTS (v1)

endpoint: `/v1/tts/stream`

POST: Body 与 POST /v1/tts 相同，如 `{"text": "...", "voice": "xiaofeng"}`，可选 `text_lang`、`media_type`。
成功: 直接返回 audio/wav 流（首包为 WAV 头，后续为 PCM 数据块），适合长文本以降低首包延迟。

### 网页端

启动 api_v2 后，在浏览器访问根路径即可使用简单网页：GET `/` 返回 `web/index.html`。
网页提供：音色列表（GET /v1/voices）、合成并播放（POST /v1/tts + GET /audio/{file_id}）、流式合成并播放（POST /v1/tts/stream）。

"""

import json
import os
import re
import secrets
import sys
import traceback
import uuid
from typing import Generator, Union

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

# Fixed voice directory: reference audio files by filename (e.g. seed1.wav) are resolved under this folder.
WAV_VOICE_DIR = "wav"
# Voice name -> filename mapping (e.g. {"xiaofeng": "seed1.wav"}) for /v1/tts.
VOICES_JSON = os.path.join(WAV_VOICE_DIR, "voices.json")
# Temporary TTS output: POST /v1/tts saves audio here and returns URL; files expire after 10 min.
TTS_AUDIO_DIR = os.path.join(now_dir, "TEMP", "tts_audio")
TTS_AUDIO_EXPIRE_SECONDS = 10 * 60  # 10 minutes
# Short-link: public URL uses short_id, mapping to real filename (uuid.wav).
TTS_AUDIO_SHORT_ID_LENGTH = 10  # chars for short id (URL-safe)

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
import threading

# In-memory short_id <-> filename for /audio/{short_id}. Cleared on restart; cleaned when file expires.
_short_id_to_file = {}
_file_to_short_id = {}
_short_id_lock = threading.Lock()

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0 (allow external access)")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI(
    title="GPT-SoVITS API",
    description="Text-to-speech API for GPT-SoVITS. Synthesize speech from text with reference voice (wav/ folder or full path). Interactive docs: /docs",
    version="1.0.0",
)


@APP.get("/health", summary="Health check", include_in_schema=False)
async def health():
    """Health check for load balancers and orchestration (e.g. k8s)."""
    return {"status": "ok"}


@APP.get("/favicon.ico", summary="Favicon", include_in_schema=False)
async def favicon():
    """Return 204 No Content so browsers do not repeatedly request favicon and get 404."""
    return Response(status_code=204, headers={"Cache-Control": "public, max-age=86400"})


import asyncio
import time


def _register_tts_short_id(short_id: str, filename: str):
    """Register short_id -> filename for GET /audio/{short_id}. Thread-safe."""
    with _short_id_lock:
        _short_id_to_file[short_id] = filename
        _file_to_short_id[filename] = short_id


def _unregister_tts_file(filename: str):
    """Remove mapping for filename (e.g. when file is deleted). Thread-safe."""
    with _short_id_lock:
        short_id = _file_to_short_id.pop(filename, None)
        if short_id is not None:
            _short_id_to_file.pop(short_id, None)


def _generate_short_id():
    """Return a URL-safe short id (e.g. 10 chars). Collision-resistant."""
    return secrets.token_urlsafe(TTS_AUDIO_SHORT_ID_LENGTH)[: TTS_AUDIO_SHORT_ID_LENGTH]


def _cleanup_expired_tts_audio():
    """Delete TTS audio files older than TTS_AUDIO_EXPIRE_SECONDS and unregister their short_ids."""
    if not os.path.isdir(TTS_AUDIO_DIR):
        return
    now = time.time()
    for name in os.listdir(TTS_AUDIO_DIR):
        if not re.match(r"^[0-9a-f-]+\.wav$", name):
            continue
        path = os.path.join(TTS_AUDIO_DIR, name)
        try:
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > TTS_AUDIO_EXPIRE_SECONDS:
                _unregister_tts_file(name)
                os.remove(path)
        except OSError:
            pass


async def _cleanup_tts_audio_loop():
    """Background task: every 60s delete expired TTS audio files."""
    while True:
        await asyncio.sleep(60)
        _cleanup_expired_tts_audio()


@APP.on_event("startup")
async def startup_event():
    os.makedirs(TTS_AUDIO_DIR, exist_ok=True)
    # 启动时自动将 wav/voices.json 中超长参考音频裁剪为 3~10 秒（写回原文件）
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _ensure_ref_audio_durations_at_startup)
    asyncio.create_task(_cleanup_tts_audio_loop())


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    ref_audio_filename: str = None  # e.g. "seed1.wav", resolved under wav/
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = 2
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    # Author: AkagawaTsurunaki
    # Issue:
    #   Stack overflow probabilistically occurs
    #   when the function `sf_writef_short` of `libsndfile_64bit.dll` is called
    #   using the Python library `soundfile`
    # Note:
    #   This is an issue related to `libsndfile`, not this project itself.
    #   It happens when you generate a large audio tensor (about 499804 frames in my PC)
    #   and try to convert it to an ogg file.
    # Related:
    #   https://github.com/RVC-Boss/GPT-SoVITS/issues/1199
    #   https://github.com/libsndfile/libsndfile/issues/1023
    #   https://github.com/bastibe/python-soundfile/issues/396
    # Suggestion:
    #   Or split the whole audio data into smaller audio segment to avoid stack overflow?

    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)



    # See: https://docs.python.org/3/library/threading.html
    # The stack size of this thread is at least 32768
    # If stack overflow error still occurs, just modify the `stack_size`.
    # stack_size = n * 4096, where n should be a positive integer.
    # Here we chose n = 4096.
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")

    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def resolve_voice_name_to_filename(voice_name: str):
    """
    Resolve voice name (e.g. xiaofeng) to ref_audio_filename under wav/ using wav/voices.json.
    Returns (filename, None) or (None, JSONResponse) on error.
    """
    if not voice_name or not voice_name.strip():
        return None, JSONResponse(status_code=400, content={"message": "voice (voice name) is required"})
    voices_path = os.path.join(now_dir, VOICES_JSON)
    if not os.path.exists(voices_path):
        example = '{"voice_name": "filename.wav"}'
        return None, JSONResponse(
            status_code=400,
            content={"message": f"Voice mapping not found: {VOICES_JSON}. Add wav/voices.json with {example}."},
        )
    try:
        with open(voices_path, "r", encoding="utf-8") as f:
            voices = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, JSONResponse(status_code=400, content={"message": f"Invalid or unreadable {VOICES_JSON}: {e}"})
    if not isinstance(voices, dict):
        return None, JSONResponse(status_code=400, content={"message": f"{VOICES_JSON} must be a JSON object (voice_name -> filename)."})
    filename = voices.get(voice_name.strip())
    if not filename:
        return None, JSONResponse(
            status_code=400,
            content={"message": f"Voice name not found: {voice_name}. Add it to {VOICES_JSON}."},
        )
    filename = filename.strip()
    normalized = os.path.normpath(filename)
    if normalized.startswith("..") or ".." in normalized or os.path.isabs(normalized):
        return None, JSONResponse(status_code=400, content={"message": "Voice filename must be a simple filename under wav/, no path traversal."})
    wav_dir_abs = os.path.abspath(os.path.join(now_dir, WAV_VOICE_DIR))
    resolved = os.path.abspath(os.path.join(now_dir, WAV_VOICE_DIR, filename))
    try:
        common = os.path.commonpath([resolved, wav_dir_abs])
    except ValueError:
        common = ""
    if common != wav_dir_abs or not os.path.exists(resolved):
        return None, JSONResponse(status_code=400, content={"message": f"Voice file not found under {WAV_VOICE_DIR}/: {filename}"})
    return filename, None


def get_available_voice_names():
    """
    Load wav/voices.json and return list of voice names for which the corresponding wav file exists under wav/.
    Returns ([name, ...], None) or ([], JSONResponse) on read/parse error.
    """
    voices_path = os.path.join(now_dir, VOICES_JSON)
    if not os.path.exists(voices_path):
        return [], None
    try:
        with open(voices_path, "r", encoding="utf-8") as f:
            voices = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], None
    if not isinstance(voices, dict):
        return [], None
    wav_dir_abs = os.path.abspath(os.path.join(now_dir, WAV_VOICE_DIR))
    result = []
    for name, filename in voices.items():
        if not name or not isinstance(filename, str):
            continue
        filename = filename.strip()
        normalized = os.path.normpath(filename)
        if normalized.startswith("..") or ".." in normalized or os.path.isabs(normalized):
            continue
        resolved = os.path.abspath(os.path.join(now_dir, WAV_VOICE_DIR, filename))
        try:
            common = os.path.commonpath([resolved, wav_dir_abs])
        except ValueError:
            continue
        if common == wav_dir_abs and os.path.isfile(resolved):
            result.append(name.strip())
    return result, None


def load_prompt_for_ref_audio(ref_audio_filename: str, default_prompt_lang: str = "zh"):
    """
    Load prompt_text and prompt_lang from sidecar files under wav/ for the given ref audio filename.
    Looks for wav/<basename>.txt + wav/<basename>.lang, or wav/<basename>.json with prompt_text and prompt_lang.
    Returns (prompt_text, prompt_lang). If no sidecar found, returns ("", default_prompt_lang).
    """
    base, _ = os.path.splitext(ref_audio_filename)
    if not base:
        return "", default_prompt_lang
    wav_dir = os.path.join(now_dir, WAV_VOICE_DIR)
    # Try .json first
    json_path = os.path.join(wav_dir, base + ".json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                pt = data.get("prompt_text") or ""
                pl = data.get("prompt_lang") or default_prompt_lang
                return (pt.strip() if isinstance(pt, str) else ""), (pl.strip() if isinstance(pl, str) else default_prompt_lang)
        except (json.JSONDecodeError, OSError):
            pass
    # Try .txt + .lang
    txt_path = os.path.join(wav_dir, base + ".txt")
    lang_path = os.path.join(wav_dir, base + ".lang")
    prompt_text = ""
    prompt_lang = default_prompt_lang
    if os.path.isfile(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                prompt_text = raw.split("\n")[0] if raw else ""
        except OSError:
            pass
    if os.path.isfile(lang_path):
        try:
            with open(lang_path, "r", encoding="utf-8") as f:
                prompt_lang = f.read().strip() or default_prompt_lang
        except OSError:
            pass
    return prompt_text, prompt_lang


def resolve_ref_audio_path(req: dict):
    """
    If ref_audio_filename is provided (and ref_audio_path is not), resolve it under WAV_VOICE_DIR
    and set req["ref_audio_path"]. Path traversal is forbidden. Returns JSONResponse on error, None otherwise.
    """
    ref_audio_path = req.get("ref_audio_path") or ""
    ref_audio_filename = req.get("ref_audio_filename") or ""
    if ref_audio_path:
        return None
    if not ref_audio_filename:
        return None
    # Forbid path traversal and absolute paths
    normalized_name = os.path.normpath(ref_audio_filename)
    if normalized_name.startswith("..") or ".." in normalized_name or os.path.isabs(normalized_name):
        return JSONResponse(
            status_code=400,
            content={"message": "ref_audio_filename must be a simple filename under wav/, path traversal not allowed"},
        )
    wav_dir_abs = os.path.abspath(os.path.join(now_dir, WAV_VOICE_DIR))
    resolved_path = os.path.abspath(os.path.join(now_dir, WAV_VOICE_DIR, ref_audio_filename))
    try:
        common = os.path.commonpath([resolved_path, wav_dir_abs])
    except ValueError:
        common = ""
    if common != wav_dir_abs or not os.path.exists(resolved_path):
        return JSONResponse(
            status_code=400,
            content={"message": f"ref_audio file not found or not under {WAV_VOICE_DIR}/: {ref_audio_filename}"},
        )
    req["ref_audio_path"] = resolved_path
    return None


# TTS 要求参考音频时长为 3~10 秒（与 TTS.py _set_prompt_semantic 一致）
REF_AUDIO_DURATION_MIN_SEC = 3.0
REF_AUDIO_DURATION_MAX_SEC = 10.0


def _trim_ref_audio_to_valid_duration(path: str, duration: float, voice_name: str = ""):
    """将超长参考音频裁剪为前 REF_AUDIO_DURATION_MAX_SEC 秒并写回原文件。"""
    data, sr = sf.read(path)
    n_keep = int(REF_AUDIO_DURATION_MAX_SEC * sr)
    if data.ndim == 2:
        data_trimmed = data[:n_keep, :]
    else:
        data_trimmed = data[:n_keep]
    base, ext = os.path.splitext(path)
    tmp_path = base + ".trim_tmp" + (ext or ".wav")
    try:
        sf.write(tmp_path, data_trimmed, sr)
        os.replace(tmp_path, path)
        label = f"音色「{voice_name}」" if voice_name else os.path.basename(path)
        print(f"[ref_audio] {label} 原时长 {duration:.2f}s > {REF_AUDIO_DURATION_MAX_SEC}s，已自动裁剪为前 {REF_AUDIO_DURATION_MAX_SEC}s")
    except Exception as e:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        print(f"[ref_audio] 裁剪失败 {path}: {e}")


def _ensure_ref_audio_durations_at_startup():
    """
    服务启动时检查 wav/voices.json 中所有参考音频时长；
    若超过 10 秒则自动裁剪为前 10 秒并写回，小于 3 秒仅打日志。
    """
    voices_path = os.path.join(now_dir, VOICES_JSON)
    if not os.path.isfile(voices_path):
        return
    try:
        with open(voices_path, "r", encoding="utf-8") as f:
            voices = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(voices, dict):
        return
    wav_dir_abs = os.path.abspath(os.path.join(now_dir, WAV_VOICE_DIR))
    for name, filename in voices.items():
        if not name or not isinstance(filename, str):
            continue
        filename = filename.strip()
        normalized = os.path.normpath(filename)
        if normalized.startswith("..") or ".." in normalized or os.path.isabs(normalized):
            continue
        resolved = os.path.abspath(os.path.join(now_dir, WAV_VOICE_DIR, filename))
        try:
            if os.path.commonpath([resolved, wav_dir_abs]) != wav_dir_abs or not os.path.isfile(resolved):
                continue
        except ValueError:
            continue
        try:
            info = sf.info(resolved)
            duration = info.duration
        except Exception as e:
            print(f"[ref_audio] 无法读取 {filename}: {e}")
            continue
        if duration > REF_AUDIO_DURATION_MAX_SEC:
            _trim_ref_audio_to_valid_duration(resolved, duration, voice_name=name.strip())
        elif duration < REF_AUDIO_DURATION_MIN_SEC:
            print(f"[ref_audio] 音色「{name.strip()}」{filename} 时长为 {duration:.2f}s，小于 {REF_AUDIO_DURATION_MIN_SEC}s，TTS 可能报错，请更换或延长参考音频")


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    # elif media_type == "ogg" and not streaming_mode:
    #     return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


def _generate_tts_bytes(req: dict):
    """
    Generate TTS audio bytes (non-streaming). req must already have ref_audio_path set and pass check_params.
    Returns (audio_bytes, media_type) or raises on error.
    """
    req_copy = dict(req)
    req_copy["streaming_mode"] = False
    req_copy["return_fragment"] = False
    req_copy["fixed_length_chunk"] = False
    media_type = req_copy.get("media_type", "wav")
    tts_generator = tts_pipeline.run(req_copy)
    sr, audio_data = next(tts_generator)
    audio_bytes = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
    return audio_bytes, media_type


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 15,                  # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "parallel_infer": True,       # bool. whether to use parallel inference.
                "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
                "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
                "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
                "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
                "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    # Resolve ref_audio_filename to ref_audio_path (voice files under wav/)
    resolve_res = resolve_ref_audio_path(req)
    if resolve_res is not None:
        return resolve_res

    check_res = check_params(req)
    if check_res is not None:
        return check_res
    
    if streaming_mode == 0:
        streaming_mode = False
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 1:
        streaming_mode = False
        return_fragment = True
        fixed_length_chunk = False
    elif streaming_mode == 2:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 3:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = True

    else:
        return JSONResponse(status_code=400, content={"message": f"the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)"})

    req["streaming_mode"] = streaming_mode
    req["return_fragment"] = return_fragment
    req["fixed_length_chunk"] = fixed_length_chunk

    print(f"{streaming_mode} {return_fragment} {fixed_length_chunk}")

    streaming_mode = streaming_mode or return_fragment


    try:
        if streaming_mode:
            tts_generator = tts_pipeline.run(req)

            def streaming_generator(tts_generator: Generator, media_type: str):
                if_frist_chunk = True
                for sr, chunk in tts_generator:
                    if if_frist_chunk and media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        media_type = "raw"
                        if_frist_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            return StreamingResponse(
                streaming_generator(
                    tts_generator,
                    media_type,
                ),
                media_type=f"audio/{media_type}",
            )

        else:
            audio_bytes, media_type = _generate_tts_bytes(req)
            filename = f"tts.{media_type}" if media_type in ("wav", "ogg", "aac") else "tts.bin"
            return Response(
                audio_bytes,
                media_type=f"audio/{media_type}",
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control", summary="Control", include_in_schema=False)
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get(
    "/tts",
    summary="Text to speech",
    description="Synthesize speech from text. Use ref_audio_filename (e.g. seed1.wav) for voice files under wav/, or ref_audio_path for full path.",
    include_in_schema=False,
)
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    ref_audio_filename: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
    streaming_mode: Union[bool, int] = False,
    overlap_length: int = 2,
    min_chunk_length: int = 16,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower() if text_lang else None,
        "ref_audio_path": ref_audio_path,
        "ref_audio_filename": ref_audio_filename,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower() if prompt_lang else None,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
        "overlap_length": int(overlap_length),
        "min_chunk_length": int(min_chunk_length),
    }
    return await tts_handle(req)


@APP.post("/tts", summary="Text to speech (POST)", include_in_schema=False)
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get(
    "/v1/voices",
    summary="List available voice names",
    description="Returns all currently available voice names (from wav/voices.json) for which the reference wav file exists. Use these names in /v1/tts as the voice parameter.",
)
async def v1_voices_list():
    names, _ = get_available_voice_names()
    return JSONResponse(status_code=200, content={"voices": names})


class V1TTSRequest(BaseModel):
    """Simple TTS: text + voice name only. Voice name is looked up in wav/voices.json."""
    text: str
    voice: str  # voice name (key in wav/voices.json), not filename
    text_lang: str = "zh"
    media_type: str = "wav"


async def v1_tts_handle(text: str, voice: str, text_lang: str = "zh", media_type: str = "wav", base_url: str = None):
    """Resolve voice name, generate audio, save to file, return JSON with url. Link expires in 10 min."""
    filename, err = resolve_voice_name_to_filename(voice)
    if err is not None:
        return err
    prompt_text, prompt_lang = load_prompt_for_ref_audio(filename, default_prompt_lang=text_lang)
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_filename": filename,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "media_type": media_type,
        "streaming_mode": False,
        "text_split_method": "cut5",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": 2,
        "top_k": 8,
        "top_p": 0.9,
        "temperature": 0.6,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False,
        "overlap_length": 2,
        "min_chunk_length": 16,
    }
    resolve_res = resolve_ref_audio_path(req)
    if resolve_res is not None:
        return resolve_res
    check_res = check_params(req)
    if check_res is not None:
        return check_res
    try:
        audio_bytes, media_type = _generate_tts_bytes(req)
    except Exception as e:
        err_msg = str(e)
        if "prompt_text" in err_msg.lower() or "prompt" in err_msg.lower():
            return JSONResponse(
                status_code=400,
                content={
                    "message": "TTS requires prompt_text for this model. Add wav/<basename>.txt and wav/<basename>.lang (or .json) for the voice.",
                    "detail": err_msg,
                },
            )
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": err_msg})
    filename = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.join(TTS_AUDIO_DIR, filename)
    try:
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
    except OSError as e:
        return JSONResponse(status_code=500, content={"message": "Failed to save audio file", "detail": str(e)})
    short_id = _generate_short_id()
    with _short_id_lock:
        while short_id in _short_id_to_file:
            short_id = _generate_short_id()
        _short_id_to_file[short_id] = filename
        _file_to_short_id[filename] = short_id
    if base_url is None:
        base_url = f"http://{host}:{port}"
    url = f"{base_url.rstrip('/')}/audio/{short_id}"
    return JSONResponse(
        status_code=200,
        content={
            "url": url,
            "expires_in": TTS_AUDIO_EXPIRE_SECONDS,
            "message": "Audio available at url; link expires in 10 minutes.",
        },
    )


@APP.post(
    "/v1/tts",
    summary="Simple TTS (text + voice name)",
    description="Returns JSON with url to the generated audio. Client fetches audio from that url. Link expires in 10 minutes. Body: {\"text\": \"...\", \"voice\": \"xiaofeng\"}. Optional: text_lang, media_type.",
)
async def v1_tts_post(request: V1TTSRequest, http_request: Request):
    base_url = str(http_request.base_url).rstrip("/")
    return await v1_tts_handle(
        text=request.text,
        voice=request.voice,
        text_lang=request.text_lang or "zh",
        media_type=request.media_type or "wav",
        base_url=base_url,
    )


def _v1_streaming_generator(tts_generator: Generator, media_type: str):
    """Yield WAV header then raw PCM chunks for streaming response."""
    if_frist_chunk = True
    for sr, chunk in tts_generator:
        if if_frist_chunk and media_type == "wav":
            yield wave_header_chunk(sample_rate=sr)
            media_type = "raw"
            if_frist_chunk = False
        yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()


def _get_v1_stream_req(voice: str, text: str, text_lang: str = "zh", media_type: str = "wav"):
    """
    Build and validate req for v1 stream TTS. Shared by HTTP stream and WebSocket.
    Returns (req, None) on success, or (None, error_message_str) on error.
    """
    filename, err = resolve_voice_name_to_filename(voice)
    if err is not None:
        try:
            body = err.body
            msg = json.loads(body.decode("utf-8")).get("message", "invalid voice")
        except Exception:
            msg = "invalid voice"
        return None, msg
    prompt_text, prompt_lang = load_prompt_for_ref_audio(filename, default_prompt_lang=text_lang)
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_filename": filename,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "media_type": media_type,
        "streaming_mode": 2,
        "return_fragment": False,
        "fixed_length_chunk": False,
        "text_split_method": "cut5",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": False,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": 2,
        "top_k": 10,
        "top_p": 0.9,
        "temperature": 0.6,
        "parallel_infer": False,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False,
        "overlap_length": 2,
        "min_chunk_length": 16,
    }
    resolve_res = resolve_ref_audio_path(req)
    if resolve_res is not None:
        try:
            msg = json.loads(resolve_res.body.decode("utf-8")).get("message", "ref_audio error")
        except Exception:
            msg = "ref_audio error"
        return None, msg
    check_res = check_params(req)
    if check_res is not None:
        try:
            msg = json.loads(check_res.body.decode("utf-8")).get("message", "params error")
        except Exception:
            msg = "params error"
        return None, msg
    return req, None


async def v1_tts_stream_handle(text: str, voice: str, text_lang: str = "zh", media_type: str = "wav"):
    """Stream TTS audio chunk by chunk. Uses existing streaming_mode=2 (semantic chunk streaming)."""
    req, err_msg = _get_v1_stream_req(voice, text, text_lang, media_type)
    if err_msg is not None:
        return JSONResponse(status_code=400, content={"message": err_msg})
    try:
        tts_generator = tts_pipeline.run(req)
        return StreamingResponse(
            _v1_streaming_generator(tts_generator, req["media_type"]),
            media_type=f"audio/{req['media_type']}",
        )
    except Exception as e:
        err_msg = str(e)
        if "prompt_text" in err_msg.lower() or "prompt" in err_msg.lower():
            return JSONResponse(
                status_code=400,
                content={
                    "message": "TTS requires prompt_text for this model. Add wav/<basename>.txt and wav/<basename>.lang (or .json) for the voice.",
                    "detail": err_msg,
                },
            )
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": err_msg})


@APP.post(
    "/v1/tts/stream",
    summary="Stream TTS (text + voice name)",
    description="Stream audio chunk by chunk. Same body as POST /v1/tts (text, voice). Optional: text_lang, media_type. Returns audio/wav stream (first chunk is WAV header, then raw PCM). Use for long text to get lower first-byte latency.",
)
async def v1_tts_stream_post(request: V1TTSRequest):
    return await v1_tts_stream_handle(
        text=request.text,
        voice=request.voice,
        text_lang=request.text_lang or "zh",
        media_type=request.media_type or "wav",
    )


@APP.websocket("/ws/v1/tts/stream")
async def ws_v1_tts_stream(websocket: WebSocket):
    """
    WebSocket stream TTS. One connection supports multiple syntheses in sequence.
    Send JSON: {"text": "...", "voice": "...", "text_lang": "zh"} to start synthesis.
    Server sends: binary frames (first = 44-byte WAV header, then PCM chunks), then {"event": "SynthesisCompleted"}.
    Send {"event": "close"} or close connection to end. On error, server sends {"event": "error", "message": "..."}.
    """
    await websocket.accept()
    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except Exception:
                break
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"event": "error", "message": "invalid JSON"}))
                continue
            if data.get("event") == "close":
                break
            text = data.get("text") or ""
            voice = (data.get("voice") or "").strip()
            text_lang = data.get("text_lang") or "zh"
            media_type = data.get("media_type") or "wav"
            req, err_msg = _get_v1_stream_req(voice, text, text_lang, media_type)
            if err_msg is not None:
                await websocket.send_text(json.dumps({"event": "error", "message": err_msg}))
                continue
            try:
                tts_generator = tts_pipeline.run(req)
                media_type = req["media_type"]
                cancelled = False
                for chunk in _v1_streaming_generator(tts_generator, media_type):
                    await websocket.send_bytes(chunk)
                    try:
                        msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.02)
                        data = json.loads(msg) if msg else {}
                        if data.get("event") in ("stop", "close"):
                            cancelled = True
                            break
                    except asyncio.TimeoutError:
                        pass
                    except (json.JSONDecodeError, Exception):
                        pass
                if not cancelled:
                    await websocket.send_text(json.dumps({"event": "SynthesisCompleted"}))
            except Exception as e:
                err_msg = str(e)
                await websocket.send_text(json.dumps({"event": "error", "message": err_msg}))
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass


# Short id: URL-safe alphanumeric (e.g. 10 chars). Not the raw filename.
_SHORT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{6,32}$")


@APP.get(
    "/audio/{file_id}",
    summary="Get TTS audio file (short link)",
    description="Download audio by short id returned from POST /v1/tts. URL is like /audio/a1b2c3d4e5, not the raw filename. Returns 404 if not found or expired (10 minutes).",
)
async def get_audio_file(file_id: str):
    if not _SHORT_ID_PATTERN.match(file_id):
        return JSONResponse(status_code=404, content={"message": "Not found"})
    with _short_id_lock:
        filename = _short_id_to_file.get(file_id)
    if not filename:
        return JSONResponse(status_code=404, content={"message": "File not found or expired"})
    path = os.path.join(TTS_AUDIO_DIR, filename)
    if not os.path.isfile(path):
        _unregister_tts_file(filename)
        return JSONResponse(status_code=404, content={"message": "File not found or expired"})
    if (time.time() - os.path.getmtime(path)) > TTS_AUDIO_EXPIRE_SECONDS:
        _unregister_tts_file(filename)
        try:
            os.remove(path)
        except OSError:
            pass
        return JSONResponse(status_code=404, content={"message": "File expired"})
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@APP.get("/set_refer_audio", summary="Set default reference audio", include_in_schema=False)
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights", summary="Switch GPT model weights", include_in_schema=False)
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights", summary="Switch SoVITS model weights", include_in_schema=False)
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# 网页端：挂载 web 目录，GET / 返回 index.html，API 路由优先
_web_dir = os.path.join(now_dir, "web")
if os.path.isdir(_web_dir):
    APP.mount("/", StaticFiles(directory=_web_dir, html=True), name="web")


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
