"""
Live Translation WebSocket Server  (streaming edition)
=======================================================
Mic audio (Int16 PCM 16kHz) → VAD → chunked Canary AST every ~1.5 s (interim)
                                   → on silence, full utterance Canary AST (final)
                                   → browser live subtitles

Canary-1b-v2 performs direct speech-to-text translation (AST) for English ↔ 24
European languages in one model call — no Groq needed for those pairs.
Groq LLM is the fallback for pairs Canary can't handle.

Env vars (same .env as s2s_server):
  CANARY_URL        — Canary endpoint base
  RUNPOD_API_KEY    — RunPod bearer token
  GROQ_API_KEY      — Groq API key (fallback translation)
  HIGGS_URL         — TTS endpoint (optional; used when speak=true)
"""

import asyncio
import io
import json
import logging
import os
import struct
import wave
from pathlib import Path

import aiohttp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Translation")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_ROOT = Path(__file__).resolve().parent
FRONTEND_PATH = APP_ROOT / "translation_frontend" / "index.html"

_CANARY_BASE = os.environ.get("CANARY_URL", "http://127.0.0.1:8001/transcribe")
# Derive both /transcribe and /translate URLs from the base
_canary_origin = _CANARY_BASE.split("/transcribe")[0].split("/translate")[0].rstrip("/")
CANARY_TRANSCRIBE_URL = _canary_origin + "/transcribe"
CANARY_TRANSLATE_URL = _canary_origin + "/translate"

# Canary's change_decoding_strategy for s2t_translation returns the source
# verbatim (English) instead of translating — AST is broken in the deployed model.
# Always use ASR (Canary transcribes) + Groq (translates) for all pairs.
CANARY_AST_LANGS: set[str] = set()

HIGGS_URL = os.environ.get("HIGGS_URL", "http://127.0.0.1:8000/generate_stream")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SAMPLE_RATE = 16000
VAD_THRESHOLD = 1500
VAD_MIN_SPEECH_FRAMES = 2
SILENCE_DURATION = 0.30   # seconds of silence before firing final utterance
MIN_AUDIO_DURATION = 0.25
TTS_CHUNK_SIZE = 4800

# Streaming: send interim chunks to Canary every INTERIM_INTERVAL seconds
# while the user is still speaking, so partial translations appear live.
INTERIM_INTERVAL = 1.0  # seconds between interim AST calls

# Keep-warm: ping Canary this often to prevent RunPod cold starts
KEEP_WARM_INTERVAL = 25  # seconds

LANG_NAMES: dict[str, str] = {
    "auto": "the detected source language",
    "en": "English",    "nl": "Dutch",          "de": "German",
    "fr": "French",     "es": "Spanish",         "it": "Italian",
    "pt": "Portuguese", "pl": "Polish",          "ru": "Russian",
    "uk": "Ukrainian",  "sv": "Swedish",         "no": "Norwegian",
    "da": "Danish",     "fi": "Finnish",         "cs": "Czech",
    "sk": "Slovak",     "ro": "Romanian",        "hu": "Hungarian",
    "el": "Greek",      "tr": "Turkish",         "ar": "Arabic",
    "he": "Hebrew",     "hi": "Hindi",           "ja": "Japanese",
    "ko": "Korean",     "zh": "Chinese (Simplified)", "id": "Indonesian",
    "ms": "Malay",      "th": "Thai",            "vi": "Vietnamese",
}


def _runpod_headers() -> dict:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}"} if RUNPOD_API_KEY else {}


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf


async def _safe_send_text(ws: WebSocket, lock: asyncio.Lock, payload: dict):
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            async with lock:
                await ws.send_text(json.dumps(payload))
    except Exception as e:
        logger.debug(f"send_text error: {e}")


async def _safe_send_bytes(ws: WebSocket, lock: asyncio.Lock, data: bytes):
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            async with lock:
                await ws.send_bytes(data)
    except Exception as e:
        logger.debug(f"send_bytes error: {e}")


_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def _groq_translate(
    session: aiohttp.ClientSession,
    text: str,
    source_lang: str,
    target_lang: str,
) -> tuple[str, str]:
    """Returns (translated_text, detected_source_lang_code)."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — returning original as translation")
        return text, source_lang

    tgt_name = LANG_NAMES.get(target_lang, target_lang)

    if source_lang == "auto":
        system_prompt = (
            "You are a professional translator. "
            f"Detect the source language of the given text and translate it to {tgt_name}. "
            "Respond ONLY with a JSON object, no markdown fences:\n"
            '{"translation": "<translated text>", "detected_lang": "<ISO 639-1 code>"}'
        )
    else:
        src_name = LANG_NAMES.get(source_lang, source_lang)
        system_prompt = (
            "You are a professional translator. "
            f"Translate the following {src_name} text to {tgt_name}. "
            "Respond ONLY with a JSON object, no markdown fences:\n"
            '{"translation": "<translated text>", "detected_lang": "' + source_lang + '"}'
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    content = ""
    try:
        async with session.post(GROQ_URL, json=payload, headers=headers, timeout=20) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Groq {resp.status}: {body[:200]}")
                return text, source_lang
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1].lstrip("json").strip() if len(parts) > 1 else content
            parsed = json.loads(content)
            return parsed.get("translation", text), parsed.get("detected_lang", source_lang)
    except json.JSONDecodeError:
        logger.error(f"Groq non-JSON response: '{content}'")
        return content or text, source_lang
    except Exception as e:
        logger.error(f"Groq exception: {e}")
        return text, source_lang


async def _call_canary_ast(
    session: aiohttp.ClientSession,
    pcm: bytes,
    source_lang: str,
    target_lang: str,
) -> str | None:
    """Single Canary AST call.  Returns translated text or None on error."""
    wav_buf = _pcm_to_wav(pcm)
    form = aiohttp.FormData()
    form.add_field("file", wav_buf, filename="audio.wav", content_type="audio/wav")
    url = f"{CANARY_TRANSLATE_URL}?source_lang={source_lang}&target_lang={target_lang}"
    try:
        async with session.post(
            url, data=form, headers=_runpod_headers(),
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                logger.error(f"Canary AST {resp.status}: {(await resp.text())[:200]}")
                return None
            data = await resp.json()
            return (data.get("text") or "").strip() or None
    except Exception as e:
        logger.error(f"Canary AST ({type(e).__name__}): {e}")
        return None


async def _call_canary_asr(
    session: aiohttp.ClientSession,
    pcm: bytes,
) -> str | None:
    """Single Canary ASR (transcription) call.  Returns text or None."""
    wav_buf = _pcm_to_wav(pcm)
    form = aiohttp.FormData()
    form.add_field("file", wav_buf, filename="audio.wav", content_type="audio/wav")
    try:
        async with session.post(
            CANARY_TRANSCRIBE_URL, data=form, headers=_runpod_headers(),
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                logger.error(f"Canary ASR {resp.status}: {(await resp.text())[:200]}")
                return None
            data = await resp.json()
            return (data.get("text") or "").strip() or None
    except Exception as e:
        logger.error(f"Canary ASR ({type(e).__name__}): {e}")
        return None


def _can_use_ast(source_lang: str, target_lang: str) -> bool:
    return (
        source_lang != "auto"
        and source_lang in CANARY_AST_LANGS
        and target_lang in CANARY_AST_LANGS
        and source_lang != target_lang
        and (source_lang == "en" or target_lang == "en")
    )


async def _translate_utterance(
    pcm: bytes,
    source_lang: str,
    target_lang: str,
    ws: WebSocket,
    lock: asyncio.Lock,
    speak: bool,
    is_interim: bool = False,
) -> None:
    duration = len(pcm) / (SAMPLE_RATE * 2)
    if duration < MIN_AUDIO_DURATION:
        return

    tag = "interim" if is_interim else "final"
    logger.info(f"[{tag}] {duration:.2f}s  [{source_lang}→{target_lang}]")
    if not is_interim:
        await _safe_send_text(ws, lock, {"state": "processing"})

    session = await _get_session()
    use_ast = _can_use_ast(source_lang, target_lang)

    translation = None
    original = None

    if use_ast:
        translation = await _call_canary_ast(session, pcm, source_lang, target_lang)
        if translation:
            logger.info(f"[{tag}] AST [{source_lang}→{target_lang}]: '{translation}'")
    else:
        original = await _call_canary_asr(session, pcm)
        if original:
            logger.info(f"[{tag}] ASR: '{original}'")
            if not is_interim:
                await _safe_send_text(ws, lock, {"state": "translating"})
            translation, detected_lang = await _groq_translate(session, original, source_lang, target_lang)
            source_lang = detected_lang
            logger.info(f"[{tag}] [{detected_lang}→{target_lang}] '{translation}'")

    if not translation:
        if not is_interim:
            await _safe_send_text(ws, lock, {"state": "idle"})
        return

    await _safe_send_text(ws, lock, {
        "state": "interim" if is_interim else "result",
        "original": original or "",
        "translation": translation,
        "detected_lang": source_lang,
        "target_lang": target_lang,
    })

    # TTS only on final
    if not is_interim and speak and translation:
        try:
            await _safe_send_text(ws, lock, {"state": "tts_start"})
            async with session.post(
                HIGGS_URL,
                json={"text": translation},
                headers=_runpod_headers(),
                timeout=aiohttp.ClientTimeout(total=120, sock_read=120),
            ) as tts_resp:
                if tts_resp.status == 200:
                    residual = bytearray()
                    async for chunk in tts_resp.content.iter_any():
                        if not chunk:
                            continue
                        residual.extend(chunk)
                        while len(residual) >= TTS_CHUNK_SIZE:
                            await _safe_send_bytes(ws, lock, bytes(residual[:TTS_CHUNK_SIZE]))
                            del residual[:TTS_CHUNK_SIZE]
                    if residual:
                        await _safe_send_bytes(ws, lock, bytes(residual))
                    await _safe_send_text(ws, lock, {"state": "tts_end"})
        except Exception as e:
            logger.warning(f"TTS error: {e}")

    if not is_interim:
        await _safe_send_text(ws, lock, {"state": "idle"})


async def _keep_warm_loop():
    """Ping Canary periodically to prevent RunPod serverless cold starts."""
    ping_url = _canary_origin + "/ping"
    await asyncio.sleep(5)  # wait for server startup
    while True:
        try:
            session = await _get_session()
            async with session.get(
                ping_url, headers=_runpod_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                logger.debug(f"Keep-warm ping → {resp.status}")
        except Exception as e:
            logger.debug(f"Keep-warm ping failed: {e}")
        await asyncio.sleep(KEEP_WARM_INTERVAL)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_keep_warm_loop())


@app.get("/")
@app.head("/")
async def index():
    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    send_lock = asyncio.Lock()

    source_lang = "en"
    target_lang = "nl"
    speak = False

    audio_buffer = bytearray()
    silence_frames = 0
    speech_frames = 0
    is_speaking = False

    # Track when speech started to schedule interim chunks
    speech_start_time: float = 0.0
    last_interim_time: float = 0.0
    interim_task: asyncio.Task | None = None

    SILENCE_FRAMES_NEEDED = int(SILENCE_DURATION * SAMPLE_RATE / 4096)

    async def _fire_interim():
        """Send an interim chunk of the current audio buffer to Canary."""
        nonlocal last_interim_time
        if len(audio_buffer) < int(MIN_AUDIO_DURATION * SAMPLE_RATE * 2):
            return
        pcm_snapshot = bytes(audio_buffer)
        last_interim_time = asyncio.get_event_loop().time()
        try:
            await _translate_utterance(
                pcm_snapshot, source_lang, target_lang,
                ws, send_lock, speak=False, is_interim=True,
            )
        except Exception as e:
            logger.debug(f"Interim error: {e}")

    async def _interim_scheduler():
        """Background loop: fires interim AST calls every INTERIM_INTERVAL while speaking."""
        nonlocal last_interim_time
        while True:
            await asyncio.sleep(0.25)  # check 4x/sec
            if not is_speaking:
                continue
            now = asyncio.get_event_loop().time()
            if now - last_interim_time >= INTERIM_INTERVAL:
                await _fire_interim()

    interim_task = asyncio.create_task(_interim_scheduler())

    await _safe_send_text(ws, send_lock, {"state": "idle"})
    logger.info("Client connected")

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg:
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "config":
                        source_lang = ctrl.get("source_lang", source_lang)
                        target_lang = ctrl.get("target_lang", target_lang)
                        speak = bool(ctrl.get("speak", speak))
                        logger.info(f"Config: {source_lang}→{target_lang} speak={speak}")
                        await _safe_send_text(ws, send_lock, {
                            "state": "configured",
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "speak": speak,
                        })
                except Exception:
                    pass
                continue

            if "bytes" not in msg or not msg["bytes"]:
                continue

            raw = msg["bytes"]
            audio_buffer.extend(raw)

            n_samples = len(raw) // 2
            samples = struct.unpack(f"<{n_samples}h", raw[: n_samples * 2])
            energy = int(np.sqrt(np.mean(np.array(samples, dtype=np.float64) ** 2)))

            if energy >= VAD_THRESHOLD:
                speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1
                speech_frames = max(0, speech_frames - 1)

            if not is_speaking and speech_frames >= VAD_MIN_SPEECH_FRAMES:
                is_speaking = True
                now = asyncio.get_event_loop().time()
                speech_start_time = now
                last_interim_time = now
                await _safe_send_text(ws, send_lock, {"state": "listening"})

            if is_speaking and silence_frames >= SILENCE_FRAMES_NEEDED:
                is_speaking = False
                speech_frames = 0
                silence_frames = 0

                utterance_pcm = bytes(audio_buffer)
                audio_buffer.clear()

                # Fire as background task so receive loop continues immediately
                # (user can start the next utterance without waiting for translation)
                asyncio.create_task(_translate_utterance(
                    utterance_pcm, source_lang, target_lang,
                    ws, send_lock, speak,
                    is_interim=False,
                ))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WS error: {e}")
    finally:
        if interim_task and not interim_task.done():
            interim_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("translation_server:app", host="0.0.0.0", port=8083, reload=False)
