"""
Live Translation WebSocket Server  (continuous-parallel edition)
================================================================
Mic audio (Int16 PCM 16kHz) streams in continuously.
A background scheduler fires ASR + Groq translation every CHUNK_INTERVAL
seconds on all accumulated audio — no waiting for silence.  Multiple
translation calls fly in parallel; results display the instant they land.
Silence just resets the buffer so the next sentence starts fresh.

Env vars (same .env as s2s_server):
  CANARY_URL        — Canary endpoint base
  RUNPOD_API_KEY    — RunPod bearer token
  GROQ_API_KEY      — Groq API key (translation)
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
import webrtcvad
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
MIN_AUDIO_DURATION = 0.20
TTS_CHUNK_SIZE = 4800

# WebRTC VAD — aggressiveness 0 (least) to 3 (most aggressive at filtering noise)
VAD_AGGRESSIVENESS = 3   # Maximum: aggressively reject noise
VAD_FRAME_MS = 30        # 30ms = 480 samples = 960 bytes
VAD_FRAME_BYTES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # 960
VAD_SPEECH_TRIGGER = 3   # frames of speech  → "listening" state
VAD_SILENCE_TRIGGER = 20 # frames of silence → sentence boundary (~600ms @30ms)

# Fire a translation call every CHUNK_INTERVAL seconds on accumulated audio.
# Calls run concurrently so the next one doesn't wait for the previous to finish.
# --- Chunk triggering ---
# Fire ONLY when silence is detected (user finished a phrase) or when
# the buffer hits MAX_CHUNK_DURATION (user speaks non-stop).
# This ensures Canary always receives complete utterances, not random slices.
SENTENCE_SILENCE = 0.6    # seconds of silence → sentence boundary (fire!)
MAX_CHUNK_DURATION = 12.0 # hard cap: fire if buffer exceeds this (non-stop speech)
MIN_CHUNK_DURATION = 0.6  # don't fire chunks shorter than this

# --- Anti-hallucination gates ---
# Gate 1: VAD — at least 20% of frames must be speech, and ≥ 5 frames (~150ms)
MIN_SPEECH_RATIO = 0.20
MIN_SPEECH_FRAMES = 5
# Gate 2: RMS energy — PCM must exceed this to be real speech
MIN_RMS_ENERGY = 150
# Gate 3: Known Canary hallucination phrases from noise
HALLUCINATION_PHRASES = {
    "thank you.", "thank you", "thanks.", "thanks",
    "i'm sorry.", "i'm sorry", "sorry.",
    "you're welcome.", "you're welcome",
    "bye.", "bye-bye.", "bye-bye",
    "i'm sorry, thank you.", "okay.", "okay",
}

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
) -> tuple[str, str, str]:
    """Correct ASR errors, then translate. Returns (corrected_original, translation, detected_lang)."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — returning original as translation")
        return text, text, source_lang

    tgt_name = LANG_NAMES.get(target_lang, target_lang)

    if source_lang == "auto":
        system_prompt = (
            "You are a professional transcription corrector and translator. "
            "The input is raw ASR (speech-to-text) output which may contain errors, "
            "broken phrases, or garbled words caused by the speech recogniser. "
            "Step 1: Lightly correct the text — fix obvious ASR errors, grammar, and incomplete "
            "phrases while preserving the speaker's original meaning and words as closely as possible. "
            "Do NOT paraphrase or add content that wasn't said. "
            f"Step 2: Translate the corrected text into {tgt_name}. "
            "Detect the source language. "
            "Respond ONLY with a JSON object, no markdown fences:\n"
            '{"corrected": "<corrected source text>", "translation": "<translated text>", "detected_lang": "<ISO 639-1 code>"}'
        )
    else:
        src_name = LANG_NAMES.get(source_lang, source_lang)
        system_prompt = (
            "You are a professional transcription corrector and translator. "
            "The input is raw ASR (speech-to-text) output which may contain errors, "
            "broken phrases, or garbled words caused by the speech recogniser. "
            f"Step 1: Lightly correct the {src_name} text — fix obvious ASR errors, grammar, and "
            "incomplete phrases while preserving the speaker's original meaning and words as closely as possible. "
            "Do NOT paraphrase or add content that wasn't said. "
            f"Step 2: Translate the corrected text into {tgt_name}. "
            "Respond ONLY with a JSON object, no markdown fences:\n"
            '{"corrected": "<corrected source text>", "translation": "<translated text>", "detected_lang": "' + source_lang + '"}'
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
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
                return text, text, source_lang
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1].lstrip("json").strip() if len(parts) > 1 else content
            parsed = json.loads(content)
            corrected = parsed.get("corrected", text)
            return corrected, parsed.get("translation", text), parsed.get("detected_lang", source_lang)
    except json.JSONDecodeError:
        logger.error(f"Groq non-JSON response: '{content}'")
        return text, content or text, source_lang
    except Exception as e:
        logger.error(f"Groq exception: {e}")
        return text, text, source_lang


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
) -> tuple[str | None, float]:
    """Single Canary ASR call. Returns (text, score). Score is log-prob confidence."""
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
                return None, 0.0
            data = await resp.json()
            text = (data.get("text") or "").strip() or None
            score = float(data.get("score", 0.0))
            return text, score
    except Exception as e:
        logger.error(f"Canary ASR ({type(e).__name__}): {e}")
        return None, 0.0


def _can_use_ast(source_lang: str, target_lang: str) -> bool:
    return (
        source_lang != "auto"
        and source_lang in CANARY_AST_LANGS
        and target_lang in CANARY_AST_LANGS
        and source_lang != target_lang
        and (source_lang == "en" or target_lang == "en")
    )


def _compute_rms(pcm: bytes) -> float:
    """Compute RMS energy of 16-bit PCM audio."""
    if len(pcm) < 2:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _is_hallucination(text: str, duration_s: float) -> bool:
    """Return True if text looks like a Canary hallucination.

    Two cases:
    1. Exact match to a known noise phrase AND the audio was long enough
       that Canary should have produced more words (< 1 word/sec is suspicious).
    2. Repetition loop — any 4-word n-gram appears more than 3 times.
    """
    normalized = text.strip().lower()
    words = normalized.split()
    word_count = len(words)

    # Repetition loop detector (applies regardless of duration)
    if word_count >= 8:
        ngram_size = 4
        ngrams: dict[tuple, int] = {}
        for i in range(word_count - ngram_size + 1):
            ng = tuple(words[i:i + ngram_size])
            ngrams[ng] = ngrams.get(ng, 0) + 1
        if ngrams and max(ngrams.values()) > 3:
            return True

    # Exact-phrase filter: only block if words-per-second is suspiciously low.
    # If someone genuinely says "I'm sorry" it takes ~0.8s → ~2.5 w/s, fine.
    # If a 4s chunk produces "I'm sorry", that's only 0.5 w/s → hallucination.
    words_per_sec = word_count / duration_s if duration_s > 0 else 0
    if normalized in HALLUCINATION_PHRASES and words_per_sec < 1.2:
        return True

    return False


async def _transcribe_and_translate(
    pcm: bytes,
    source_lang: str,
    target_lang: str,
) -> tuple[str | None, str | None, str]:
    """ASR then Groq translate. Returns (original, translation, detected_lang).
    Applies multiple anti-hallucination gates."""
    duration = len(pcm) / (SAMPLE_RATE * 2)
    if duration < MIN_AUDIO_DURATION:
        return None, None, source_lang

    # Gate 2: RMS energy
    rms = _compute_rms(pcm)
    if rms < MIN_RMS_ENERGY:
        logger.info(f"Rejected: RMS {rms:.0f} < {MIN_RMS_ENERGY}")
        return None, None, source_lang

    session = await _get_session()

    if _can_use_ast(source_lang, target_lang):
        translation = await _call_canary_ast(session, pcm, source_lang, target_lang)
        return None, translation, source_lang

    original, score = await _call_canary_asr(session, pcm)
    if not original:
        logger.info(f"Canary returned empty ({duration:.1f}s, RMS={rms:.0f}) — skipping")
        return None, None, source_lang

    # Gate 3: known hallucination phrases
    if _is_hallucination(original, duration):
        logger.info(f"Rejected: hallucination — '{original}'")
        return None, None, source_lang

    logger.info(f"ASR: '{original}' ({duration:.1f}s, RMS={rms:.0f})")
    corrected, translation, detected = await _groq_translate(session, original, source_lang, target_lang)
    return corrected, translation, detected


async def _stream_tts(
    translation: str,
    ws: WebSocket,
    lock: asyncio.Lock,
    cancel: asyncio.Event,
):
    """Stream TTS audio, aborting early if cancel is set.
    Higgs returns a WAV file — strip the 44-byte header so the browser
    receives raw 16-bit PCM which playPCM() expects.
    """
    WAV_HEADER_BYTES = 44
    session = await _get_session()
    try:
        await _safe_send_text(ws, lock, {"state": "tts_start"})
        async with session.post(
            HIGGS_URL,
            json={"text": translation},
            headers=_runpod_headers(),
            timeout=aiohttp.ClientTimeout(total=120, sock_read=120),
        ) as tts_resp:
            if tts_resp.status != 200:
                body = await tts_resp.text()
                logger.warning(f"TTS {tts_resp.status}: {body[:120]}")
                return
            residual = bytearray()
            header_skipped = 0  # how many header bytes we've consumed so far
            async for chunk in tts_resp.content.iter_any():
                if cancel.is_set():
                    return
                if not chunk:
                    continue
                # Skip WAV header bytes from the very first chunk(s)
                if header_skipped < WAV_HEADER_BYTES:
                    need = WAV_HEADER_BYTES - header_skipped
                    if len(chunk) <= need:
                        header_skipped += len(chunk)
                        continue
                    chunk = chunk[need:]
                    header_skipped = WAV_HEADER_BYTES
                residual.extend(chunk)
                while len(residual) >= TTS_CHUNK_SIZE:
                    await _safe_send_bytes(ws, lock, bytes(residual[:TTS_CHUNK_SIZE]))
                    del residual[:TTS_CHUNK_SIZE]
            if residual and not cancel.is_set():
                await _safe_send_bytes(ws, lock, bytes(residual))
        await _safe_send_text(ws, lock, {"state": "tts_end"})
        logger.info(f"TTS done: '{translation[:40]}'")
    except Exception as e:
        logger.warning(f"TTS error: {e}")


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


# Global WebRTC VAD instance (thread-safe for reads after init)
_vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


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
    _vad_buf = bytearray()   # accumulates raw bytes until we have a full VAD frame
    silence_frames = 0
    speech_frames = 0
    chunk_speech_frames = 0  # speech frames since last chunk fire
    chunk_total_frames = 0   # total VAD frames since last chunk fire
    has_speech = False  # any speech detected in current buffer

    # Monotonic sequence number — lets us discard stale results
    _seq = 0
    _last_shown_seq = 0
    _last_chunk_time: float = 0.0  # when we last fired a chunk

    # FIFO TTS queue: (seq, translation_text).  A background consumer
    # plays them strictly in order so chunk #1 always speaks before #2.
    _tts_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

    background_tasks: set[asyncio.Task] = set()

    async def _tts_consumer():
        """Drain the TTS queue in strict FIFO order."""
        cancel = asyncio.Event()  # unused placeholder — no cancellation in FIFO mode
        while True:
            seq, text = await _tts_queue.get()
            try:
                logger.debug(f"TTS FIFO: playing chunk#{seq}")
                await _stream_tts(text, ws, send_lock, cancel)
            except Exception as e:
                logger.warning(f"TTS FIFO error on chunk#{seq}: {e}")
            finally:
                _tts_queue.task_done()

    async def _fire_chunk(pcm: bytes, seq: int, sl: str, tl: str, do_speak: bool):
        """Run ASR+translate in background. Send result if not stale."""
        nonlocal _last_shown_seq
        dur = len(pcm) / (SAMPLE_RATE * 2)
        rms = _compute_rms(pcm)
        logger.info(f"[chunk#{seq}] {dur:.2f}s RMS={rms:.0f} [{sl}→{tl}]")

        original, translation, detected = await _transcribe_and_translate(pcm, sl, tl)

        if not translation:
            return

        # Only show if this result is newer than the last shown
        if seq <= _last_shown_seq:
            logger.debug(f"[chunk#{seq}] stale, skipping (last shown={_last_shown_seq})")
            return
        _last_shown_seq = seq

        logger.info(f"[chunk#{seq}] [{detected}→{tl}] '{translation}'")

        await _safe_send_text(ws, send_lock, {
            "state": "result",
            "original": original or "",
            "translation": translation,
            "detected_lang": detected,
            "target_lang": tl,
            "seq": seq,
        })

        # Enqueue for FIFO TTS playback (consumer will play in order)
        if do_speak and translation:
            await _tts_queue.put((seq, translation))

    async def _chunk_scheduler():
        """Fire a chunk when the user pauses (silence) or buffer hits max duration.
        This ensures Canary receives complete utterances, not arbitrary slices."""
        nonlocal _seq, _last_chunk_time, has_speech, silence_frames
        nonlocal speech_frames, chunk_speech_frames, chunk_total_frames
        while True:
            await asyncio.sleep(0.10)
            buf_duration = len(audio_buffer) / (SAMPLE_RATE * 2)

            if buf_duration < MIN_CHUNK_DURATION:
                continue

            silence_secs = silence_frames * VAD_FRAME_MS / 1000
            is_pause = silence_secs >= SENTENCE_SILENCE and has_speech
            is_max = buf_duration >= MAX_CHUNK_DURATION

            if not is_pause and not is_max:
                continue

            # Gate 1: VAD speech ratio + absolute frame count
            speech_ratio = chunk_speech_frames / chunk_total_frames if chunk_total_frames > 0 else 0
            if speech_ratio < MIN_SPEECH_RATIO or chunk_speech_frames < MIN_SPEECH_FRAMES:
                logger.debug(f"Skip: ratio={speech_ratio:.2f} frames={chunk_speech_frames}")
                audio_buffer.clear()
                silence_frames = 0
                speech_frames = 0
                has_speech = False
                chunk_speech_frames = 0
                chunk_total_frames = 0
                continue

            # Trim trailing silence from the PCM so Canary gets clean audio
            trailing_silence_bytes = int(silence_frames * VAD_FRAME_BYTES)
            if trailing_silence_bytes > 0 and trailing_silence_bytes < len(audio_buffer):
                pcm_snapshot = bytes(audio_buffer[:-trailing_silence_bytes])
            else:
                pcm_snapshot = bytes(audio_buffer)

            _seq += 1
            seq = _seq
            now = asyncio.get_event_loop().time()

            audio_buffer.clear()
            silence_frames = 0
            speech_frames = 0
            has_speech = False
            chunk_speech_frames = 0
            chunk_total_frames = 0
            _last_chunk_time = now

            task = asyncio.create_task(
                _fire_chunk(pcm_snapshot, seq, source_lang, target_lang, speak)
            )
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

    scheduler = asyncio.create_task(_chunk_scheduler())
    tts_task = asyncio.create_task(_tts_consumer())
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
            _vad_buf.extend(raw)

            # Process complete 30ms WebRTC VAD frames
            while len(_vad_buf) >= VAD_FRAME_BYTES:
                frame = bytes(_vad_buf[:VAD_FRAME_BYTES])
                del _vad_buf[:VAD_FRAME_BYTES]
                try:
                    is_speech = _vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if is_speech:
                    speech_frames += 1
                    silence_frames = 0
                    has_speech = True
                    chunk_speech_frames += 1
                else:
                    silence_frames += 1
                    speech_frames = max(0, speech_frames - 1)
                chunk_total_frames += 1

            # Let the frontend know we're listening when speech detected
            if speech_frames == VAD_SPEECH_TRIGGER:
                await _safe_send_text(ws, send_lock, {"state": "listening"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WS error: {e}")
    finally:
        scheduler.cancel()
        tts_task.cancel()
        for t in background_tasks:
            t.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("translation_server:app", host="0.0.0.0", port=8083, reload=False)
