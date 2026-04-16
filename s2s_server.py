import asyncio
import os
import json
import logging
import io
import wave
import struct
import time
import base64
from pathlib import Path
import aiohttp
import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("S2S-Orchestrator")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import HTMLResponse

APP_ROOT = Path(__file__).resolve().parent
FRONTEND_PATH = Path(os.environ.get("S2S_FRONTEND_PATH", APP_ROOT / "s2s_frontend" / "index.html"))

@app.get("/")
@app.head("/")
async def get():
    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# Configuration
CANARY_URL = os.environ.get("CANARY_URL", "http://127.0.0.1:8001/transcribe")
HIGGS_URL = os.environ.get("HIGGS_URL", "http://127.0.0.1:8000/generate_stream")
VOXTRAL_URL = os.environ.get("VOXTRAL_URL", "http://127.0.0.1:8002/generate_stream")
# TTS_BACKEND: "higgs" (raw int16 PCM @ 24kHz) or "voxtral" (WAV @ 24kHz)
TTS_BACKEND = os.environ.get("TTS_BACKEND", "higgs").lower()

def tts_url() -> str:
    return VOXTRAL_URL if TTS_BACKEND == "voxtral" else HIGGS_URL

# /generate_stream on higgs:v5+ yields raw int16 PCM @ 24kHz as-generated (first chunk ~270ms)
# Groq Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

# RunPod API key — set this when using RunPod load-balancing endpoints.
# Leave unset for local dev (no auth header will be added).
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

def _runpod_headers() -> dict:
    """Return Authorization header for RunPod endpoints, empty dict for local."""
    if RUNPOD_API_KEY:
        return {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    return {}


SAMPLE_RATE = 16000

# WebRTC VAD config
# Aggressiveness: 0 (least) to 3 (most aggressive at filtering non-speech)
VAD_AGGRESSIVENESS = 3
# Frame size: WebRTC VAD requires 10/20/30ms frames. 30ms = 960 bytes @ 16kHz int16
VAD_FRAME_MS = 30
VAD_FRAME_BYTES = SAMPLE_RATE * 2 * VAD_FRAME_MS // 1000  # 960 bytes
# Speech/silence thresholds (number of 30ms frames)
# Need 3+ speech frames out of last 8 to trigger speech start (~90ms min)
VAD_SPEECH_FRAMES_THRESHOLD = 3
VAD_RING_SIZE = 8
# Silence duration after last speech frame to trigger processing
SILENCE_DURATION = 0.5
MIN_AUDIO_DURATION = 0.5

# TTS output normalization: emit fixed-size chunks for smooth playback
# 4800 bytes = 2400 samples @ 24kHz int16 mono = 100ms per chunk
TTS_CHUNK_SIZE = 4800

class ConnectionManager:
    def __init__(self):
        # Map websocket to its own Lock
        self.locks: dict[WebSocket, asyncio.Lock] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.locks[websocket] = asyncio.Lock()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.locks:
            del self.locks[websocket]

    def get_lock(self, websocket: WebSocket):
        return self.locks.get(websocket)

class ConversationManager:
    """Manages conversation history for context awareness"""
    def __init__(self, max_turns=10):
        self.history: dict[WebSocket, list] = {}
        self.max_turns = max_turns

    def add_message(self, websocket: WebSocket, role: str, content: str):
        if websocket not in self.history:
            self.history[websocket] = []
        
        # Append message
        self.history[websocket].append({"role": role, "content": content})
        
        # Trim history if needed (keep system prompt separate in actual call)
        if len(self.history[websocket]) > self.max_turns:
            self.history[websocket] = self.history[websocket][-self.max_turns:]

    def get_history(self, websocket: WebSocket):
        return self.history.get(websocket, [])

    def clear(self, websocket: WebSocket):
        if websocket in self.history:
            del self.history[websocket]

manager = ConnectionManager()
conversation_manager = ConversationManager(max_turns=6) # Keep last 3 exchanges (User+AI)
_global_session = None

async def get_session():
    global _global_session
    if _global_session is None or _global_session.closed:
        _global_session = aiohttp.ClientSession()
    return _global_session

@app.on_event("startup")
async def startup_warmup():
    pass  # Voxtral worker is always-on — no warmup pings needed

def create_wav_buffer(audio_data: bytes) -> io.BytesIO:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)
    buffer.seek(0)
    return buffer

async def safe_send_text(websocket: WebSocket, msg_dict: dict):
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            lock = manager.get_lock(websocket)
            if lock:
                async with lock:
                    await websocket.send_text(json.dumps(msg_dict))
    except Exception as e:
        logger.error(f"Error sending text: {e}")

async def safe_send_bytes(websocket: WebSocket, data: bytes):
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            lock = manager.get_lock(websocket)
            if lock:
                async with lock:
                    await websocket.send_bytes(data)
    except Exception as e:
        logger.error(f"Error sending bytes: {e}")


async def call_groq_via_manager(session, text, websocket, session_state):
    """Wrapper to handle history injection"""
    # Add user message to memory
    conversation_manager.add_message(websocket, "user", content=text)
    
    current_voice = "Cloned User Voice" if session_state.get("voice_mode") == "user" else "System Voice"
    
    messages = [
        {"role": "system", "content": (
            "You are a state-of-the-art Real-time Speech-to-Speech (S2S) AI assistant. "
            "You can hear the user and respond with a synthetic voice. "
            "CRITICAL: Respond ONLY in the same language the user used. "
            "If the user spoke English, you MUST respond in English. "
            "If the user spoke Dutch, you MUST respond in Dutch. "
            "Do not mix languages. Provide short, natural-sounding responses under 50 words. "
            "Be warm, upbeat and expressive — use natural energy and enthusiasm in your wording. "
            "Use short punchy sentences. End sentences with '!' when excited or '.' otherwise. "
            "Use natural filler words to sound human: 'Oh!', 'Hmm.', 'Well,', 'Right!', 'Oh wow!', 'Aww,' — sprinkle them in naturally, don't overdo it. "
            "NEVER use asterisks, actions, or written laughter like 'Haha'/'Hehe' — the TTS reads them literally and it sounds robotic.\n\n"
            f"Current Voice Mode: {current_voice}.\n"
            "You HAVE the capability to clone the user's voice. "
            "If the user asks to 'clone my voice' or 'speak like me':\n"
            "1. If you have learned their voice, confirm you are switching.\n"
            "2. If you haven't heard enough yet, explain that you need them to speak a bit more (at least 3-4 words) so you can learn their voice pattern before you can clone it.\n"
            "If the user asks to 'reset voice' or 'stop copying', confirm you are switching back to the system voice."
        )}
    ]
    
    # Inject history
    messages.extend(conversation_manager.get_history(websocket))
    
    # We don't verify the last message is the current one because we just added it.
    # But wait, 'text' is the current user input.
    # If we added it to history, it's already in the list.
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 150,
        "stream": True
    }
    try:
        async with session.post(GROQ_URL, headers=headers, json=payload, timeout=15) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Groq stream failed: {error}")
                yield f"Error {resp.status}"
                return
            
            full_response = ""
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        chunk = data["choices"][0]["delta"].get("content", "")
                        if chunk:
                            full_response += chunk
                            yield chunk
                    except Exception as e:
                        logger.error(f"Error parsing Groq chunk: {e}")
            
            # Add assistant response to memory after stream completes
            if full_response.strip():
                conversation_manager.add_message(websocket, "assistant", content=full_response.strip())
                
    except Exception as e:
        logger.error(f"Groq Exception: {e}")
        yield "Thinking error."

async def split_into_sentences(text_stream):
    """Yield phrases from LLM stream, optimized for minimum TTS latency.
    
    Strategy:
    - Strong punctuation (.!?) splits immediately at 3+ chars → catches 'Hi!', 'Yes.'
    - Weak punctuation (,;:) splits at 30+ chars → avoids 'Hello,' as a lone chunk
    - Force-flush at 80 chars (split at last space) → prevents huge run-on chunks
    """
    buffer = ""
    strong = {'.', '!', '?', '\n'}
    weak = {',', ';', ':'}

    def emit(s):
        s = s.strip()
        # Skip lone punctuation / single junk chars
        if len(s) < 2 or all(c in ".,!?;:'\"" for c in s):
            return None
        return s

    async for chunk in text_stream:
        for char in chunk:
            buffer += char
            stripped = buffer.strip()
            if char in strong and len(stripped) >= 3:
                if emit(stripped):
                    yield emit(stripped)
                buffer = ""
            elif char in weak and len(stripped) >= 30:
                if emit(stripped):
                    yield emit(stripped)
                buffer = ""
            elif len(stripped) >= 80:
                # Force-flush at word boundary to avoid cutting mid-word
                last_space = stripped.rfind(' ')
                if last_space > 20:
                    if emit(stripped[:last_space]):
                        yield emit(stripped[:last_space])
                    buffer = stripped[last_space:]

    if buffer.strip():
        s = buffer.strip()
        if len(s) >= 2 and not all(c in ".,!?;:'\"" for c in s):
            yield s

async def process_audio(audio_buffer: bytes, websocket: WebSocket, session_state: dict):
    """
    1. Send Audio to Canary (ASR)
    2. Save Audio for potentially cloning
    3. Send Text to Groq (LLM)
    4. Send Result to Higgs (TTS)
    5. Send Audio back to WebSocket
    """
    # Calculate duration
    duration = len(audio_buffer) / (SAMPLE_RATE * 2)
    if duration < MIN_AUDIO_DURATION:
        logger.info(f"Audio too short ({duration:.2f}s), ignoring.")
        return

    logger.info(f"Processing {len(audio_buffer)} bytes ({duration:.2f}s) of audio...")
    
    # Save user audio for cloning
    try:
        user_audio_path = f"/tmp/user_voice_{id(websocket)}.wav"
        with wave.open(user_audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_buffer)
        session_state["current_audio_chunk"] = user_audio_path
    except Exception as e:
        logger.error(f"Failed to save user audio: {e}")

    await safe_send_text(websocket, {"state": "processing", "message": "Transcribing..."})

    session = await get_session()
    # 1. ASR - Canary
    text = ""
    try:
        wav_buffer = create_wav_buffer(audio_buffer)
        data = aiohttp.FormData()
        data.add_field('file', wav_buffer, filename='input.wav', content_type='audio/wav')
        # Increase timeout to 30s for the larger v2 model's first few requests
        async with session.post(CANARY_URL, data=data, headers=_runpod_headers(), timeout=30) as resp:
            if resp.status != 200:
                await safe_send_text(websocket, {"state": "error", "message": f"ASR Error: {resp.status}"})
                return
            result = await resp.json()
        text = result.get("text", "").strip()
        score = result.get("score", 0.0)
        logger.info(f"Canary Transcript: '{text}' (Score: {score})")

        # Voice Command Logic
        text_lower = text.lower()
        # Broaden trigger list to handle common ASR mis-transcriptions
        clone_cmds = [
            "clone my voice", "speak like me", "copy my voice", 
            "switch to my voice", "speak in my voice", "learn my voice",
            "close my voice", "cloud my voice", "clone voice", "clown my voice"
        ]
        reset_cmds = ["reset voice", "normal voice", "stop copying", "your voice", "system voice"]
        is_clone_cmd = any(cmd in text_lower for cmd in clone_cmds)
        is_reset_cmd = any(cmd in text_lower for cmd in reset_cmds)
        is_command = is_clone_cmd or is_reset_cmd
        
        # Save the transcript for voice cloning reference
        if text and len(text.split()) >= 3 and not is_command:
            session_state["best_user_text"] = text
            if "current_audio_chunk" in session_state:
                session_state["best_user_audio"] = session_state["current_audio_chunk"]
                logger.info(f"Saved new voice reference: '{text}' at {session_state['best_user_audio']}")
        
        if is_clone_cmd:
            if "best_user_audio" in session_state:
                session_state["voice_mode"] = "user"
                logger.info("Switched to User Voice Mode")
                logger.info(f"Active Clone Reference: '{session_state.get('best_user_text')}'")
            else:
                logger.warning("User requested clone, but NO reference audio yet.")
                session_state["voice_mode"] = "system" 
        elif is_reset_cmd:
            session_state["voice_mode"] = "system"
            logger.info("Switching to System Voice Mode")
            
        # Confidence threshold check
        CONFIDENCE_THRESHOLD = -1.0 
            
        if score < CONFIDENCE_THRESHOLD and len(text.split()) > 2:
            logger.warning(f"Low ASR confidence ({score}). Triggering rephrase.")
            text = "I'm sorry, I didn't hear you clearly. Could you please repeat what you said?"
            await safe_send_text(websocket, {"state": "transcribed", "text": "[Low confidence] " + text})
            await safe_send_text(websocket, {"state": "processing", "message": "Asking to repeat..."})

            async with session.post(tts_url(), json={"text": text}, headers=_runpod_headers(), timeout=aiohttp.ClientTimeout(total=60)) as tts_resp:
                if tts_resp.status == 200:
                    residual = bytearray()
                    _wav_header_stripped = TTS_BACKEND != "voxtral"  # Higgs: raw PCM; Voxtral: WAV
                    async for raw in tts_resp.content.iter_any():
                        if not raw:
                            continue
                        if not _wav_header_stripped:
                            combined = bytes(residual) + raw
                            wav_end = combined.find(b"data")
                            if wav_end != -1:
                                raw = combined[wav_end + 8:]  # skip 'data' + 4-byte size
                                residual = bytearray()
                                _wav_header_stripped = True
                            else:
                                residual.extend(raw)
                                continue
                        residual.extend(raw)
                        while len(residual) >= TTS_CHUNK_SIZE:
                            await safe_send_bytes(websocket, bytes(residual[:TTS_CHUNK_SIZE]))
                            del residual[:TTS_CHUNK_SIZE]
                    if residual:
                        await safe_send_bytes(websocket, bytes(residual))

            await safe_send_text(websocket, {"state": "idle", "message": "Listening..."})
            return

        # Filter short hallucinations but allow clear short words if score is high
        valid_shorts = ["hi", "no", "yes", "hey", "ok", "bye"]
        if not text or (len(text) < 3 and text.lower() not in valid_shorts):
            await safe_send_text(websocket, {"state": "idle", "message": "No clear speech detected."})
            return
        
        # Additional score check for very short inputs to avoid noise-hallucinations
        if len(text.split()) == 1 and score < -0.5:
            await safe_send_text(websocket, {"state": "idle", "message": "Ignored low-confidence noise."})
            return

    except Exception as e:
        logger.error(f"ASR Exception: {e}")
        await safe_send_text(websocket, {"state": "error", "message": str(e)})
        return

    # 2 & 3. Streaming LLM → sentence splitting → streaming TTS → websocket
    # TTS backend: higgs (raw int16 PCM @ 24kHz) or voxtral (WAV @ 24kHz, header stripped).
    #
    # Architecture: producer/consumer with asyncio.Queue of streaming tasks.
    #   Producer:  for each sentence from LLM → immediately open TTS HTTP stream → enqueue the response
    #   Consumer:  drain chunks from each TTS stream in order → relay to websocket
    #
    # Sentence N+1's TTS synthesis starts the moment the LLM emits it (while N is still playing).
    # Gapless thanks to a shared residual buffer across sentence boundaries.
    await safe_send_text(websocket, {"state": "processing", "message": "Thinking..."})

    def build_tts_payload(sentence: str) -> dict:
        payload = {"text": sentence}
        if TTS_BACKEND == "voxtral":
            payload["voice"] = os.environ.get("VOXTRAL_VOICE", "casual_male")
        elif session_state.get("voice_mode") == "user" and session_state.get("best_user_audio"):
            try:
                with open(session_state["best_user_audio"], "rb") as af:
                    payload["ref_audio_base64"] = base64.b64encode(af.read()).decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to read ref audio: {e}")
            payload["temperature"] = 0.3
            if session_state.get("best_user_text"):
                payload["ref_text"] = session_state["best_user_text"]
        return payload

    # Each item in the queue is an aiohttp response context manager (already opened).
    # Using a queue of futures that resolve to async iterators.
    chunk_queue: asyncio.Queue = asyncio.Queue()  # items: asyncio.Queue[bytes] | None sentinel
    full_response = ""

    async def producer():
        """For each LLM sentence, open TTS stream and push a per-sentence byte queue."""
        nonlocal full_response

        async def _tts_call(text_batch, sq):
            """Send a single TTS call (possibly multi-sentence) and push PCM to sq.
            
            Higgs: streams raw int16 PCM — forward each chunk immediately.
            Voxtral: returns WAV — buffer, strip header, normalize, then push.
            """
            try:
                is_voxtral = TTS_BACKEND == "voxtral"
                wav_stripped = not is_voxtral  # Higgs: already raw PCM
                for attempt in range(3):
                    try:
                        async with session.post(
                            tts_url(), json=build_tts_payload(text_batch),
                            headers=_runpod_headers(),
                            timeout=aiohttp.ClientTimeout(total=90)
                        ) as resp:
                            if resp.status == 502 and attempt < 2:
                                logger.warning(f"TTS 502 (attempt {attempt+1}), retrying: {text_batch[:40]}")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            if resp.status != 200:
                                body = await resp.text()
                                logger.error(f"TTS {resp.status} for: {text_batch[:40]} | body: {body[:300]}")
                            else:
                                total_bytes = 0
                                if is_voxtral:
                                    # Voxtral: buffer entire WAV, strip header, normalize
                                    pcm_buf = bytearray()
                                    async for raw in resp.content.iter_any():
                                        if not raw:
                                            continue
                                        total_bytes += len(raw)
                                        if not wav_stripped:
                                            wav_end = raw.find(b"data")
                                            if wav_end != -1:
                                                raw = raw[wav_end + 8:]
                                                wav_stripped = True
                                            else:
                                                continue
                                        pcm_buf.extend(raw)
                                    if pcm_buf:
                                        import numpy as np
                                        samples = np.frombuffer(pcm_buf, dtype=np.int16).astype(np.float32)
                                        peak = np.abs(samples).max()
                                        if peak > 0:
                                            target = 28000.0
                                            samples = np.clip(samples * (target / peak), -32768, 32767)
                                            pcm_buf = samples.astype(np.int16).tobytes()
                                        await sq.put(bytes(pcm_buf))
                                    logger.info(f"TTS response: {total_bytes} total, {len(pcm_buf)} PCM for: {text_batch[:40]}")
                                else:
                                    # Higgs: stream raw PCM chunks directly for lowest latency
                                    streamed = 0
                                    async for raw in resp.content.iter_any():
                                        if not raw:
                                            continue
                                        total_bytes += len(raw)
                                        streamed += len(raw)
                                        await sq.put(raw)
                                    logger.info(f"TTS streamed: {total_bytes} total, {streamed} PCM for: {text_batch[:40]}")
                            break
                    except aiohttp.ClientError as e:
                        if attempt < 2:
                            logger.warning(f"TTS connection error (attempt {attempt+1}): {e}")
                            await asyncio.sleep(2 ** attempt)
                        else:
                            logger.error(f"TTS connection failed after retries: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"TTS stream error: {e}")
            finally:
                await sq.put(None)  # sentinel: batch done

        # For Voxtral: send the first sentence immediately for fast first-audio,
        # then batch subsequent sentences (~200 chars max) to reduce round-trips.
        VOXTRAL_BATCH_CHARS = 200
        pending_batch = []
        first_sent = True

        try:
            async for sentence in split_into_sentences(
                call_groq_via_manager(session, text, websocket, session_state)
            ):
                full_response += " " + sentence
                logger.info(f"Sentence → TTS: {sentence}")
                await safe_send_text(websocket, {"state": "thinking", "text": full_response.strip()})

                if TTS_BACKEND == "voxtral":
                    if first_sent:
                        # Send first sentence immediately for lowest latency
                        first_sent = False
                        sq = asyncio.Queue()
                        await chunk_queue.put(sq)
                        logger.info(f"TTS batch (first): {sentence}")
                        await _tts_call(sentence, sq)
                    else:
                        pending_batch.append(sentence)
                        batch_text = " ".join(pending_batch)
                        if len(batch_text) >= VOXTRAL_BATCH_CHARS:
                            sq = asyncio.Queue()
                            await chunk_queue.put(sq)
                            logger.info(f"TTS batch ({len(pending_batch)} sents, {len(batch_text)} chars): {batch_text[:60]}")
                            await _tts_call(batch_text, sq)
                            pending_batch = []
                else:
                    # Higgs: serialize calls — stream PCM chunks as they arrive
                    sq = asyncio.Queue()
                    await chunk_queue.put(sq)
                    await _tts_call(sentence, sq)

            # Flush remaining Voxtral batch
            if TTS_BACKEND == "voxtral" and pending_batch:
                batch_text = " ".join(pending_batch)
                sq = asyncio.Queue()
                await chunk_queue.put(sq)
                logger.info(f"TTS batch (final, {len(pending_batch)} sents, {len(batch_text)} chars): {batch_text[:60]}")
                await _tts_call(batch_text, sq)
        except asyncio.CancelledError:
            raise
        finally:
            await chunk_queue.put(None)  # sentinel: all sentences done

    async def consumer():
        """Drain per-sentence byte queues in order, relay fixed-size PCM chunks."""
        residual = bytearray()
        chunks_sent = 0
        total_pcm = 0
        session_state["tts_playing"] = True
        try:
            while True:
                sent_q = await chunk_queue.get()
                if sent_q is None:
                    break  # all done
                # Drain this sentence's byte stream
                while True:
                    raw = await sent_q.get()
                    if raw is None:
                        break  # sentence done
                    total_pcm += len(raw)
                    residual.extend(raw)
                    while len(residual) >= TTS_CHUNK_SIZE:
                        await safe_send_bytes(websocket, bytes(residual[:TTS_CHUNK_SIZE]))
                        del residual[:TTS_CHUNK_SIZE]
                        chunks_sent += 1
            # Flush any partial tail chunk
            if residual:
                await safe_send_bytes(websocket, bytes(residual))
                chunks_sent += 1
            logger.info(f"Consumer done: {total_pcm} PCM bytes, {chunks_sent} chunks sent to websocket")
        finally:
            # Grace period: keep tts_playing True briefly so VAD ignores echo tail
            try:
                await asyncio.sleep(0.4)
            except asyncio.CancelledError:
                pass
            session_state["tts_playing"] = False

    try:
        await asyncio.gather(producer(), consumer())
        await safe_send_text(websocket, {"state": "idle", "message": "Listening..."})

    except asyncio.CancelledError:
        logger.info("Process audio task cancelled (interrupted by user)")
        raise
    except Exception as e:
        logger.error(f"Stream/TTS loop exception: {e}")
        await safe_send_text(websocket, {"state": "error", "message": "Processing interrupted."})



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("Client connected")
    
    audio_buffer = bytearray()
    silence_start = None
    is_speaking = False
    current_task = None
    current_task_start = None  # time when current_task was created
    leftover = bytearray()  # leftover bytes not yet forming a full VAD frame
    
    # WebRTC VAD instance (per-connection)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    # Rolling ring buffer of recent frame results (True=speech, False=silence)
    ring = [False] * VAD_RING_SIZE
    ring_idx = 0
    
    # Session state for voice cloning
    session_state = {
        "voice_mode": "system",
        "last_user_audio": None,
    }
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except Exception as e:
                logger.error(f"WebSocket receive error: {e}")
                break
            
            if "bytes" in message:
                data = message["bytes"]
            elif "text" in message:
                try:
                    text_data = json.loads(message["text"])
                    if text_data.get("type") == "ping":
                        continue
                except:
                    pass
                continue
            else:
                if message.get("type") == "websocket.disconnect":
                    logger.info("WebSocket disconnect message received")
                    break
                continue
            
            # Accumulate bytes and process in 30ms frames for WebRTC VAD
            leftover.extend(data)
            
            while len(leftover) >= VAD_FRAME_BYTES:
                frame = bytes(leftover[:VAD_FRAME_BYTES])
                del leftover[:VAD_FRAME_BYTES]
                
                # While TTS is playing: hard-suppress all VAD, discard audio entirely.
                # This prevents ambient noise / speaker echo from interrupting the assistant.
                if session_state.get("tts_playing", False):
                    # Keep ring clear so there's no stale speech history when TTS ends
                    ring = [False] * VAD_RING_SIZE
                    ring_idx = 0
                    audio_buffer = bytearray()
                    is_speaking = False
                    silence_start = None
                    continue
                
                # Run WebRTC VAD on this 30ms frame
                try:
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    is_speech = False
                
                ring[ring_idx] = is_speech
                ring_idx = (ring_idx + 1) % VAD_RING_SIZE
                speech_count = sum(ring)
                
                if speech_count >= VAD_SPEECH_FRAMES_THRESHOLD:
                    # Speech detected
                    if not is_speaking:
                        is_speaking = True
                        task_age = (time.time() - current_task_start) if current_task_start else 999
                        logger.info(f"Speech detected (WebRTC VAD, count={speech_count}, task_age={task_age:.1f}s)")
                        if current_task and not current_task.done():
                            if task_age >= 3.0:
                                current_task.cancel()
                                current_task_start = None
                                logger.info("Interrupted current task")
                                await safe_send_text(websocket, {"state": "interrupted"})
                            else:
                                logger.info(f"Suppressing interrupt (task too young: {task_age:.1f}s < 3s)")
                                is_speaking = False
                                ring = [False] * VAD_RING_SIZE
                                ring_idx = 0
                                continue
                        await safe_send_text(websocket, {"state": "listening", "message": "Listening..."})
                    
                    silence_start = None
                    audio_buffer.extend(frame)
                    
                else:
                    if is_speaking:
                        audio_buffer.extend(frame)
                        
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_DURATION:
                            is_speaking = False
                            logger.info("Silence detected, processing buffer")
                            
                            if len(audio_buffer) > 0:
                                current_buffer = bytes(audio_buffer)
                                audio_buffer = bytearray()
                                current_task = asyncio.create_task(
                                    process_audio(current_buffer, websocket, session_state)
                                )
                                current_task_start = time.time()
                            
                            silence_start = None
                            # Reset ring to avoid re-triggering on stale frames
                            ring = [False] * VAD_RING_SIZE
                            ring_idx = 0

    except WebSocketDisconnect:
        logger.info("Client disconnected naturally")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
    finally:
        manager.disconnect(websocket)
        if current_task and not current_task.done():
            current_task.cancel()
        conversation_manager.clear(websocket)
        logger.info("Cleaned up WebSocket connection and history")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    return {"connected_clients": len(manager.locks)}
