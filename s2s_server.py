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
# Groq Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

# Vast.ai Serverless configuration (optional — set these to route ASR/TTS through Vast)
VAST_API_KEY = os.environ.get("VAST_API_KEY")
VAST_CANARY_ENDPOINT = os.environ.get("VAST_CANARY_ENDPOINT")  # e.g. "canary-asr"
VAST_HIGGS_ENDPOINT = os.environ.get("VAST_HIGGS_ENDPOINT")    # e.g. "higgs-tts"
USE_VAST = bool(VAST_API_KEY and VAST_CANARY_ENDPOINT and VAST_HIGGS_ENDPOINT)

_vast_client = None
_vast_endpoints: dict = {}

async def get_vast_client():
    global _vast_client
    if _vast_client is None:
        from vastai import Serverless
        _vast_client = Serverless(VAST_API_KEY)
    return _vast_client

async def get_vast_endpoint(name: str):
    if name not in _vast_endpoints:
        client = await get_vast_client()
        _vast_endpoints[name] = await client.get_endpoint(name)
    return _vast_endpoints[name]

async def vast_transcribe(audio_buffer: bytes) -> dict:
    """Send audio to Canary on Vast.ai and return {text, score}."""
    client = await get_vast_client()
    endpoint = await get_vast_endpoint(VAST_CANARY_ENDPOINT)
    audio_b64 = base64.b64encode(audio_buffer).decode()
    # create_wav_buffer is defined below; we need raw WAV bytes here
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_buffer)
    wav_b64 = base64.b64encode(wav_io.getvalue()).decode()
    req = client.queue_endpoint_request(
        endpoint=endpoint,
        worker_route="/transcribe_b64",
        worker_payload={"audio_b64": wav_b64},
        timeout=60.0,
    )
    result = await asyncio.wrap_future(req)
    return result.get("response", {})

async def vast_tts(payload: dict) -> bytes:
    """Send TTS request to Higgs on Vast.ai, return raw PCM bytes."""
    client = await get_vast_client()
    endpoint = await get_vast_endpoint(VAST_HIGGS_ENDPOINT)
    req = client.queue_endpoint_request(
        endpoint=endpoint,
        worker_route="/generate_b64",
        worker_payload=payload,
        timeout=120.0,
    )
    result = await asyncio.wrap_future(req)
    resp = result.get("response", {})
    # Higgs returns base64-encoded audio when called via JSON route
    audio_b64 = resp.get("audio_b64") if isinstance(resp, dict) else None
    if audio_b64:
        return base64.b64decode(audio_b64)
    return b""


SAMPLE_RATE = 16000
VAD_THRESHOLD = 1800 # Adjusted sensitivity based on user feedback
SILENCE_DURATION = 0.8  # Stability over extreme speed
MIN_AUDIO_DURATION = 0.5 # Minimum audio duration to process

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
            "You CAN laugh if the user is funny, but you MUST ONLY use 'Haha' or 'Hehe'. NEVER use asterisks or actions like '*laughs*'.\n\n"
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
    """Helper to yield sentences/chunks from a stream of characters/words"""
    buffer = ""
    # Sub-sentence and sentence splitting marks
    terminals = {'.', '!', '?', '\n', ',', ';', ':'}
    
    async for chunk in text_stream:
        for char in chunk:
            buffer += char
            # Check if we have a chunk (15 chars is ~3-4 words)
            if char in terminals and len(buffer.strip()) > 15:
                yield buffer.strip()
                buffer = ""
    
    if buffer.strip():
        yield buffer.strip()

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
        if USE_VAST:
            result = await vast_transcribe(audio_buffer)
        else:
            wav_buffer = create_wav_buffer(audio_buffer)
            data = aiohttp.FormData()
            data.add_field('file', wav_buffer, filename='input.wav', content_type='audio/wav')
            # Increase timeout to 30s for the larger v2 model's first few requests
            async with session.post(CANARY_URL, data=data, timeout=30) as resp:
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

            if USE_VAST:
                audio_data = await vast_tts({"text": text})
                if audio_data:
                    await safe_send_bytes(websocket, audio_data)
            else:
                async with session.post(HIGGS_URL, json={"text": text}, timeout=aiohttp.ClientTimeout(total=30)) as tts_resp:
                    if tts_resp.status == 200:
                        async for chunk in tts_resp.content.iter_any():
                            if chunk: await safe_send_bytes(websocket, chunk)

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

    # 2 & 3. Streaming Intelligence + TTS
    await safe_send_text(websocket, {"state": "processing", "message": "Thinking..."})
    
    full_response = ""
    try:
        # Iterate through sentences as they are generated
        # Use call_groq_via_manager to include history
        async for sentence in split_into_sentences(call_groq_via_manager(session, text, websocket, session_state)):
            full_response += " " + sentence
            logger.info(f"Sentence ready for TTS: {sentence}")
            
            # Notify client of partial text (Thinking...)
            await safe_send_text(websocket, {"state": "thinking", "text": full_response.strip()})
            
            # Immediately call TTS for this sentence (streaming)
            try:
                # Determine voice parameters
                tts_payload = {"text": sentence}
                if session_state.get("voice_mode") == "user" and session_state.get("best_user_audio"):
                    try:
                        with open(session_state["best_user_audio"], "rb") as af:
                            audio_bytes = af.read()
                        tts_payload["ref_audio_base64"] = base64.b64encode(audio_bytes).decode("utf-8")
                    except Exception as e:
                        logger.error(f"Failed to read ref audio for base64: {e}")
                    tts_payload["temperature"] = 0.3 # Lower temperature for stable voice cloning
                    if session_state.get("best_user_text"):
                        tts_payload["ref_text"] = session_state["best_user_text"]
                
                if USE_VAST:
                    audio_data = await vast_tts(tts_payload)
                    if audio_data:
                        await safe_send_bytes(websocket, audio_data)
                    logger.info(f"Finished Vast TTS for sentence: {sentence[:20]}...")
                else:
                    async with session.post(HIGGS_URL, json=tts_payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            # Stream the audio chunks from Higgs
                            async for chunk in resp.content.iter_any():
                                if chunk:
                                    await safe_send_bytes(websocket, chunk)
                            logger.info(f"Finished relaying Higgs stream for sentence: {sentence[:20]}...")
                        else:
                            logger.error(f"Higgs failed for sentence: {resp.status}")
            except Exception as e:
                logger.error(f"TTS sentence exception: {e}")

        # After all sentences are done
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
    is_speaking = False
    current_task = None
    
    # Session state for voice cloning
    session_state = {
        "voice_mode": "system",   # 'system' or 'user'
        "last_user_audio": None
    }
    
    try:
        while True:
            # We use receive() instead of receive_bytes() to handle text (heartbeats) too
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
                        # Heartbeat, ignore
                        continue
                except:
                    pass
                continue
            else:
                # This includes 'websocket.disconnect' types
                if message.get("type") == "websocket.disconnect":
                    logger.info("WebSocket disconnect message received")
                    break
                continue
            
            # Simple VAD (Energy based)
            # 16-bit PCM, 16kHz
            chunk_np = np.frombuffer(data, dtype=np.int16)
            if len(chunk_np) == 0:
                continue
                
            # Calculate RMS amplitude
            energy = np.sqrt(np.mean(chunk_np.astype(float)**2))
            
            # Diagnostic: occasionally log energy for calibration (e.g. every 100 frames or if > 1000)
            if energy > 1000:
                logger.debug(f"Audio energy: {energy:.1f}")
            
            if energy > VAD_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    logger.info("Speech detected")
                    # Interruption: Cancel existing task and notify client
                    if current_task and not current_task.done():
                        current_task.cancel()
                        logger.info("Interrupted current task")
                        await safe_send_text(websocket, {"state": "interrupted"})
                    
                    await safe_send_text(websocket, {"state": "listening", "message": "Listening..."})
                
                # Reset silence timer
                silence_start = None
                audio_buffer.extend(data)
                
            else:
                # Silence frame
                if is_speaking:
                    # Append silence if we are in "speaking" mode to catch trailing words
                    audio_buffer.extend(data)
                    
                    if silence_start is None:
                        silence_start = time.time()
                    else:
                        if time.time() - silence_start > SILENCE_DURATION:
                            # Silence confirmed, process buffer
                            is_speaking = False
                            logger.info("Silence detected, processing buffer")
                            
                            if len(audio_buffer) > 0:
                                current_buffer = bytes(audio_buffer)
                                audio_buffer = bytearray()
                                
                                # Start as background task
                                # Note: We don't wait for it here
                                current_task = asyncio.create_task(process_audio(current_buffer, websocket, session_state))
                            
                            silence_start = None
                else:
                    # Just silence, do nothing
                    pass

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
