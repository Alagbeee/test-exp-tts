import asyncio
import os
import json
import logging
import io
import wave
import struct
import time
import aiohttp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

# Configure logging
logging.basicConfig(level=logging.INFO)
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

@app.get("/")
@app.head("/")
async def get():
    with open("/workspace/exp/s2s_frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())


# Configuration
CANARY_URL = "http://127.0.0.1:8001/transcribe"
HIGGS_URL = "http://127.0.0.1:8000/generate_stream"
# Groq Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

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
            "You are a helpful, concise AI assistant. "
            "CRITICAL: Respond ONLY in the same language the user used. "
            "If the user spoke English, you MUST respond in English. "
            "If the user spoke Dutch, you MUST respond in Dutch. "
            "Do not mix languages. Provide short, natural-sounding responses under 50 words.\n\n"
            f"Current Voice Mode: {current_voice}.\n"
            "If the user asks to 'clone my voice' or 'speak like me', confirm you are switching to their voice.\n"
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
        session_state["last_user_audio"] = user_audio_path
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
            if "clone my voice" in text_lower or "speak like me" in text_lower or "copy my voice" in text_lower:
                session_state["voice_mode"] = "user"
                logger.info("Switching to User Voice Mode")
            elif "reset voice" in text_lower or "normal voice" in text_lower or "stop copying" in text_lower:
                session_state["voice_mode"] = "system"
                logger.info("Switching to System Voice Mode")
                
            # Confidence threshold check
            # Canary scores are log-probs; 0.0 is perfect, lower is worse.
            # A threshold of -1.0 is roughly "some uncertainty"
            CONFIDENCE_THRESHOLD = -1.0 
                
            if score < CONFIDENCE_THRESHOLD and len(text.split()) > 2:
                logger.warning(f"Low ASR confidence ({score}). Triggering rephrase.")
                text = "I'm sorry, I didn't hear you clearly. Could you please repeat what you said?"
                # We skip Groq and go straight to TTS with this message
                await safe_send_text(websocket, {"state": "transcribed", "text": "[Low confidence] " + text})
                await safe_send_text(websocket, {"state": "processing", "message": "Asking to repeat..."})
                
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
                if session_state.get("voice_mode") == "user" and session_state.get("last_user_audio"):
                    tts_payload["ref_audio_path"] = session_state["last_user_audio"]
                
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
