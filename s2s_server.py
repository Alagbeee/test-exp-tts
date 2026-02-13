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
# Assuming direct access to services on localhost without proxy path rewrites
CANARY_URL = "http://localhost:8001/transcribe"
HIGGS_URL = "http://localhost:8000/generate_stream"
# Groq Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

SAMPLE_RATE = 16000
VAD_THRESHOLD = 1500 # Very safe threshold to avoid self-triggering loops
SILENCE_DURATION = 0.6  # Seconds of silence to trigger processing
MIN_AUDIO_DURATION = 0.5 # Minimum audio duration to process

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

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
            await websocket.send_text(json.dumps(msg_dict))
    except Exception as e:
        logger.error(f"Error sending text: {e}")

async def safe_send_bytes(websocket: WebSocket, data: bytes):
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_bytes(data)
    except Exception as e:
        logger.error(f"Error sending bytes: {e}")

async def call_groq_stream(session, text):
    """Stream intelligent response from Groq"""
    logger.info(f"Steaming Groq LLM for: {text}")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful, concise AI assistant. Provide short, natural-sounding responses suitable for a voice assistant. Keep it under 50 words."},
            {"role": "user", "content": text}
        ],
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
            
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        chunk = data["choices"][0]["delta"].get("content", "")
                        if chunk:
                            yield chunk
                    except Exception as e:
                        logger.error(f"Error parsing Groq chunk: {e}")
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

async def process_audio(audio_buffer: bytes, websocket: WebSocket):
    """
    1. Send Audio to Canary (ASR)
    2. Send Text to Groq (LLM)
    3. Send Result to Higgs (TTS)
    4. Send Audio back to WebSocket
    """
    # Calculate duration
    duration = len(audio_buffer) / (SAMPLE_RATE * 2)
    if duration < MIN_AUDIO_DURATION:
        logger.info(f"Audio too short ({duration:.2f}s), ignoring.")
        return

    logger.info(f"Processing {len(audio_buffer)} bytes ({duration:.2f}s) of audio...")
    
    await safe_send_text(websocket, {"state": "processing", "message": "Transcribing..."})

    async with aiohttp.ClientSession() as session:
        # 1. ASR - Canary
        text = ""
        try:
            wav_buffer = create_wav_buffer(audio_buffer)
            data = aiohttp.FormData()
            data.add_field('file', wav_buffer, filename='input.wav', content_type='audio/wav')
            
            async with session.post(CANARY_URL, data=data, timeout=15) as resp:
                if resp.status != 200:
                    await safe_send_text(websocket, {"state": "error", "message": f"ASR Error: {resp.status}"})
                    return
                result = await resp.json()
                text = result.get("text", "").strip()
                score = result.get("score", 0.0)
                logger.info(f"Canary Transcript: '{text}' (Score: {score})")
                
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

                # Filter short hallucinations
                if not text or len(text) < 3 or text.lower() in ["problem.", "wow.", "wow", "oh."]:
                    await safe_send_text(websocket, {"state": "idle", "message": "No speech detected."})
                    return
                await safe_send_text(websocket, {"state": "transcribed", "text": text})

        except Exception as e:
            logger.error(f"ASR Exception: {e}")
            await safe_send_text(websocket, {"state": "error", "message": str(e)})
            return

        # 2 & 3. Streaming Intelligence + TTS
        await safe_send_text(websocket, {"state": "processing", "message": "Thinking..."})
        
        full_response = ""
        try:
            # Iterate through sentences as they are generated
            async for sentence in split_into_sentences(call_groq_stream(session, text)):
                full_response += " " + sentence
                logger.info(f"Sentence ready for TTS: {sentence}")
                
                # Notify client of partial text (Thinking...)
                await safe_send_text(websocket, {"state": "thinking", "text": full_response.strip()})
                
                # Immediately call TTS for this sentence (streaming)
                try:
                    async with session.post(HIGGS_URL, json={"text": sentence}, timeout=aiohttp.ClientTimeout(total=30)) as resp:
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
    current_task = None
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Simple VAD (Energy based)
            # 16-bit PCM, 16kHz
            chunk_np = np.frombuffer(data, dtype=np.int16)
            if len(chunk_np) == 0:
                continue
                
            # Calculate RMS amplitude
            energy = np.sqrt(np.mean(chunk_np.astype(float)**2))
            
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
                                current_task = asyncio.create_task(process_audio(current_buffer, websocket))
                            
                            silence_start = None
                else:
                    # Just silence, do nothing
                    pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # manager.disconnect(websocket)

@app.get("/health")
def health():
    return {"status": "ok"}
