from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import time
import io
import sys
import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, Response, HTTPException
sys.path.append("/workspace/exp/higgs-audio")

import numpy as np
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
# Force GPU 0 if available, else CPU (though Higgs needs GPU)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000

def create_wav_chunk(audio_data: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    # Safe clipping to avoid distortion
    audio_clipped = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    sf.write(buffer, audio_int16, sample_rate, format='WAV')
    return buffer.getvalue()

print(f"Loading model on {DEVICE}...")
try:
    serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=DEVICE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # We don't exit here so the container stays alive for debugging if needed, but the endpoint will fail
    serve_engine = None

class GenerateRequest(BaseModel):
    text: str
    temperature: float = 0.5
    top_p: float = 0.95
    max_new_tokens: int = 1024
    seed: int = 42
    ref_audio_path: str = None
    ref_text: str = None

@app.get("/health")
def health():
    if serve_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "device": DEVICE}

@app.post("/generate")
def generate_audio(req: GenerateRequest):
    start_time = time.time()
    if serve_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    system_prompt = (
        "Generate audio. Use a professional male voice. Adapt the emotional tone to the content. "
        "If the text is funny, sound amused. If it includes laughter (e.g. 'Haha'), ensure the voice laughs naturally.\n\n"
        "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
    )

    # Reference turn to lock voice (professional male) OR use dynamic request
    ref_audio_path = req.ref_audio_path if req.ref_audio_path else "/workspace/exp/higgs-audio/examples/voice_prompts/en_man.wav"
    
    if req.ref_text:
        ref_text = req.ref_text
    else:
        # Fallback to default text file if using default audio, or generic text
        ref_text_path = "/workspace/exp/higgs-audio/examples/voice_prompts/en_man.txt"
        try:
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        except Exception as e:
            print(f"Error reading ref text: {e}")
            ref_text = "The sun rises in the east and sets in the west."

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=ref_text),
        Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)),
        Message(role="user", content=req.text),
    ]

    try:
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=50,
            seed=req.seed,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        # Save to buffer
        buffer = io.BytesIO()
        # Use soundfile directly to avoid torchaudio/torchcodec issues
        # output.audio might be a tensor or numpy array
        audio_data = output.audio
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.float().cpu().numpy()
            
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
            
        sf.write(buffer, audio_data, output.sampling_rate, format='WAV')
        buffer.seek(0)
        
        total_time = time.time() - start_time
        print(f"DEBUG: Higgs generation took {total_time:.3f}s for '{req.text}'")
        
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/generate_stream")
async def generate_audio_stream(req: GenerateRequest):
    if serve_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    system_prompt = (
        "Generate audio. Use a professional male voice. Adapt the emotional tone to the content. "
        "If the text is funny, sound amused. If it includes laughter (e.g. 'Haha'), ensure the voice laughs naturally.\n\n"
        "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
    )

    ref_audio_path = req.ref_audio_path if req.ref_audio_path else "/workspace/exp/higgs-audio/examples/voice_prompts/en_man.wav"
    
    if req.ref_text:
        ref_text = req.ref_text
    else:
        ref_text_path = "/workspace/exp/higgs-audio/examples/voice_prompts/en_man.txt"
        try:
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        except Exception:
            ref_text = "The sun rises in the east and sets in the west."

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=ref_text),
        Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)),
        Message(role="user", content=req.text),
    ]

    async def audio_generator():
        all_audio_tokens = []
        yielded_tokens = 0
        num_codebooks = serve_engine.audio_num_codebooks
        tps = serve_engine.audio_tokenizer_tps
        sr = serve_engine.audio_tokenizer.sampling_rate
        samples_per_token = int(sr // tps)
        
        # Buffer for decoding context (to prevent edge artifacts)
        DECODE_CONTEXT = 16 
        # Chunk size to yield to client (buffer this many stable tokens)
        CHUNK_TOKENS = 12 # ~160ms at 75 TPS
        
        async for delta in serve_engine.generate_delta_stream(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=50,
            seed=req.seed,
        ):
            if delta.audio_tokens is not None:
                all_audio_tokens.append(delta.audio_tokens.cpu())
                
                # Check how many 'stable' tokens we have after delay pattern reversal
                if len(all_audio_tokens) >= num_codebooks + CHUNK_TOKENS:
                    tokens_tensor = torch.stack(all_audio_tokens, dim=1)
                    reverted = revert_delay_pattern(tokens_tensor)
                    current_reverted_len = reverted.shape[1]
                    
                    # We skip the very first token of the stream [:, 1:-1]
                    start_token = max(yielded_tokens, 1)
                    # We leave at least 1 token at the end for the final trim
                    end_token = current_reverted_len - 1
                    
                    if end_token - start_token >= CHUNK_TOKENS:
                        # Decode with context history
                        decode_start = max(0, start_token - DECODE_CONTEXT)
                        vq_code = reverted[:, decode_start : end_token].clip(0, serve_engine.audio_codebook_size - 1)
                        
                        wv_full = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0).to(DEVICE))[0, 0]
                        
                        # Extract the part we haven't yielded yet
                        offset = (start_token - decode_start) * samples_per_token
                        count = (end_token - start_token) * samples_per_token
                        wv_chunk = wv_full[offset : offset + count]
                        
                        # Yield raw PCM int16
                        audio_int16 = (np.clip(wv_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                        yield audio_int16.tobytes()
                        yielded_tokens = end_token

        # Flush remaining tokens
        if len(all_audio_tokens) >= num_codebooks:
            tokens_tensor = torch.stack(all_audio_tokens, dim=1)
            reverted = revert_delay_pattern(tokens_tensor)
            
            start_token = max(yielded_tokens, 1)
            end_token = reverted.shape[1] - 1 # Final trim
            
            if end_token > start_token:
                decode_start = max(0, start_token - DECODE_CONTEXT)
                vq_code = reverted[:, decode_start : end_token].clip(0, serve_engine.audio_codebook_size - 1)
                wv_full = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0).to(DEVICE))[0, 0]
                
                offset = (start_token - decode_start) * samples_per_token
                wv_chunk = wv_full[offset:]
                
                audio_int16 = (np.clip(wv_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                yield audio_int16.tobytes()

    return StreamingResponse(audio_generator(), media_type="application/octet-stream")
