"""
Voxtral TTS server for RunPod serverless (load-balancing endpoint).

Uses vllm-omni's Omni API in-process (same pattern as Higgs TTS).
No subprocess — model is loaded directly into Python like HiggsAudioServeEngine.

Interface (matches Higgs TTS so translation_server.py needs zero changes):
  GET  /ping            → 204 (loading) | 200 (ready)
  GET  /health          → 503 | {"status": "ok"}
  POST /generate_stream → WAV audio (24 kHz mono)
"""

import gc
import io
import os
import time
import traceback
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vllm import SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.environ.get("VOXTRAL_MODEL_PATH", "/models/voxtral")
DEFAULT_VOICE = os.environ.get("VOXTRAL_VOICE", "casual_male")
SAMPLE_RATE   = 24000

# ── State ─────────────────────────────────────────────────────────────────────
_omni = None
_tokenizer = None
_model_loading = True


# Custom stage config (enforce_eager=true for GPU compatibility)
CUSTOM_STAGE_CONFIG = "/workspace/voxtral_eager.yaml"


def _load_model():
    """Load Voxtral via vllm-omni's Omni API (in-process, like Higgs)."""
    global _omni, _tokenizer, _model_loading

    # Log GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu}, VRAM: {mem:.1f}GB", flush=True)
        else:
            print("WARNING: No CUDA GPU detected!", flush=True)
    except Exception as e:
        print(f"GPU check error: {e}", flush=True)

    print(f"Loading Voxtral model from {MODEL_PATH} ...", flush=True)
    try:
        from vllm_omni.entrypoints.omni import Omni
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        # Use custom eager stage config (enforce_eager=true disables CUDA graphs
        # for compatibility with non-H100 GPUs)
        stage_configs = CUSTOM_STAGE_CONFIG
        if not os.path.isfile(stage_configs):
            # Fallback to default
            import vllm_omni
            stage_configs = os.path.join(
                os.path.dirname(vllm_omni.__file__),
                "model_executor", "stage_configs", "voxtral_tts.yaml",
            )
        print(f"Using stage config: {stage_configs}", flush=True)

        _omni = Omni(model=MODEL_PATH, stage_configs_path=stage_configs)
        _tokenizer = MistralTokenizer.from_file(
            str(Path(MODEL_PATH) / "tekken.json")
        )
        print("Voxtral model loaded successfully.", flush=True)
    except Exception as e:
        print(f"ERROR loading Voxtral model: {e}", flush=True)
        traceback.print_exc()
        _omni = None
    finally:
        _model_loading = False


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health / ping ─────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    """RunPod load-balancer health check — 204 while loading, 200 when ready."""
    if _model_loading:
        return Response(status_code=204)
    if _omni is None:
        raise HTTPException(status_code=503, detail="Model failed to load")
    return {"status": "healthy"}


@app.get("/health")
def health():
    if _omni is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": MODEL_PATH, "voice": DEFAULT_VOICE}


# ── TTS endpoint ──────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    ref_audio: str | None = None  # base64-encoded WAV for voice cloning/accent


@app.post("/generate_stream")
def generate_stream(req: GenerateRequest):
    """
    Generate WAV audio for the given text.

    Compatible with translation_server.py's _stream_tts():
      - Accepts  POST {"text": "...", "voice": "..."}
      - Returns  audio/wav (24 kHz mono)
    """
    if _omni is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    if _tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not ready")

    from mistral_common.protocol.speech.request import SpeechRequest

    try:
        # Tokenize the speech request
        instruct_tokenizer = _tokenizer.instruct_tokenizer
        tokenized = instruct_tokenizer.encode_speech_request(
            SpeechRequest(input=req.text, voice=req.voice, ref_audio=req.ref_audio)
        )

        inputs = {
            "additional_information": {"voice": [req.voice]},
            "prompt_token_ids": tokenized.tokens,
        }

        # Two-stage sampling params (text-tokens stage + audio-decode stage)
        sp = SamplingParams(max_tokens=2500)
        sampling_params_list = [sp, sp]

        start = time.time()
        outputs = _omni.generate(inputs, sampling_params_list)
        elapsed = time.time() - start

        for o in outputs:
            audio_tensor = torch.cat(o.multimodal_output["audio"])
            audio_array = audio_tensor.float().cpu().numpy()

            buf = io.BytesIO()
            sf.write(buf, audio_array, SAMPLE_RATE, format="WAV")
            buf.seek(0)

            dur = len(audio_array) / SAMPLE_RATE
            mode = "ref_audio_clone" if req.ref_audio else req.voice
            print(f"TTS: {dur:.2f}s audio in {elapsed:.2f}s "
                  f"(RTF={dur/elapsed:.2f}) voice={mode}", flush=True)

            return Response(content=buf.read(), media_type="audio/wav")

        raise HTTPException(status_code=500, detail="No audio generated")

    except HTTPException:
        raise
    except Exception as e:
        import traceback as tb
        full_trace = tb.format_exc(chain=True)
        print(f"TTS generation error:\n{full_trace}", flush=True)
        # Include repr(e) + tail of traceback so it's visible in the API response
        detail = f"EXCEPTION: {repr(e)}\n\nTRACEBACK:\n{full_trace}"
        raise HTTPException(status_code=500, detail=detail[-3000:])
