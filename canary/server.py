from contextlib import asynccontextmanager
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import nemo.collections.asr as nemo_asr
import torch
import os
import glob
import shutil
import tempfile
import base64

# Configuration
MODEL_NAME = "nvidia/canary-1b-v2"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# RunPod model caching stores HF models here
RUNPOD_HF_CACHE = "/runpod-volume/huggingface-cache/hub"
CANARY_MODEL_PATH = os.environ.get("CANARY_MODEL_PATH", None)

# Model state — loaded in background thread so /ping can respond during startup
asr_model = None
_model_loading = True


def _find_nemo_file():
    """Search for the .nemo file in multiple locations (priority order):
    1. CANARY_MODEL_PATH env var (baked-in or network volume)
    2. RunPod model cache (/runpod-volume/huggingface-cache/hub/...)
    3. Default HuggingFace cache (~/.cache/huggingface/hub/...)
    Returns the path to the .nemo file or None if not found.
    """
    search_paths = []

    # 1. Explicit model path (baked-in)
    if CANARY_MODEL_PATH and os.path.isdir(CANARY_MODEL_PATH):
        search_paths.append(CANARY_MODEL_PATH)

    # 2. RunPod cached model path
    rp_model_dir = os.path.join(RUNPOD_HF_CACHE, "models--nvidia--canary-1b-v2")
    if os.path.isdir(rp_model_dir):
        snapshots_dir = os.path.join(rp_model_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            for d in sorted(os.listdir(snapshots_dir), reverse=True):
                snap = os.path.join(snapshots_dir, d)
                if os.path.isdir(snap):
                    search_paths.append(snap)

    # 3. Default HF cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    hf_model_dir = os.path.join(hf_cache, "models--nvidia--canary-1b-v2")
    if os.path.isdir(hf_model_dir):
        snapshots_dir = os.path.join(hf_model_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            for d in sorted(os.listdir(snapshots_dir), reverse=True):
                snap = os.path.join(snapshots_dir, d)
                if os.path.isdir(snap):
                    search_paths.append(snap)

    for path in search_paths:
        nemo_files = glob.glob(os.path.join(path, "*.nemo"))
        if nemo_files:
            print(f"Found .nemo file at: {nemo_files[0]}")
            return nemo_files[0]
        # Also check one level deeper
        nemo_files = glob.glob(os.path.join(path, "**", "*.nemo"), recursive=True)
        if nemo_files:
            print(f"Found .nemo file at: {nemo_files[0]}")
            return nemo_files[0]

    print("No cached .nemo file found in any search path.")
    print(f"  Searched: {search_paths}")
    return None


def _load_model():
    global asr_model, _model_loading
    print(f"=== Canary ASR Server Starting ===")
    print(f"Device: {DEVICE}")
    print(f"CANARY_MODEL_PATH: {CANARY_MODEL_PATH}")
    print(f"RunPod cache: {RUNPOD_HF_CACHE} (exists: {os.path.isdir(RUNPOD_HF_CACHE)})")
    try:
        nemo_path = _find_nemo_file()
        if nemo_path:
            print(f"Loading from local .nemo: {nemo_path}")
            model = nemo_asr.models.EncDecMultiTaskModel.restore_from(nemo_path)
        else:
            print(f"Downloading model via from_pretrained: {MODEL_NAME}")
            model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=MODEL_NAME)
        asr_model = model.to(DEVICE)
        print("Canary model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        asr_model = None
    finally:
        _model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_load_model, daemon=True)
    thread.start()
    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    """RunPod load-balancing health check. 204=initializing, 200=healthy."""
    if _model_loading:
        return Response(status_code=204)
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model failed to load")
    return {"status": "healthy"}


@app.get("/health")
def health():
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "device": DEVICE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        results = asr_model.transcribe([tmp_path], return_hypotheses=True)

        text = ""
        score = 0.0

        if results and len(results) > 0:
            hyp = results[0]
            if isinstance(hyp, list):
                hyp = hyp[0]

            text = hyp.text
            score = float(getattr(hyp, 'score', 0.0))
            print(f"Transcription: '{text}' (Score: {score})")

        return {"text": text, "score": score}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class TranscribeRequest(BaseModel):
    audio_b64: str  # base64-encoded WAV bytes


@app.post("/transcribe_b64")
async def transcribe_b64(req: TranscribeRequest):
    """JSON endpoint — receives base64-encoded WAV."""
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    audio_bytes = base64.b64decode(req.audio_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        results = asr_model.transcribe([tmp_path], return_hypotheses=True)
        text, score = "", 0.0
        if results:
            hyp = results[0]
            if isinstance(hyp, list):
                hyp = hyp[0]
            text = hyp.text
            score = float(getattr(hyp, "score", 0.0))
        return {"text": text, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
