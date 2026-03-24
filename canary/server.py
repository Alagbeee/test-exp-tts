from contextlib import asynccontextmanager
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
# NOTE: nemo, torch, OmegaConf are imported lazily inside _load_model / _run_inference
# so that uvicorn starts instantly and /ping can respond during init.
import os
import glob
import shutil
import tempfile
import base64

# Configuration
MODEL_NAME = "nvidia/canary-1b-v2"
DEVICE = "cpu"  # set properly in _load_model once torch is imported

# RunPod model caching stores HF models here
RUNPOD_HF_CACHE = "/runpod-volume/huggingface-cache/hub"
CANARY_MODEL_PATH = os.environ.get("CANARY_MODEL_PATH", None)

# Model state — loaded in background thread so /ping can respond during startup
asr_model = None
_model_loading = True
# Serialize all GPU inference + model-config changes (transcription and translation share the model)
_infer_lock = threading.Lock()


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
    global asr_model, _model_loading, DEVICE
    import torch
    import nemo.collections.asr as nemo_asr
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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


# ---------------------------------------------------------------------------
# Translation helpers — Canary-1b supports speech-to-text translation between
# {en, de, es, fr} (English must be one of the two languages).
# We serialise all inference through _infer_lock because change_decoding_strategy
# mutates shared model state.
# ---------------------------------------------------------------------------

SUPPORTED_LANGS = {
    "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru",
    "sk", "sl", "es", "sv", "uk",
}


def _run_inference(audio_path: str, task: str, source_lang: str, target_lang: str):
    """Blocking GPU call — must be called while holding _infer_lock."""
    from omegaconf import OmegaConf
    try:
        decoding_cfg = OmegaConf.to_container(asr_model.cfg.decoding, resolve=True)
        decoding_cfg["task"] = task
        decoding_cfg["source_lang"] = source_lang
        decoding_cfg["target_lang"] = target_lang
        decoding_cfg["pnc"] = "yes"
        asr_model.change_decoding_strategy(OmegaConf.create(decoding_cfg))
        results = asr_model.transcribe([audio_path], return_hypotheses=True)
    finally:
        # Always restore to vanilla ASR so /transcribe still works correctly
        try:
            restore_cfg = OmegaConf.to_container(asr_model.cfg.decoding, resolve=True)
            restore_cfg["task"] = "asr"
            restore_cfg["source_lang"] = source_lang
            restore_cfg["target_lang"] = source_lang
            asr_model.change_decoding_strategy(OmegaConf.create(restore_cfg))
        except Exception:
            pass

    text, score = "", 0.0
    if results:
        hyp = results[0]
        if isinstance(hyp, list):
            hyp = hyp[0]
        text = hyp.text
        score = float(getattr(hyp, "score", 0.0))
    return text, score


@app.post("/translate")
async def translate_audio(
    file: UploadFile = File(...),
    source_lang: str = Query("en", description="Source language code: en / de / es / fr"),
    target_lang: str = Query("de", description="Target language code: en / de / es / fr"),
):
    """Speech-to-text translation. Canary-1b supports en↔{de,es,fr}."""
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if source_lang not in SUPPORTED_LANGS or target_lang not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported lang. Supported: {SUPPORTED_LANGS}")
    if source_lang == target_lang:
        raise HTTPException(status_code=400, detail="source_lang and target_lang must differ")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import asyncio
        loop = asyncio.get_event_loop()
        text, score = await loop.run_in_executor(
            None,
            lambda: _run_with_lock(tmp_path, "s2t_translation", source_lang, target_lang),
        )
        print(f"Translation [{source_lang}→{target_lang}]: '{text}' (score={score:.3f})")
        return {"text": text, "score": score, "source_lang": source_lang, "target_lang": target_lang}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _run_with_lock(audio_path, task, source_lang, target_lang):
    with _infer_lock:
        return _run_inference(audio_path, task, source_lang, target_lang)


class TranslateRequest(BaseModel):
    audio_b64: str
    source_lang: str = "en"
    target_lang: str = "de"


@app.post("/translate_b64")
async def translate_b64(req: TranslateRequest):
    """JSON endpoint for translation — receives base64-encoded WAV."""
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if req.source_lang not in SUPPORTED_LANGS or req.target_lang not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported lang pair")

    audio_bytes = base64.b64decode(req.audio_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        text, score = await loop.run_in_executor(
            None,
            lambda: _run_with_lock(tmp_path, "s2t_translation", req.source_lang, req.target_lang),
        )
        return {"text": text, "score": score, "source_lang": req.source_lang, "target_lang": req.target_lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
