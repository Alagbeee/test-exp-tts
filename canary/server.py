from contextlib import asynccontextmanager
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import nemo.collections.asr as nemo_asr
import torch
import os
import shutil
import tempfile
import base64

# Configuration
# options: nvidia/canary-1b, nvidia/canary-1b-v2
MODEL_NAME = "nvidia/canary-1b-v2"
# Force GPU 1
DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu"

CANARY_MODEL_PATH = os.environ.get("CANARY_MODEL_PATH", None)

# Model state — loaded in background thread so /ping can respond during startup
asr_model = None
_model_loading = True

def _load_model():
    global asr_model, _model_loading
    print(f"Loading Canary model on {DEVICE}...")
    try:
        if CANARY_MODEL_PATH:
            import glob
            nemo_files = glob.glob(os.path.join(CANARY_MODEL_PATH, "*.nemo"))
            if nemo_files:
                print(f"Loading from baked-in weights: {nemo_files[0]}")
                model = nemo_asr.models.EncDecMultiTaskModel.restore_from(nemo_files[0])
            else:
                print(f"No .nemo file found in {CANARY_MODEL_PATH}, falling back to from_pretrained")
                model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=MODEL_NAME)
        else:
            model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=MODEL_NAME)
        asr_model = model.to(DEVICE)
        print("Canary model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
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

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Canary expects a list of files
        # We return hypotheses to get the confidence scores
        results = asr_model.transcribe([tmp_path], return_hypotheses=True)
        
        text = ""
        score = 0.0
        
        if results and len(results) > 0:
            # results is List[List[Hypothesis]]
            # hyp is usually the top prediction
            hyp = results[0]
            if isinstance(hyp, list):
                hyp = hyp[0]
            
            text = hyp.text
            score = float(getattr(hyp, 'score', 0.0))
            print(f"DEBUG: Transcription: '{text}' (Score: {score})")
        
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
    """JSON endpoint for Vast.ai serverless — receives base64-encoded WAV."""
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
