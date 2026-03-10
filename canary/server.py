from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nemo.collections.asr as nemo_asr
import torch
import os
import shutil
import tempfile

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
# options: nvidia/canary-1b, nvidia/canary-1b-v2
MODEL_NAME = "nvidia/canary-1b-v2"
# Force GPU 1
DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu"

CANARY_MODEL_PATH = os.environ.get("CANARY_MODEL_PATH", None)

print(f"Loading Canary model on {DEVICE}...")
try:
    if CANARY_MODEL_PATH:
        import glob
        nemo_files = glob.glob(os.path.join(CANARY_MODEL_PATH, "*.nemo"))
        if nemo_files:
            print(f"Loading from baked-in weights: {nemo_files[0]}")
            asr_model = nemo_asr.models.EncDecMultiTaskModel.restore_from(nemo_files[0])
        else:
            print(f"No .nemo file found in {CANARY_MODEL_PATH}, falling back to from_pretrained")
            asr_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=MODEL_NAME)
    else:
        asr_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=MODEL_NAME)
    asr_model = asr_model.to(DEVICE)
    print("Canary model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    asr_model = None

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
