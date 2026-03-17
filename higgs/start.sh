#!/bin/bash
# Startup script for Higgs TTS on RunPod serverless (load-balancing endpoint).
# Starts uvicorn immediately; model loads in a background thread inside server.py.
# RunPod health-checks /ping: returns 204 while loading, 200 when ready.

export HIGGS_MODEL_PATH=/models/higgs
export HIGGS_TOKENIZER_PATH=/models/higgs-tokenizer
export PYTHONUNBUFFERED=1
PORT=${PORT:-80}

exec uvicorn --app-dir /workspace server:app --host 0.0.0.0 --port "$PORT"
