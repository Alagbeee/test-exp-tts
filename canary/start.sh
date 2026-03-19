#!/bin/bash
# Startup script for Canary ASR on RunPod serverless (load-balancing endpoint).
# Starts uvicorn immediately; model loads in a background thread inside server.py.
# RunPod health-checks /ping: returns 204 while loading, 200 when ready.

export CANARY_MODEL_PATH=/models/canary
export PYTHONUNBUFFERED=1
PORT=${PORT:-80}

echo "=== Canary ASR start.sh ==="
echo "PORT=$PORT"
echo "CANARY_MODEL_PATH=$CANARY_MODEL_PATH"
echo "Starting uvicorn on 0.0.0.0:$PORT ..."

exec uvicorn --app-dir /workspace server:app --host 0.0.0.0 --port "$PORT"
