#!/bin/bash
# Startup script for Voxtral TTS on RunPod serverless (load-balancing endpoint).
#
# Flow:
#   RunPod boots container → this script runs → uvicorn starts immediately
#   uvicorn lifespan() → model loads in background thread (in-process via Omni)
#   /ping returns 204 while loading, 200 once model is ready
#   RunPod load-balancer only routes traffic once /ping returns 200

export VOXTRAL_MODEL_PATH=${VOXTRAL_MODEL_PATH:-/models/voxtral}
export VOXTRAL_VOICE=${VOXTRAL_VOICE:-casual_male}
export PYTHONUNBUFFERED=1
PORT=${PORT:-80}

echo "=== Voxtral TTS start.sh ==="
echo "PORT=$PORT"
echo "VOXTRAL_MODEL_PATH=$VOXTRAL_MODEL_PATH"
echo "VOXTRAL_VOICE=$VOXTRAL_VOICE"
echo "Starting uvicorn on 0.0.0.0:$PORT ..."

exec uvicorn --app-dir /workspace server:app --host 0.0.0.0 --port "$PORT"
