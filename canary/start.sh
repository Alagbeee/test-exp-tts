#!/bin/bash
# Startup script for Canary ASR on Vast.ai serverless.
# 1. Start the model server in the background.
# 2. Start the PyWorker (foreground).

mkdir -p /var/log/canary

uvicorn --app-dir /workspace server:app --host 0.0.0.0 --port 8001 \
  2>&1 | tee /var/log/canary/server.log &

echo "Waiting for model server to start..."
until curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1; do
  sleep 5
done
echo "Model server is up. Starting PyWorker..."

python3 /workspace/worker.py
