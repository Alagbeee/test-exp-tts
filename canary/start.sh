#!/bin/bash
# Startup script for Canary ASR on Vast.ai serverless.
# 1. Start the model server in the background.
# 2. Start the PyWorker (foreground).

mkdir -p /var/log/canary

uvicorn --app-dir /workspace server:app --host 0.0.0.0 --port 8001 \
  2>&1 | tee /var/log/canary/server.log &

echo "Waiting for model server to start..."
until python3 - <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
do
  sleep 5
done
echo "Model server is up. Starting PyWorker..."

python3 /workspace/worker.py
