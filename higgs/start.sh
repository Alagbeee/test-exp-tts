#!/bin/bash
# Startup script for Higgs TTS on Vast.ai serverless.
# 1. Start the model server in the background (logging to file for PyWorker readiness detection).
# 2. Start the PyWorker (foreground, keeps container alive).

mkdir -p /var/log/higgs

uvicorn --app-dir /workspace server:app --host 0.0.0.0 --port 8000 \
  2>&1 | tee /var/log/higgs/server.log &

# Wait for uvicorn to start accepting connections
echo "Waiting for model server to start..."
until python3 - <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
do
  sleep 5
done
echo "Model server is up. Starting PyWorker..."

python3 /workspace/worker.py
