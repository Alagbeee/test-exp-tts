#!/bin/bash
# Startup script for Higgs TTS on Vast.ai serverless.
# 1. Download model weights if not already present.
# 2. Start the model server in the background (logging to file for PyWorker readiness detection).
# 3. Start the PyWorker (foreground, keeps container alive).

mkdir -p /var/log/higgs

# Download models on first start (skipped if weights already exist)
if [ ! -d "/models/higgs" ] || [ -z "$(ls -A /models/higgs 2>/dev/null)" ]; then
  echo "Downloading model weights..."
  python3 /workspace/download_models.py
fi

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

# Always run the latest vastai-sdk so Vast.ai's autoscaler never rejects
# the worker as 'pyworker outdated'.
pip install --upgrade --quiet vastai-sdk

python3 /workspace/worker.py
