#!/bin/bash
set -e

# Create directories
mkdir -p /workspace/exp/venv_higgs
mkdir -p /workspace/exp/venv_canary

# --- Setup Higgs Environment ---
echo "Setting up Higgs Audio environment..."
python3 -m venv /workspace/exp/venv_higgs
source /workspace/exp/venv_higgs/bin/activate

# Install dependencies for Higgs
# We need to check what torch version is compatible. Higgs V2 usually needs modern torch.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart httpx

# Install Higgs Audio
cd /workspace/exp/higgs-audio
pip install -r requirements.txt
pip install -e .
deactivate

# --- Setup Canary Environment ---
echo "Setting up Canary environment..."
python3 -m venv /workspace/exp/venv_canary
source /workspace/exp/venv_canary/bin/activate

# Install dependencies for Canary (NeMo)
# NeMo often requires specific versions, let's install the latest compatible
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nemo_toolkit[asr]
pip install fastapi uvicorn python-multipart
deactivate

echo "Environment setup complete."
