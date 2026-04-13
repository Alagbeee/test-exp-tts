"""
Download Voxtral-4B-TTS-2603 weights to the RunPod network volume.

Run this once from the RunPod network volume job before creating the endpoint:
  python3 download_models.py

Requires:
  pip install huggingface_hub
  HF_TOKEN env var (needed because model has gated access)
"""

import os
from huggingface_hub import snapshot_download

tok = os.environ.get("HF_TOKEN") or None
if not tok:
    print("WARNING: HF_TOKEN not set — download may fail for gated models")

print("Downloading mistralai/Voxtral-4B-TTS-2603 ...")
snapshot_download(
    "mistralai/Voxtral-4B-TTS-2603",
    local_dir="/models/voxtral",
    token=tok,
    ignore_patterns=["original/*"],  # keep voice_embedding/*.pt files
)
print("Download complete → /models/voxtral")
