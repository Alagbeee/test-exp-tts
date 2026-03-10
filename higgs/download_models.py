import os
from huggingface_hub import snapshot_download

tok = os.environ.get("HF_TOKEN") or None
print("Downloading higgs-audio-v2-generation-3B-base...")
snapshot_download("bosonai/higgs-audio-v2-generation-3B-base", local_dir="/models/higgs", token=tok)
print("Downloading higgs-audio-v2-tokenizer...")
snapshot_download("bosonai/higgs-audio-v2-tokenizer", local_dir="/models/higgs-tokenizer", token=tok)
print("Model download complete.")
