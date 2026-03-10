from huggingface_hub import snapshot_download

print("Downloading canary-1b-v2...")
snapshot_download("nvidia/canary-1b-v2", local_dir="/models/canary")
print("Model download complete.")
