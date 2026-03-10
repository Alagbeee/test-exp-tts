FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /workspace/nemo-canary

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart

# Copy server script outside the framework directory and point Uvicorn at it explicitly.
COPY canary/server.py /workspace/server.py

# Bake model weights into the image at build time for instant cold starts.
# nvidia/canary-1b-v2 is public on HuggingFace — no token needed.
RUN python3 -c "
from huggingface_hub import snapshot_download
print('Downloading canary-1b-v2...')
snapshot_download('nvidia/canary-1b-v2', local_dir='/models/canary')
print('Model download complete.')
"

ENV CANARY_MODEL_PATH=/models/canary

CMD ["uvicorn", "--app-dir", "/workspace", "server:app", "--host", "0.0.0.0", "--port", "8001"]
