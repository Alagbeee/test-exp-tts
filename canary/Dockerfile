FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /workspace/nemo-canary

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart vastai-sdk

# Bake model weights into the image at build time for instant cold starts.
# nvidia/canary-1b-v2 is public on HuggingFace — no token needed.
COPY canary/download_models.py /tmp/download_models.py
RUN pip install huggingface_hub && python3 /tmp/download_models.py

# Copy server files after the baked model layer so code-only changes do not force a re-download.
COPY canary/server.py /workspace/server.py
COPY canary/worker.py /workspace/worker.py
COPY canary/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

ENV CANARY_MODEL_PATH=/models/canary

# Default: plain uvicorn (for docker-compose / local use).
# On Vast.ai serverless, override CMD to ["/workspace/start.sh"] in the template.
CMD ["uvicorn", "--app-dir", "/workspace", "server:app", "--host", "0.0.0.0", "--port", "8001"]
