FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /workspace/nemo-canary

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart "vastai-sdk>=0.5.1"

# Model weights are downloaded at first startup (in start.sh) on the Vast.ai
# GPU worker, which has ample RAM and fast egress to HuggingFace.
# hf_transfer enables multi-connection downloads for 5-10x speed.
RUN pip install huggingface_hub hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1
COPY canary/download_models.py /workspace/download_models.py

# Copy server files
COPY canary/server.py /workspace/server.py
COPY canary/worker.py /workspace/worker.py
COPY canary/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

ENV CANARY_MODEL_PATH=/models/canary

# Default: plain uvicorn (for docker-compose / local use).
# On Vast.ai serverless, override CMD to ["/workspace/start.sh"] in the template.
CMD ["uvicorn", "--app-dir", "/workspace", "server:app", "--host", "0.0.0.0", "--port", "8001"]
