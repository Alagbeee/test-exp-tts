FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /workspace/nemo-canary

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart "vastai-sdk>=0.5.1"

# Download model weights at build time so cold starts are fast.
RUN pip install huggingface_hub hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1
COPY canary/download_models.py /workspace/download_models.py
RUN python3 /workspace/download_models.py

# Copy server files
COPY canary/server.py /workspace/server.py
COPY canary/worker.py /workspace/worker.py
COPY canary/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

ENV CANARY_MODEL_PATH=/models/canary

# start.sh starts the model server, waits for readiness, then launches
# the Vast.ai PyWorker (which handles serverless request routing).
CMD ["/workspace/start.sh"]
