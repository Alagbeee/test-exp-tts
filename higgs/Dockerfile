FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /workspace/higgs-audio

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies for API
RUN pip install fastapi uvicorn python-multipart vastai-sdk

# Copy local Higgs Audio repo (we will mount it or copy it, mounting is better for dev but copying is safer for build context if repo is outside)
# Since we cloned it to /workspace/exp/higgs-audio, and this dockerfile is inside /workspace/exp/higgs/Dockerfile, 
# we need to be careful with build context. 
# Plan: The docker-compose context will be /workspace/exp. 
# So we copy higgs-audio from the context.

COPY higgs-audio /workspace/higgs-audio

# Install Higgs Audio dependencies
RUN pip install -r requirements.txt
RUN pip uninstall -y flash-attn || true
RUN pip install -e .

# Bake model weights into the image at build time for instant cold starts.
# Pass --build-arg HF_TOKEN=xxx if the repos require authentication.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
COPY higgs/download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py

# Copy server files after the baked model layer so code-only changes do not force a re-download.
COPY higgs/server.py /workspace/server.py
COPY higgs/worker.py /workspace/worker.py
COPY higgs/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

ENV HIGGS_MODEL_PATH=/models/higgs
ENV HIGGS_TOKENIZER_PATH=/models/higgs-tokenizer

# Default: plain uvicorn (for docker-compose / local use).
# On Vast.ai serverless, override CMD to ["/workspace/start.sh"] in the template.
CMD ["uvicorn", "--app-dir", "/workspace", "server:app", "--host", "0.0.0.0", "--port", "8000"]
