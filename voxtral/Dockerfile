# Voxtral TTS — RunPod serverless image (model baked in)
#
# Base: vllm/vllm-omni:v0.18.0 (CUDA 12, Python 3.12, vLLM + vllm-omni pre-installed)
# Model loaded in-process via vllm_omni.Omni (same pattern as Higgs TTS)
# Requires GPU with >= 16 GB VRAM (24 GB recommended)
#
# Build (HF_TOKEN required — model is gated):
#   docker build --build-arg HF_TOKEN=hf_xxx -t zarttechjobpro/voxtral-tts:v9 .

FROM vllm/vllm-omni:v0.18.0

WORKDIR /workspace

# FastAPI wrapper + audio dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    soundfile \
    huggingface_hub

# ── Bake model weights into the image (same approach as Canary/Higgs) ─────────
ARG HF_TOKEN
COPY download_models.py /workspace/download_models.py
RUN HF_TOKEN=${HF_TOKEN} python3 /workspace/download_models.py

COPY server.py /workspace/server.py
COPY start.sh  /workspace/start.sh
COPY voxtral_eager.yaml /workspace/voxtral_eager.yaml
COPY patch_omni.py /workspace/patch_omni.py
RUN python3 /workspace/patch_omni.py
RUN chmod +x /workspace/start.sh

# Environment defaults
ENV VOXTRAL_MODEL_PATH=/models/voxtral
ENV VOXTRAL_VOICE=casual_male
ENV PYTHONUNBUFFERED=1
# Skip NVIDIA container runtime driver version check (requires cuda>=12.9)
ENV NVIDIA_DISABLE_REQUIRE=1

EXPOSE 80

CMD ["/workspace/start.sh"]
