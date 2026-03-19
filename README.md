# joble-s2s

Real-time Speech-to-Speech (S2S) system using RunPod serverless GPU infrastructure.

## Architecture

| Component | Service | Docker Image | RunPod Endpoint |
|-----------|---------|-------------|-----------------|
| ASR | NVIDIA Canary 1B v2 | `15wins/canary:v4` | `joble-asr` |
| TTS | Higgs Audio v2 3B | `15wins/higgs:v4` | `joble-tts` |
| Orchestrator | FastAPI + WebSockets | — | `s2s_server.py` |
| Frontend | HTML5 Web Audio | — | `s2s_frontend/` |

## Services

### ASR — `canary/`
- **Model**: `nvidia/canary-1b-v2` (baked into image at `/models/canary`)
- **Server**: FastAPI on port 80, routes: `GET /ping`, `GET /health`, `POST /transcribe`, `POST /transcribe_b64`
- **Dockerfile**: `Dockerfile.canary-v4` (builds from `15wins/canary:v3`)

### TTS — `higgs/`
- **Models**: `bosonai/higgs-audio-v2-generation-3B-base` + tokenizer (baked into image)
- **Server**: FastAPI on port 80, routes: `GET /ping`, `GET /health`, `POST /generate_stream`, `POST /generate_b64`
- **Dockerfile**: `Dockerfile.higgs-v4` (builds from `15wins/higgs:v3`)

### Orchestrator — `s2s_server.py`
FastAPI + WebSocket server that pipes audio through ASR → LLM → TTS in real time.

### Frontend — `s2s_frontend/`
HTML5 Web Audio API client for browser-based S2S calls.

## RunPod Setup
Both endpoints use:
- **GPU**: RTX 4090 / RTX A5000 / RTX A6000 (24–32GB VRAM)
- **FlashBoot**: enabled
- **workersMin**: 1 (always warm)
- **workersMax**: 3–5

## Environment Variables (`.env`)
```
CANARY_URL=https://6j1py0eqi9aokv.api.runpod.ai/transcribe_b64
HIGGS_URL=https://nyw4niybx5auc0.api.runpod.ai/generate_stream
HF_TOKEN=...
```
