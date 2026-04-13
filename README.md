# Live Translation

Real-time speech translation via WebSocket. Speak in one language, get subtitles and optional TTS in another — with sub-second latency.

**Pipeline:** Browser mic → WebSocket → Canary ASR (NVIDIA) → Groq LLM (correct + translate) → Higgs TTS (optional)

---

## Quick Start (Docker)

```bash
# 1. Clone
git clone https://github.com/Alagbeee/test-exp-tts.git
cd test-exp-tts

# 2. Configure secrets
cp .env.example .env
# Edit .env — add your GROQ_API_KEY, RUNPOD_API_KEY, CANARY_URL, HIGGS_URL

# 3. Run
docker compose up --build
```

Open `http://localhost:8083` in your browser.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (LLM translation + ASR correction) |
| `RUNPOD_API_KEY` | RunPod bearer token |
| `CANARY_URL` | Canary ASR endpoint — `https://<id>.api.runpod.ai/transcribe` |
| `HIGGS_URL` | Higgs TTS endpoint — `https://<id>.api.runpod.ai/generate_stream` (optional) |

---

## API

The server exposes a WebSocket API at `ws://<host>:8083/ws`.

See [API_REFERENCE.md](API_REFERENCE.md) for the full protocol — intended for frontend engineers integrating the service.

### HTTP
- `GET /health` — health check JSON
- `GET /` — built-in demo frontend

---

## Architecture

| Component | Role |
|-----------|------|
| **NVIDIA Canary-1b-v2** (RunPod) | Speech-to-text (ASR) |
| **Groq llama-3.3-70b-versatile** | Correct ASR output + translate |
| **Higgs Audio v2** (RunPod) | Text-to-speech (optional) |
| **WebRTC VAD** | Detect speech vs silence — fires translation on natural pauses |

Audio flows: raw 16kHz mono PCM from browser → VAD-gated silence detection → Canary ASR on complete utterances → Groq correction + translation → result sent back over WebSocket.

---

## Running Locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
set -a && . ./.env && set +a
uvicorn translation_server:app --host 0.0.0.0 --port 8083
```

---

## Infrastructure (RunPod)

Both AI services run as RunPod serverless endpoints:

| Service | Endpoint ID | Docker Image |
|---------|-------------|--------------|
| Canary ASR | `d8b6cpdq7sorxt` | `alagbeee/canary:latest` |
| Higgs TTS | `nyw4niybx5auc0` | `alagbeee/higgs:latest` |

