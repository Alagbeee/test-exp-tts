# test-exp-tts

Real-time Speech-to-Speech (S2S) system optimizing for sub-second latency (~700ms-950ms).

## Components
- **Orchestrator**: `s2s_server.py` (FastAPI + WebSockets)
- **ASR**: NVIDIA Canary
- **TTS**: Higgs Audio (Streaming)
- **Frontend**: HTML5 Web Audio API
