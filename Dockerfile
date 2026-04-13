FROM python:3.12-slim

# webrtcvad needs gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY translation_server.py .
COPY translation_frontend/ translation_frontend/

# Run as non-root
RUN useradd -m appuser
USER appuser

EXPOSE 8083

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

CMD ["uvicorn", "translation_server:app", "--host", "0.0.0.0", "--port", "8083", "--workers", "1"]
