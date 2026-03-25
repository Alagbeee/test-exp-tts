FROM python:3.12-slim

# webrtcvad-wheels needs gcc for the C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY translation_server.py .
COPY translation_frontend/ translation_frontend/

EXPOSE 8083

CMD ["uvicorn", "translation_server:app", "--host", "0.0.0.0", "--port", "8083"]
