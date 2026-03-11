FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi 'uvicorn[standard]' aiohttp numpy vastai-sdk

COPY s2s_server.py /app/s2s_server.py
COPY s2s_frontend /app/s2s_frontend

CMD ["uvicorn", "s2s_server:app", "--host", "0.0.0.0", "--port", "8082"]