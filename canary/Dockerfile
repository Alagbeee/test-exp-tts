FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /workspace/nemo-canary

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart

# Models loaded from network volume at /models at runtime

# Copy server files
COPY canary/server.py /workspace/server.py
COPY canary/worker.py /workspace/worker.py
COPY canary/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

ENV CANARY_MODEL_PATH=/models/canary
ENV WORKER_PORT=70703

# start.sh starts the model server, waits for readiness, then launches
# the Vast.ai PyWorker (which handles serverless request routing).
CMD ["/workspace/start.sh"]
