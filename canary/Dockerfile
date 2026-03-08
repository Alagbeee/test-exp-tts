FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /workspace/nemo-canary

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart

# Copy server script outside the framework directory and point Uvicorn at it explicitly.
COPY canary/server.py /workspace/server.py

# Pre-download the model (optional but good for startup time)
# We can do a python script to download it during build, or just let it download on first run.
# Letting it download on first run is easier for now to avoid build-time network issues or secrets.

CMD ["uvicorn", "--app-dir", "/workspace", "server:app", "--host", "0.0.0.0", "--port", "8001"]
