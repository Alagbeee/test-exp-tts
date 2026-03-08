#!/bin/bash

# Start Frontend Proxy (Runs on 8081)
echo "Starting Frontend Proxy on port 8081..."
source /root/test-exp-tts/venv_higgs/bin/activate
cd /root/test-exp-tts/frontend
# Run proxy server using uvicorn
uvicorn proxy_server:app --host 0.0.0.0 --port 8081 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
deactivate

# Start Higgs Service (GPU 0)
echo "Starting Higgs Audio Service on port 8000 (GPU 0)..."
source /root/test-exp-tts/venv_higgs/bin/activate
cd /root/test-exp-tts/higgs
# We set CUDA_VISIBLE_DEVICES to only 0 to ensure it doesn't see GPU 1 and accidentally use it
export CUDA_VISIBLE_DEVICES=0
uvicorn server:app --host 0.0.0.0 --port 8000 > ../higgs.log 2>&1 &
HIGGS_PID=$!
deactivate

# Start Canary Service (GPU 1)
echo "Starting Canary Service on port 8001 (GPU 1)..."
source /root/test-exp-tts/venv_canary/bin/activate
cd /root/test-exp-tts/canary
# Strict isolation: Only expose GPU 1.
# Inside the container/process, this will appear as "cuda:0".
# The python script logic handles "cuda:0" if count == 1.
export CUDA_VISIBLE_DEVICES=1
uvicorn server:app --host 0.0.0.0 --port 8001 > ../canary.log 2>&1 &
CANARY_PID=$!
deactivate

# Start Cloudflare Tunnel
# Start S2S Orchestrator (Runs on 8082)
echo "Starting S2S Orchestrator on port 8082..."
# Ensure we are in the correct directory
cd /root/test-exp-tts
# Use higgs venv as it has audio libs and aiohttp
source /root/test-exp-tts/venv_higgs/bin/activate
# Ensure aiohttp is installed (we did it manually, but good to be safe)
uvicorn s2s_server:app --host 0.0.0.0 --port 8082 > s2s.log 2>&1 &
S2S_PID=$!
deactivate

echo "Waiting for services to initialize..."
sleep 10

# Start Cloudflare Tunnel (Pointing to S2S Server 8082)
echo "Starting Cloudflare Tunnel..."
if [ -f "cloudflared" ]; then
    ./cloudflared tunnel --url http://localhost:8082 > ../tunnel.log 2>&1 &
else
    # Fallback to local execution if sudo fails (likely in this environment)
    curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x cloudflared
    ./cloudflared tunnel --url http://localhost:8082 > ../tunnel.log 2>&1 &
fi
TUNNEL_PID=$!

echo "Services started. PIDs: S2S=$S2S_PID, Higgs=$HIGGS_PID, Canary=$CANARY_PID, Tunnel=$TUNNEL_PID"
echo "Tunnel log is being written to tunnel.log. You can check it for the URL."
echo "Press Ctrl+C to stop all services."

# Trap Ctrl+C to kill all background processes
trap "kill $S2S_PID $HIGGS_PID $CANARY_PID $TUNNEL_PID; exit" INT
wait
