from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
import os

app = FastAPI()

# Proxy configuration
SERVICES = {
    "/api/higgs": "http://127.0.0.1:8000",
    "/api/canary": "http://127.0.0.1:8001",
}

client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

# Proxy endpoints
@app.api_route("/api/higgs/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_higgs(path: str, request: Request):
    target_url = f"{SERVICES['/api/higgs']}/{path}"
    return await proxy_request(target_url, request)

@app.api_route("/api/canary/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_canary(path: str, request: Request):
    target_url = f"{SERVICES['/api/canary']}/{path}"
    return await proxy_request(target_url, request)

async def proxy_request(url: str, request: Request):
    try:
        content = await request.body()
        req_headers = dict(request.headers)
        req_headers.pop("host", None)
        req_headers.pop("content-length", None) # Let httpx handle specific content length

        response = await client.request(
            method=request.method,
            url=url,
            content=content,
            headers=req_headers,
            timeout=120.0 # Longer timeout for audio generation
        )
        
        return StreamingResponse(
            response.aiter_bytes(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            background=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
# Mount root to serve index.html? 
# Usually we mount /static. For root, we can use a catch-all or just serve index.
# Let's verify if index.html is in /workspace/exp/frontend
FRONTEND_DIR = "/root/test-exp-tts/frontend"

@app.get("/")
async def read_index():
    return FileResponse(f"{FRONTEND_DIR}/index.html")

# Configure static files for anything else (css, js if we had them separate)
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")
