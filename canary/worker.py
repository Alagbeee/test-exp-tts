"""
Vast.ai PyWorker for the Canary ASR service.
The HTTP model server (uvicorn) is started by start.sh before this runs.
"""
from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8001,
    model_log_file="/var/log/canary/server.log",
    handlers=[
        HandlerConfig(
            route="/transcribe_b64",
            allow_parallel_requests=False,
            max_queue_time=60.0,
            workload_calculator=lambda payload: 100.0,
            benchmark_config=BenchmarkConfig(
                # Minimal silent WAV (44 bytes header, no audio data) encoded as base64
                generator=lambda: {"audio_b64": "UklGRiQAAABXQVZFZm10IBAAAA"
                                                 "EAAQAAgD4AAAD4AAEAAgAYAAAA"
                                                 "ZGF0YQAAAAA="},
                runs=2,
                concurrency=1,
            ),
        )
    ],
    log_action_config=LogActionConfig(
        on_load=["Canary model loaded successfully."],
        on_error=["Error loading model", "Traceback (most recent call last):"],
    ),
)

Worker(worker_config).run()
