"""
Vast.ai PyWorker for the Higgs TTS service.
The HTTP model server (uvicorn) is started by start.sh before this runs.
"""
from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8000,
    model_log_file="/var/log/higgs/server.log",
    handlers=[
        HandlerConfig(
            route="/generate_b64",
            allow_parallel_requests=False,  # Higgs is stateful / single-GPU
            max_queue_time=120.0,
            workload_calculator=lambda payload: 100.0,
            benchmark_config=BenchmarkConfig(
                generator=lambda: {"text": "Hello, this is a benchmark."},
                runs=2,
                concurrency=1,
            ),
        )
    ],
    log_action_config=LogActionConfig(
        on_load=["Model loaded successfully."],
        on_error=["Error loading model", "Traceback (most recent call last):"],
    ),
)

Worker(worker_config).run()
