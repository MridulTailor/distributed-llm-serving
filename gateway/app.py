import asyncio
import os
import time
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000").rstrip("/")
TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000").rstrip("/")
TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30"))
RETRIES = max(0, int(os.getenv("UPSTREAM_RETRIES", "2")))
REQUESTS = Counter(
    "gateway_requests_total",
    "Total requests received by the gateway.",
    ("route", "backend", "status"),
)
LATENCY = Histogram(
    "gateway_request_duration_seconds",
    "Request duration including retries.",
    ("route", "backend"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
BACKEND_FAILURES = Counter(
    "gateway_backend_failures_total",
    "Upstream failures and retryable 5xx responses.",
    ("backend", "reason"),
)

def record_response(route: str, backend: str, status: int, started: float) -> None:
    LATENCY.labels(route, backend).observe(time.perf_counter() - started)
    REQUESTS.labels(route, backend, str(status)).inc()

@asynccontextmanager
async def lifespan(_: FastAPI):
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        app.state.client = client
        yield

app = FastAPI(title="LLM serving gateway", lifespan=lifespan)

async def dependency_ready(client: httpx.AsyncClient, url: str, path: str) -> bool:
    try:
        response = await client.get(f"{url}{path}", timeout=3)
        return response.is_success
    except (httpx.HTTPError, TimeoutError):
        return False

@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/health/ready")
async def ready(request: Request) -> Response:
    client = request.app.state.client
    vllm, triton = await asyncio.gather(
        dependency_ready(client, VLLM_URL, "/health"),
        dependency_ready(client, TRITON_URL, "/v2/health/ready"),
    )
    if not (vllm and triton):
        return JSONResponse(
            {"status": "not_ready", "vllm": vllm, "triton": triton}, status_code=503
        )
    return JSONResponse({"status": "ready", "vllm": True, "triton": True})

@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

async def proxy(
    request: Request, upstream: str, path: str, route: str, backend: str
) -> Response:
    started = time.perf_counter()
    payload = await request.body()
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    client: httpx.AsyncClient = request.app.state.client
    last_error: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            upstream_response = await client.request(
                request.method, f"{upstream}{path}", content=payload, headers=headers
            )
            if upstream_response.status_code >= 500 and attempt < RETRIES:
                BACKEND_FAILURES.labels(backend, "upstream_5xx").inc()
                await asyncio.sleep(0.1 * (2**attempt))
                continue
            record_response(route, backend, upstream_response.status_code, started)
            return Response(
                upstream_response.content,
                status_code=upstream_response.status_code,
                headers={
                    "content-type": upstream_response.headers.get(
                        "content-type", "application/json"
                    )
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            last_error = error
            BACKEND_FAILURES.labels(
                backend, "timeout" if isinstance(error, httpx.TimeoutException) else "network"
            ).inc()
            if attempt < RETRIES:
                await asyncio.sleep(0.1 * (2**attempt))
    status = 504 if isinstance(last_error, httpx.TimeoutException) else 503
    record_response(route, backend, status, started)
    return JSONResponse({"error": "upstream unavailable"}, status_code=status)

@app.post("/v1/chat/completions")
async def completions(request: Request) -> Response:
    return await proxy(
        request, VLLM_URL, "/v1/chat/completions", "/v1/chat/completions", "vllm"
    )

@app.post("/triton/infer")
async def triton_infer(request: Request) -> Response:
    return await proxy(request, TRITON_URL, "/v2/models/echo/infer", "/triton/infer", "triton")
