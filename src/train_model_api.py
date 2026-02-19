from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest

from src.models.train_model_mlflow import train_model_mlflow

app = FastAPI(title="COVID-19 Radiography API", version="1.0.0")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API Requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency"
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    with REQUEST_LATENCY.time():
        response = await call_next(request)
    REQUEST_COUNT.labels(request.method, request.url.path).inc()
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train():
    try:
        train_model_mlflow()
        return {"status": "trained"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))