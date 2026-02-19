from fastapi import FastAPI, HTTPException

from src.models.train_model_mlflow import train_model_mlflow

app = FastAPI(title="COVID-19 Radiography API", version="1.0.0")


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