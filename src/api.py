from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional

from src.models.train_model import train_and_save
from src.models.predict_model import predict_from_bytes

app = FastAPI(title="COVID-19 Radiography API", version="1.0.0")


class TrainRequest(BaseModel):
    epochs: Optional[int] = 200
    force: Optional[bool] = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train(request: TrainRequest):
    try:
        model_path = train_and_save(epochs=request.epochs or 200, force=bool(request.force))
        return {"status": "trained", "model_path": str(model_path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_from_bytes(image_bytes)
        return {"filename": file.filename, **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
