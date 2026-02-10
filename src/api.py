from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional

from src.models.train_model import train_and_save
from src.models.predict_model import predict_from_bytes, predict_from_db

app = FastAPI(title="COVID-19 Radiography API", version="1.0.0")


class TrainRequest(BaseModel):
    epochs: Optional[int] = 200
    force: Optional[bool] = True
    data_source: Optional[str] = "db"
    db_table: Optional[str] = "images_dataset"
    db_limit: Optional[int] = None
    db_batch_size: Optional[int] = 1000
    apply_masks: Optional[bool] = True


class PredictDbRequest(BaseModel):
    image_id: int
    db_table: Optional[str] = "images_dataset"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train(request: TrainRequest):
    try:
        model_path = train_and_save(
            epochs=request.epochs or 200,
            force=bool(request.force),
            data_source=request.data_source or "db",
            db_table=request.db_table or "images_dataset",
            db_limit=request.db_limit,
            db_batch_size=request.db_batch_size or 1000,
            apply_masks=bool(request.apply_masks),
        )
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


@app.post("/predict-from-db")
async def predict_from_db_endpoint(request: PredictDbRequest):
    try:
        result = await predict_from_db(
            image_id=request.image_id,
            table=request.db_table or "images_dataset",
        )
        return {"image_id": request.image_id, **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
