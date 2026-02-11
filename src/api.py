import base64
import io
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2

from src.models.train_model import train_and_save
from src.models.predict_model import predict_from_bytes, predict_from_db, predict_with_gradcam
from src.data.database_access import DatabaseAccess
from src.settings import DatabaseSettings

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


def _heatmap_to_base64_png(heatmap: np.ndarray) -> str:
    """Encode heatmap (0-1 float) as PNG base64."""
    h_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", h_uint8)
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.post("/predict-with-gradcam")
async def predict_with_gradcam_endpoint(file: UploadFile = File(...)):
    """
    Prédiction + image + Grad-CAM. Retourne:
    - label, probabilities, class_index, image_size
    - image_base64: l'image envoyée (base64)
    - heatmap_base64: la heatmap Grad-CAM (base64 PNG)
    """
    try:
        image_bytes = await file.read()
        result = predict_with_gradcam(image_bytes)
        heatmap = np.array(result.pop("heatmap"), dtype=np.float32)
        result["heatmap_base64"] = _heatmap_to_base64_png(heatmap)
        result["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
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


def _get_secrets_path() -> Path:
    """Chemin vers le fichier de secrets (secrets.yaml, secret.yml, etc.)."""
    candidates = ("secrets.yaml", "secrets.yml", "secret.yaml", "secret.yml")
    for base in [Path.cwd(), Path(__file__).resolve().parents[2]]:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    return Path("secrets.yaml")


@app.post("/feedback")
async def submit_feedback(
    image: UploadFile = File(None),
    predicted_class: str = Form(...),
    diagnostic: str = Form(...),
    comment: str = Form(""),
):
    """
    Enregistre un feedback médecin uniquement en base (table feedback). Pas d’upload S3.
    """
    try:
        secrets_path = _get_secrets_path()
        if not secrets_path.exists():
            raise HTTPException(status_code=500, detail="secrets.yaml not found")

        db_settings = DatabaseSettings(str(secrets_path))
        api_url, api_key = db_settings.database_url
        db = DatabaseAccess(api_url=api_url, api_key=api_key)
        feedback_date = datetime.utcnow().isoformat()
        row = {
            "predicted_class": predicted_class,
            "diagnostic": diagnostic,
            "comment": comment,
            "feedback_date": feedback_date,
        }
        await db.insert("feedback", row)
        return {"status": "ok"}
    except httpx.HTTPStatusError as e:
        detail = (e.response.text or "").strip() or f"HTTP {e.response.status_code}"
        raise HTTPException(status_code=500, detail=f"Supabase: {detail}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        msg = str(exc)
        if "Expecting value" in msg or "JSON" in msg:
            msg = "Réponse Supabase vide ou invalide. Vérifiez l’URL REST et la clé API dans secrets.yaml."
        raise HTTPException(status_code=500, detail=msg)
