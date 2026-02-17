import base64
import io
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2

from src.models.predict_model import predict_from_bytes, predict_from_db
from src.models.train_model_mlflow import train_model_mlflow

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
async def predict_with_gradcam_endpoint(
    file: UploadFile = File(...),
    save_to_dataset: bool = False,
    dataset_class: Optional[str] = None,
):
    """
    Prédiction + image + Grad-CAM.
    Si save_to_dataset=true: enregistre l'image dans images_dataset.
    Le label (class_type) vient du diagnostic : dataset_class "0" = Non-COVID, "1" = COVID.
    La prédiction du modèle n'est jamais utilisée comme label ; le diagnostic fait foi.
    """
    try:
        image_bytes = await file.read()
        result = predict_with_gradcam(image_bytes)
        heatmap = np.array(result.pop("heatmap"), dtype=np.float32)
        result["heatmap_base64"] = _heatmap_to_base64_png(heatmap)
        result["image_base64"] = base64.b64encode(image_bytes).decode("ascii")

        if save_to_dataset:
            secrets_path = _get_secrets_path()
            if secrets_path.exists():
                image_url = None
                storage_error = None
                storage_error_saved = None
                try:
                    s3_settings = S3Settings(str(secrets_path))
                    bucket_name, access_key, secret_key = s3_settings.s3_access
                    if bucket_name and access_key and secret_key:
                        ext = (Path(file.filename or "").suffix or ".png").lstrip(".").lower()
                        if ext not in ("png", "jpg", "jpeg"):
                            ext = "png"
                        image_url = upload_feedback_image(
                            bucket_name=bucket_name,
                            access_key=access_key,
                            secret_key=secret_key,
                            image_bytes=image_bytes,
                            s3_prefix="feedback",
                            extension=ext,
                        )
                except (KeyError, FileNotFoundError):
                    pass
                if not image_url:
                    db_settings = DatabaseSettings(str(secrets_path))
                    api_url, api_key = db_settings.database_url
                    base_url = api_url.replace("/rest/v1", "").rstrip("/")
                    ext = (Path(file.filename or "").suffix or ".png").lstrip(".").lower()
                    if ext not in ("png", "jpg", "jpeg"):
                        ext = "png"
                    # Chemin type dataset/COVID/images/COVID-{uuid}.png (comme les URLs existantes)
                    class_type = dataset_class if dataset_class in ("0", "1") else "0"
                    folder = "COVID" if class_type == "1" else "Non-COVID"
                    name = f"{folder}-{uuid.uuid4().hex}.{ext}"
                    object_path = f"dataset/{folder}/images/{name}"
                    image_url, storage_error = _supabase_storage_upload(
                        base_url, api_key, "images", object_path, image_bytes, f"image/{ext}"
                    )
                # Fallback si Storage a échoué : stocker l'image en data URL (à éviter : configurer Storage pour avoir de vraies URLs)
                storage_error_saved = storage_error  # garder pour l'afficher à l'utilisateur
                if not image_url and image_bytes and len(image_bytes) <= 1_000_000:
                    mime = "image/png" if ext == "png" else "image/jpeg"
                    b64 = base64.b64encode(image_bytes).decode("ascii")
                    image_url = f"data:{mime};base64,{b64}"
                if image_url:
                    db_settings = DatabaseSettings(str(secrets_path))
                    api_url, api_key = db_settings.database_url
                    db = DatabaseAccess(api_url=api_url, api_key=api_key)
                    class_type = dataset_class if dataset_class in ("0", "1") else "0"
                    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
                    img_row = {
                        "image_url": image_url,
                        "mask_url": None,
                        "class_type": class_type,
                        "injection_date": now,
                    }
                    try:
                        inserted = await db.insert("images_dataset", img_row, return_representation=True)
                        image_id = inserted.get("id") if isinstance(inserted, dict) else None
                        result["image_id"] = image_id
                        result["image_saved"] = True
                        if image_url.startswith("data:"):
                            result["image_stored_as_data_url"] = True
                            result["storage_error"] = storage_error_saved
                            result["image_save_hint"] = (
                                "Image enregistrée en data URL (Storage a échoué). "
                                "Pour avoir une vraie URL : Supabase → Storage → créer le bucket « images », le rendre public, "
                                "ajouter une policy INSERT. Erreur Storage : " + (storage_error_saved or "inconnue")
                            )
                    except httpx.HTTPStatusError as insert_err:
                        result["image_id"] = None
                        result["image_saved"] = False
                        err_body = (insert_err.response.text or "").strip()[:300]
                        result["image_save_hint"] = f"Insertion images_dataset échouée ({insert_err.response.status_code}): {err_body}"
                    except Exception as insert_err:
                        result["image_id"] = None
                        result["image_saved"] = False
                        result["image_save_hint"] = f"Insertion en base échouée: {str(insert_err)[:200]}"
                else:
                    result["image_id"] = None
                    result["image_saved"] = False
                    hint = (
                        "Créer le bucket 'images' dans Supabase Storage (Storage > New bucket), "
                        "le rendre public ou autoriser les uploads (Policies). Sinon configurer S3 dans secrets.yaml."
                    )
                    if storage_error:
                        hint += f" Erreur: {storage_error}"
                    if image_bytes and len(image_bytes) > 1_000_000:
                        hint += " Image trop grande pour le fallback data URL (>1 Mo)."
                    result["image_save_hint"] = hint
            else:
                result["image_id"] = None
                result["image_saved"] = False
                result["image_save_hint"] = "Fichier secrets.yaml introuvable (créer à partir de secrets.yaml.example)."

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
    image_id: Optional[str] = Form(None),
):
    """
    Enregistre l'image dans images_dataset (label = diagnostic), puis feedback avec img_id. Pas d’S3 requis (secrets.yaml).
    """
    try:
        secrets_path = _get_secrets_path()
        if not secrets_path.exists():
            raise HTTPException(status_code=500, detail="secrets.yaml not found")

        db_settings = DatabaseSettings(str(secrets_path))
        api_url, api_key = db_settings.database_url
        db = DatabaseAccess(api_url=api_url, api_key=api_key)

        def _diag_to_class(d: str) -> str:
            return "1" if d and d.strip().upper().startswith("COVID") and "Non" not in d else "0"

        class_type = _diag_to_class(diagnostic)
        final_image_id = None
        _img_id = int(image_id) if image_id and str(image_id).strip().isdigit() else None

        if _img_id is not None:
            # Mise à jour du diagnostic dans images_dataset (le diagnostic fait foi, pas la prédiction)
            updated = await db.update(
                "images_dataset",
                {"class_type": class_type},
                {"id": f"eq.{_img_id}"},
            )
            if isinstance(updated, list) and len(updated) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Le diagnostic n’a pas pu être appliqué à l’image (vérifier RLS sur images_dataset ou l’id).",
                )
            final_image_id = _img_id
        else:
            # Pas d'image_id : upload S3 ou Supabase Storage, puis insert images_dataset
            image_url = None
            if image:
                image_bytes = await image.read()
                if image_bytes:
                    ext = (Path(image.filename or "").suffix or ".png").lstrip(".").lower()
                    if ext not in ("png", "jpg", "jpeg"):
                        ext = "png"
                    try:
                        s3_settings = S3Settings(str(secrets_path))
                        bucket_name, access_key, secret_key = s3_settings.s3_access
                        if bucket_name and access_key and secret_key:
                            image_url = upload_feedback_image(
                                bucket_name=bucket_name,
                                access_key=access_key,
                                secret_key=secret_key,
                                image_bytes=image_bytes,
                                s3_prefix="feedback",
                                extension=ext,
                            )
                    except (KeyError, FileNotFoundError):
                        pass
                    if not image_url:
                        api_url, api_key = db_settings.database_url
                        base_url = api_url.replace("/rest/v1", "").rstrip("/")
                        folder = "COVID" if class_type == "1" else "Non-COVID"
                        name = f"{folder}-{uuid.uuid4().hex}.{ext}"
                        object_path = f"dataset/{folder}/images/{name}"
                        image_url, _ = _supabase_storage_upload(
                            base_url, api_key, "images", object_path, image_bytes, f"image/{ext}"
                        )
                    if not image_url and image_bytes and len(image_bytes) <= 1_000_000:
                        mime = "image/png" if ext == "png" else "image/jpeg"
                        image_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
            if image_url:
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
                img_row = {
                    "image_url": image_url,
                    "mask_url": None,
                    "class_type": class_type,
                    "injection_date": now,
                }
                try:
                    inserted = await db.insert("images_dataset", img_row, return_representation=True)
                    final_image_id = inserted.get("id") if isinstance(inserted, dict) else None
                except httpx.HTTPStatusError as e:
                    detail = (e.response.text or "").strip() or f"HTTP {e.response.status_code}"
                    raise HTTPException(
                        status_code=500,
                        detail=f"Insertion dans images_dataset impossible: {detail[:400]}",
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Insertion images_dataset: {str(e)[:200]}")
            # Si pas de S3 : on enregistre quand même le feedback (sans img_id)

        feedback_date = datetime.utcnow().isoformat()
        feedback_row = {
            "predicted_class": predicted_class,
            "diagnostic": diagnostic,
            "comment": comment,
            "feedback_date": feedback_date,
        }
        if final_image_id is not None:
            feedback_row["img_id"] = final_image_id
        await db.insert("feedback", feedback_row)
        return {"status": "ok", "image_id": final_image_id}
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
