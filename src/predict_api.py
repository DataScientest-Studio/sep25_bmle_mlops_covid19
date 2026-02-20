import base64
from datetime import datetime
from pathlib import Path
import sys
import uuid
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
import httpx
import numpy as np
from typing import Optional

from pydantic import BaseModel

sys.path.append(str(Path().resolve()))
from src.data.database_access import DatabaseAccess
from src.utils.s3_utils import upload_feedback_image
from src.models.predict_model import predict_from_bytes, predict_from_db, predict_with_gradcam
from src.settings import DatabaseSettings, S3Settings

app = FastAPI(title="COVID-19 Radiography API", version="1.0.0")

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
            print(f"{secrets_path = }")
            if secrets_path.exists():
                image_url = None
                storage_error = None
                storage_error_saved = None
                print("lets try!!")
                try:
                    s3_settings = S3Settings(str(secrets_path))
                    bucket_name, access_key, secret_key = s3_settings.s3_access
                    print(f"{bucket_name = }, {access_key = }, {secret_key =}")
                    if bucket_name and access_key and secret_key:
                        ext = (Path(file.filename or "").suffix or ".png").lstrip(".").lower()
                        print(f"{ext = }")
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
                except (KeyError, FileNotFoundError) as e:
                    print(e)
                print(f"{image_url = }")
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
                print(f"{storage_error = }")
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
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    img_row = {
                        "image_url": image_url,
                        "mask_url": None,
                        "class_type": class_type,
                        "injection_date": now,
                    }
                    print("lets try again!!")
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
    
    
def _heatmap_to_base64_png(heatmap: np.ndarray) -> str:
    """Encode heatmap (0-1 float) as PNG base64."""
    h_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", h_uint8)
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _get_secrets_path() -> Path:
    """Chemin vers le fichier de secrets (secrets.yaml, secret.yml, etc.)."""
    candidates = ("secrets.yaml", "secrets.yml", "secret.yaml", "secret.yml")
    for base in [Path.cwd(), Path(__file__).resolve().parents[2]]:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    return Path("secrets.yaml")

def _supabase_storage_upload(
    base_url: str,
    api_key: str,
    bucket: str,
    object_path: str,
    content: bytes,
    content_type: str = "image/png",
) -> tuple[Optional[str], Optional[str]]:
    """
    Upload un fichier vers Supabase Storage.
    Essaie multipart/form-data puis body binaire. Retourne (url_publique, None) ou (None, message_erreur).
    base_url: https://PROJECT.supabase.co (sans /rest/v1)
    """
    base_url = base_url.rstrip("/")
    url = f"{base_url}/storage/v1/object/{bucket}/{object_path}"
    auth_headers = {
        "Authorization": f"Bearer {api_key}",
        "apikey": api_key,
    }

    def _try_upload(use_multipart: bool) -> tuple[Optional[str], Optional[str]]:
        try:
            with httpx.Client(timeout=30.0) as client:
                if use_multipart:
                    filename = object_path.split("/")[-1]
                    r = client.post(url, files={"file": (filename, content, content_type)}, headers=auth_headers)
                else:
                    r = client.post(url, content=content, headers={**auth_headers, "Content-Type": content_type})
                r.raise_for_status()
                return f"{base_url}/storage/v1/object/public/{bucket}/{object_path}", None
        except httpx.HTTPStatusError as e:
            err_body = (e.response.text or "")[:250]
            return None, f"Storage {e.response.status_code}: {err_body}"
        except Exception as e:
            return None, str(e)[:200]

    # Essai 1 : multipart/form-data (format standard)
    url_or_none, err = _try_upload(use_multipart=True)
    if url_or_none:
        return url_or_none, None
    # Essai 2 : body binaire (certaines configs l'acceptent)
    url_or_none, err2 = _try_upload(use_multipart=False)
    if url_or_none:
        return url_or_none, None
    return None, err or err2
