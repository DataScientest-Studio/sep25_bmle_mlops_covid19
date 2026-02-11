from pathlib import Path
from typing import Dict, Any
import numpy as np
import tensorflow as tf
import httpx

from src.utils.db_dataset_utils import fetch_image_row_by_id

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "src" / "models"


def find_latest_model(models_dir: Path = MODELS_DIR) -> Path:
    candidates = list(models_dir.glob("*.keras"))
    if not candidates:
        raise FileNotFoundError("No trained model found in src/models.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(model_path: Path | None = None):
    model_path = Path(model_path) if model_path else find_latest_model()
    return tf.keras.models.load_model(model_path)


def _prepare_image_from_bytes(image_bytes: bytes, image_size: tuple[int, int]) -> tf.Tensor:
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def predict_from_bytes(image_bytes: bytes, model_path: Path | None = None) -> Dict[str, Any]:
    model = load_model(model_path)
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height, width = input_shape[1], input_shape[2]

    img = _prepare_image_from_bytes(image_bytes, (height, width))
    batch = tf.expand_dims(img, axis=0)
    preds = model.predict(batch)

    if preds.ndim == 2:
        pred = preds[0]
        class_idx = int(np.argmax(pred))
        probs = [float(p) for p in pred.tolist()]
    else:
        class_idx = int(preds[0] >= 0.5)
        probs = [float(preds[0])]

    label = "COVID" if class_idx == 1 else "Non-COVID"
    return {
        "class_index": class_idx,
        "label": label,
        "probabilities": probs,
    }


def predict_from_path(image_path: Path, model_path: Path | None = None) -> Dict[str, Any]:
    image_bytes = Path(image_path).read_bytes()
    return predict_from_bytes(image_bytes, model_path=model_path)


async def predict_from_db(
    image_id: int,
    model_path: Path | None = None,
    table: str = "images_dataset",
) -> Dict[str, Any]:
    row = await fetch_image_row_by_id(image_id=image_id, table=table)
    image_url = row.get("image_url")
    if not image_url:
        raise ValueError(f"No image_url found for id={image_id}")

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(image_url)
        response.raise_for_status()
        image_bytes = response.content

    return predict_from_bytes(image_bytes, model_path=model_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/models/predict_model.py /path/to/image.png")
    result = predict_from_path(Path(sys.argv[1]))
    print(result)
