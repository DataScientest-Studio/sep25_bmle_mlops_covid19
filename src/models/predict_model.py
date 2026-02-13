import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from keras import Model
import httpx
import cv2

from src.utils.db_dataset_utils import fetch_image_row_by_id
from src.utils.modele_utils import find_last_conv_layer
from src.utils.image_utils import overlay_gradcam_on_gray

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "src" / "models"

# Modèle par défaut du projet (à la racine du repo)
DEFAULT_MODEL_PATH = BASE_DIR / "EfficientNetv2B0_model_augmented_COVID_mask_full_new2_best.keras"

# Variable d'environnement pour surcharger le modèle (optionnel)
MODEL_PATH_ENV = "MODEL_PATH"


def get_model_path(model_path: Path | str | None = None) -> Path:
    """
    Détermine le chemin du modèle à charger:
    1. Argument model_path si fourni et fichier existe
    2. Variable d'environnement MODEL_PATH si définie et fichier existe
    3. DEFAULT_MODEL_PATH (EfficientNetv2B0...) si le fichier existe
    4. Dernier modèle (par date) dans src/models/*.keras
    """
    if model_path is not None:
        p = Path(model_path)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Model path not found: {p}")
    env_path = os.environ.get(MODEL_PATH_ENV)
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"MODEL_PATH not found: {p}")
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return find_latest_model()


def find_latest_model(models_dir: Path = MODELS_DIR) -> Path:
    candidates = list(models_dir.glob("*.keras"))
    if not candidates:
        raise FileNotFoundError("No trained model found in src/models.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(model_path: Path | str | None = None):
    path = get_model_path(model_path)
    return tf.keras.models.load_model(path)


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


def _make_gradcam_heatmap(
    model: tf.keras.Model,
    img_batch: tf.Tensor,
    pred_index: int | None = None,
) -> Tuple[np.ndarray, int]:
    """Calcule la heatmap Grad-CAM pour un batch d'une image."""
    last_conv_layer_name = find_last_conv_layer(model)
    grad_model = Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]).numpy())
        class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    return heatmap.astype(np.float32), pred_index


def predict_with_gradcam(
    image_bytes: bytes,
    model_path: Path | None = None,
) -> Dict[str, Any]:
    """
    Prédiction + heatmap Grad-CAM.
    Retourne le même format que predict_from_bytes plus 'heatmap' (np.ndarray 2D 0-1)
    et 'image_size' (height, width) pour le redimensionnement côté client.
    """
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
    heatmap, _ = _make_gradcam_heatmap(model, batch, pred_index=class_idx)

    return {
        "class_index": class_idx,
        "label": label,
        "probabilities": probs,
        "heatmap": heatmap.tolist(),
        "image_size": [int(height), int(width)],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/models/predict_model.py /path/to/image.png")
    result = predict_from_path(Path(sys.argv[1]))
    print(result)
