"""
Application Streamlit clinique: chargement d'image, prédiction via API, GradCAM,
réglage opacité/zoom, et zone de feedback médecin (diagnostic + commentaire) avec envoi vers API.
"""
import base64
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import httpx

# Racine projet pour imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.image_utils import overlay_gradcam_on_gray

# --- Config ---
st.set_page_config(
    page_title="Aide au diagnostic COVID-19",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URL de l'API (env ou défaut)
API_BASE = os.environ.get("COVID_API_URL", "http://127.0.0.1:8000")
PREDICT_GRADCAM_URL = f"{API_BASE}/predict-with-gradcam"
FEEDBACK_URL = f"{API_BASE}/feedback"

# Options diagnostic médecin
DIAGNOSTIC_OPTIONS = ["COVID", "Non-COVID", "Indécis / à revoir"]


def decode_heatmap_from_base64(b64: str) -> np.ndarray:
    """Décode une image PNG base64 en heatmap 0-1."""
    raw = base64.b64decode(b64)
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid heatmap image")
    return (img.astype(np.float32) / 255.0).clip(0, 1)


def build_overlay(
    image_bytes: bytes,
    heatmap_b64: str,
    image_size: list,
    alpha: float,
) -> np.ndarray:
    """Construit l'image avec overlay GradCAM (opacité alpha)."""
    heatmap = decode_heatmap_from_base64(heatmap_b64)
    # Charger l'image originale
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = image_size[0], image_size[1]
    if gray.shape[0] != h or gray.shape[1] != w:
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return overlay_gradcam_on_gray(gray, heatmap, alpha=alpha)


def main():
    st.title("Aide au diagnostic – Radiographies COVID-19")
    st.markdown("Chargez une image, lancez la prédiction, consultez le Grad-CAM et envoyez votre feedback.")

    # --- État session ---
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None

    # --- Sidebar: zoom ---
    with st.sidebar:
        st.subheader("Affichage")
        st.slider(
            "Zoom",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="zoom",
        )
    zoom = st.session_state.get("zoom", 1.0)

    # --- Layout: gauche = image + opacité, droite = feedback ---
    col_image, col_feedback = st.columns([3, 1])

    with col_image:
        st.subheader("Image et Grad-CAM")
        uploaded = st.file_uploader(
            "Choisir une radiographie (PNG / JPG)",
            type=["png", "jpg", "jpeg"],
            key="upload",
        )
        if uploaded is not None:
            raw = uploaded.read()
            if raw:
                st.session_state.uploaded_image_bytes = raw

        if st.session_state.uploaded_image_bytes and st.button("Lancer la prédiction", type="primary"):
            with st.spinner("Appel de l'API..."):
                try:
                    files = {"file": ("image.png", st.session_state.uploaded_image_bytes, "image/png")}
                    with httpx.Client(timeout=60.0) as client:
                        r = client.post(PREDICT_GRADCAM_URL, files=files)
                    r.raise_for_status()
                    st.session_state.prediction_result = r.json()
                except httpx.HTTPError as e:
                    st.error(f"Erreur API: {e}")
                    st.session_state.prediction_result = None
                except Exception as e:
                    st.error(str(e))
                    st.session_state.prediction_result = None

        result = st.session_state.prediction_result
        if result and st.session_state.uploaded_image_bytes:
            image_size = result.get("image_size", [224, 224])
            heatmap_b64 = result.get("heatmap_base64")
            if not heatmap_b64:
                st.warning("Pas de heatmap dans la réponse API.")
            else:
                # Molette opacité (sous l'image)
                alpha = st.slider(
                    "Opacité du Grad-CAM",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.05,
                    key="alpha",
                )
                overlay = build_overlay(
                    st.session_state.uploaded_image_bytes,
                    heatmap_b64,
                    image_size,
                    alpha,
                )
                # Redimensionnement pour affichage zoom
                h, w = overlay.shape[:2]
                display_w = int(w * zoom)
                display_h = int(h * zoom)
                overlay_display = cv2.resize(
                    overlay,
                    (display_w, display_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                st.image(
                    cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                )

    with col_feedback:
        st.subheader("Feedback médecin")
        result = st.session_state.prediction_result
        if result:
            pred_label = result.get("label", "—")
            probs = result.get("probabilities", [])
            st.markdown(f"**Prédiction du modèle:** {pred_label}")
            if probs:
                if len(probs) == 2:
                    st.caption(f"Non-COVID: {probs[0]:.2%}  |  COVID: {probs[1]:.2%}")
                else:
                    st.caption(f"Probabilités: {probs}")

            diagnostic = st.selectbox(
                "Diagnostic du médecin",
                options=DIAGNOSTIC_OPTIONS,
                key="diagnostic",
            )
            comment = st.text_area(
                "Commentaire (optionnel)",
                height=100,
                key="comment",
                placeholder="Remarques, doute, contexte…",
            )
            if st.button("Envoyer le feedback"):
                if not st.session_state.uploaded_image_bytes:
                    st.warning("Aucune image chargée.")
                else:
                    with st.spinner("Envoi du feedback..."):
                        try:
                            files = {"image": ("image.png", st.session_state.uploaded_image_bytes, "image/png")}
                            data = {
                                "predicted_class": pred_label,
                                "diagnostic": diagnostic,
                                "comment": comment or "",
                            }
                            with httpx.Client(timeout=30.0) as client:
                                r = client.post(FEEDBACK_URL, files=files, data=data)
                            r.raise_for_status()
                            st.success("Feedback enregistré en base.")
                        except httpx.HTTPStatusError as e:
                            detail = ""
                            try:
                                body = e.response.json()
                                detail = body.get("detail", str(body))
                            except Exception:
                                detail = e.response.text or str(e)
                            st.error(f"Erreur envoi feedback ({e.response.status_code}): {detail}")
                        except httpx.HTTPError as e:
                            st.error(f"Erreur envoi feedback: {e}")
                        except Exception as e:
                            st.error(str(e))
        else:
            st.info("Lancez une prédiction pour afficher la prédiction et envoyer un feedback.")


if __name__ == "__main__":
    main()
