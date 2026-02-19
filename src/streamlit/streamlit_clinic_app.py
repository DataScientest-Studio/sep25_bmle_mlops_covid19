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

# Style ergonomique (couleurs héritées du thème pour une bonne lisibilité)
st.markdown("""
<style>
    /* Conteneur principal */
    .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1200px; }
    
    /* Titre et sous-titre : pas de couleur forcée, le thème assure le contraste */
    h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
    .app-subtitle { font-size: 0.95rem; margin-bottom: 1.5rem; line-height: 1.4; opacity: 0.9; }
    
    /* Sous-titres de section */
    h2, h3 { font-size: 1.1rem; font-weight: 600; }
    
    /* Boutons */
    .stButton > button {
        width: 100%;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    /* Formulaire : bordures discrètes */
    [data-testid="stFileUploader"] { border-radius: 8px; }
    .stSelectbox > div, .stTextArea > div { border-radius: 8px; }
    
    /* Messages */
    [data-testid="stSuccess"], [data-testid="stError"], [data-testid="stInfo"] {
        border-radius: 8px;
    }
    
    /* Sidebar : pas de fond forcé, taille des titres */
    [data-testid="stSidebar"] h2 { font-size: 1rem; }
    
    /* Espacement vertical */
    hr { margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

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
    st.markdown('<p class="app-subtitle">Chargez une radiographie, lancez la prédiction, analysez le Grad-CAM puis enregistrez votre diagnostic.</p>', unsafe_allow_html=True)

    # --- État session ---
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None
    if "image_id" not in st.session_state:
        st.session_state.image_id = None

    result = st.session_state.prediction_result
    has_image = result and st.session_state.uploaded_image_bytes and result.get("heatmap_base64")

    # --- 1. Upload + prédiction ---
    st.subheader("1. Image et prédiction")
    col_upload, col_opts = st.columns([1, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Choisir une radiographie (PNG / JPG)",
            type=["png", "jpg", "jpeg"],
            key="upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            raw = uploaded.read()
            if raw:
                st.session_state.uploaded_image_bytes = raw
    with col_opts:
        store_prediction_s3 = st.checkbox(
            "Stocker l'image de prédiction dans S3",
            value=True,
            help="Upload de l'image source vers S3 (dossier predictions/) au moment de la prédiction.",
        )
        save_to_dataset = st.checkbox(
            "Enregistrer dans le dataset",
            value=True,
            help="Enregistre l'image dans la table images_dataset (Supabase).",
        )
        dataset_class_param = "0"
        if save_to_dataset:
            dataset_label_choice = st.selectbox(
            "Diagnostic à enregistrer (fait foi pour le dataset)",
            options=["Non-COVID", "COVID"],
            index=0,
            help="Votre diagnostic sera enregistré comme label ; la prédiction du modèle n’est pas utilisée.",
        )
        dataset_class_param = "1" if dataset_label_choice == "COVID" else "0"
        predict_clicked = st.button("Lancer la prédiction", type="primary", use_container_width=True)
    if st.session_state.uploaded_image_bytes and predict_clicked:
        with st.spinner("Prédiction en cours..."):
            try:
                files = {"file": ("image.png", st.session_state.uploaded_image_bytes, "image/png")}
                params = {
                    "save_to_dataset": "true" if save_to_dataset else "false",
                    "store_prediction_s3": "true" if store_prediction_s3 else "false",
                }
                if save_to_dataset:
                    params["dataset_class"] = dataset_class_param
                with httpx.Client(timeout=60.0) as client:
                    r = client.post(
                        PREDICT_GRADCAM_URL,
                        files=files,
                        params=params,
                    )
                r.raise_for_status()
                data = r.json()
                st.session_state.prediction_result = data
                st.session_state.image_id = data.get("image_id")
                st.rerun()
            except httpx.HTTPError as e:
                st.error(f"Erreur API: {e}")
                st.session_state.prediction_result = None
            except Exception as e:
                st.error(str(e))
                st.session_state.prediction_result = None

    # --- 2. Image + Grad-CAM (visible uniquement après prédiction) ---
    if has_image:
        st.divider()
        st.subheader("2. Résultat visuel (Grad-CAM)")
        image_size = result.get("image_size", [224, 224])
        heatmap_b64 = result.get("heatmap_base64")
        with st.sidebar:
            st.subheader("Affichage")
            zoom = st.slider(
                "Zoom",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="zoom",
            )
            alpha = st.slider(
                "Opacité Grad-CAM",
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
    else:
        with st.sidebar:
            st.caption("Réglages disponibles après la prédiction.")

    # --- 3. Feedback médecin (visible uniquement après prédiction) ---
    if result:
        st.divider()
        st.subheader("3. Votre diagnostic")
        pred_label = result.get("label", "—")
        probs = result.get("probabilities", [])
        st.markdown(f"**Prédiction du modèle :** {pred_label}")
        if probs and len(probs) == 2:
            st.caption(f"Non-COVID {probs[0]:.0%}  ·  COVID {probs[1]:.0%}")
        # Statut enregistrement dans images_dataset (quand « Enregistrer dans le dataset » était coché)
        if "image_saved" in result:
            if result.get("image_saved"):
                st.success(f"Image enregistrée dans la table images_dataset (id: {result.get('image_id', '—')}).")
            elif result.get("image_save_hint"):
                st.error("Enregistrement en base échoué: " + result["image_save_hint"])
        if "prediction_image_s3_saved" in result:
            if result.get("prediction_image_s3_saved") and result.get("prediction_image_s3_url"):
                st.success("Image de prédiction stockée dans S3.")
                st.code(result["prediction_image_s3_url"])
            elif result.get("prediction_image_s3_error"):
                st.warning("Stockage S3 de l'image de prédiction échoué: " + result["prediction_image_s3_error"])

        diagnostic = st.selectbox(
            "Diagnostic du médecin",
            options=DIAGNOSTIC_OPTIONS,
            index=0,
            key="diagnostic",
            help="Choisissez votre diagnostic dans la liste.",
        )
        comment = st.text_area(
            "Commentaire (optionnel)",
            height=80,
            key="comment",
            placeholder="Remarques, doute, contexte clinique…",
        )
        if st.button("Envoyer le feedback"):
            with st.spinner("Enregistrement..."):
                try:
                    data = {
                        "predicted_class": pred_label,
                        "diagnostic": diagnostic,
                        "comment": comment or "",
                    }
                    if st.session_state.get("image_id") is not None:
                        data["image_id"] = str(st.session_state.image_id)
                        r = httpx.post(FEEDBACK_URL, data=data, timeout=30.0)
                    else:
                        files = {"image": ("image.png", st.session_state.uploaded_image_bytes or b"", "image/png")}
                        with httpx.Client(timeout=30.0) as client:
                            r = client.post(FEEDBACK_URL, files=files, data=data)
                    r.raise_for_status()
                    if st.session_state.get("image_id") is not None:
                        st.success("Feedback enregistré. Le diagnostic a été appliqué à l’image dans images_dataset (class_type mis à jour).")
                    else:
                        st.success("Feedback enregistré en base.")
                except httpx.HTTPStatusError as e:
                    detail = ""
                    try:
                        body = e.response.json()
                        detail = body.get("detail", str(body))
                    except Exception:
                        detail = e.response.text or str(e)
                    st.error(f"Erreur ({e.response.status_code}): {detail}")
                except httpx.HTTPError as e:
                    st.error(f"Erreur envoi: {e}")
                except Exception as e:
                    st.error(str(e))
    else:
        if st.session_state.uploaded_image_bytes and not result:
            st.info("Cliquez sur « Lancer la prédiction » pour afficher le résultat et le formulaire de diagnostic.")
        elif not st.session_state.uploaded_image_bytes:
            st.info("Chargez une radiographie pour commencer.")


if __name__ == "__main__":
    main()
