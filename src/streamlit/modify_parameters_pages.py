import asyncio
import streamlit as st
import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.database_utils import get_parameters, post_parameters
from src.utils.mlflow_utils import log_parameters_to_mlflow

st.set_page_config(page_title="Parameters Manager", layout="centered")
st.title("Training Parameters Manager")

# ==========================================================
# Chargement des paramètres actuels
# ==========================================================
with st.spinner("Loading current parameters..."):
    current_params = asyncio.run(get_parameters()) 

del current_params["validity_date"]
original_params = copy.deepcopy(current_params)

st.info("Tous les paramètres sont obligatoires. Au moins une modification est nécessaire pour pouvoir insérer.")

# ==========================================================
# Formulaire
# ==========================================================
with st.form("parameters_form"):

    cols = st.columns([1,1,1])
    with cols[0]:
        retraining_trigger_ratio = st.number_input(
            "Retraining trigger ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(current_params["retraining_trigger_ratio"]),
            step=0.05,
        )

        img_width = st.number_input(
            "Image width",
            min_value=1,
            value=int(current_params["img_width"]),
        )

        img_height = st.number_input(
            "Image height",
            min_value=1,
            value=int(current_params["img_height"]),
        )

        gray_mode = st.checkbox(
            "Gray mode",
            value=bool(current_params["gray_mode"]),
        )

        batch_size = st.selectbox(
            "Batch size",
            [16, 32, 64],
            index=[16, 32, 64].index(current_params["batch_size"]),
        )

        train_size = st.slider(
            "Train size",
            min_value=0.5,
            max_value=0.95,
            value=float(current_params["train_size"]),
            step=0.05,
        )

        random_state = st.number_input(
            "Random state",
            value=int(current_params["random_state"]),
        )
    with cols[1]:
        optimizer_name = st.selectbox(
            "Optimizer",
            ["adam", "sgd", "rmsprop"],
            index=["adam", "sgd", "rmsprop"].index(current_params["optimizer_name"]),
        )

        loss_cat = st.selectbox(
            "Loss function",
            ["categorical_crossentropy", "binary_crossentropy"],
            index=["categorical_crossentropy", "binary_crossentropy"].index(current_params["loss_cat"]),
        )

        metrics = st.text_input(
            "Metrics",
            value=current_params["metrics"],
        )

        es_patience = st.number_input(
            "EarlyStopping patience",
            min_value=1,
            value=int(current_params["es_patience"]),
        )

        es_min_delta = st.number_input(
            "EarlyStopping min_delta",
            min_value=0.0,
            value=float(current_params["es_min_delta"]),
            step=0.001,
        )

        es_mode = st.selectbox(
            "EarlyStopping mode",
            ["min", "max"],
            index=["min", "max"].index(current_params["es_mode"]),
        )

        es_monitor = st.text_input(
            "EarlyStopping monitor",
            value=current_params["es_monitor"],
        )
        
    with cols[2]:
        rlrop_patience = st.number_input(
            "ReduceLROnPlateau patience",
            min_value=1,
            value=int(current_params["rlrop_patience"]),
        )

        rlrop_monitor = st.text_input(
            "ReduceLROnPlateau monitor",
            value=current_params["rlrop_monitor"],
        )

        rlrop_min_delta = st.number_input(
            "ReduceLROnPlateau min_delta",
            min_value=0.0,
            value=float(current_params["rlrop_min_delta"]),
            step=0.001,
        )

        rlrop_factor = st.number_input(
            "ReduceLROnPlateau factor",
            min_value=0.0,
            max_value=1.0,
            value=float(current_params["rlrop_factor"]),
            step=0.05,
        )

        rlrop_cooldown = st.number_input(
            "ReduceLROnPlateau cooldown",
            min_value=0,
            value=int(current_params["rlrop_cooldown"]),
        )

        nb_layer_to_freeze = st.number_input(
            "Number of layers to freeze",
            min_value=0,
            value=int(current_params["nb_layer_to_freeze"]),
        )

    submitted = st.form_submit_button("Insérer")

# ==========================================================
# Validation + insertion
# ==========================================================
if submitted:

    new_params = {
        "retraining_trigger_ratio": retraining_trigger_ratio,
        "img_width": img_width,
        "img_height": img_height,
        "gray_mode": gray_mode,
        "batch_size": batch_size,
        "train_size": train_size,
        "random_state": random_state,
        "optimizer_name": optimizer_name,
        "loss_cat": loss_cat,
        "metrics": metrics,
        "es_patience": es_patience,
        "es_min_delta": es_min_delta,
        "es_mode": es_mode,
        "es_monitor": es_monitor,
        "rlrop_patience": rlrop_patience,
        "rlrop_monitor": rlrop_monitor,
        "rlrop_min_delta": rlrop_min_delta,
        "rlrop_factor": rlrop_factor,
        "rlrop_cooldown": rlrop_cooldown,
        "nb_layer_to_freeze": nb_layer_to_freeze,
    }
    
    error = False
    is_same = True
    
    # --- Vérification : tous renseignés ---
    if any(v is None or v == "" for v in new_params.values()):
        st.error("Tous les paramètres doivent être renseignés.")
        error = True

    # --- Vérification : au moins une modification ---
    for k in original_params:
        if str(original_params[k]) != str(new_params[k]):
            is_same = False
    
    if is_same:
        st.warning("Aucun paramètre modifié.")
        error = True
    
    if not error:
        asyncio.run(post_parameters(new_params))
        log_parameters_to_mlflow()
        st.success("Nouveaux paramètres bien insérés")
