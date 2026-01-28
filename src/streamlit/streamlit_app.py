from pathlib import Path
import sys
import os
sys.path.append(str(Path(os.getcwd()).parent.parent))
from src.streamlit.streamlit_pages import  page_best_model, page_conclusion, page_data_analysis, page_demo_predict, page_end, page_models, page_preprocessing, page_project, page_title
import streamlit as st

# --- CONFIG ---
st.set_page_config(page_title="Présentation Projet - Streamlit", layout="wide")

def main():
    # --- Page config ---
    st.set_page_config(
        page_title="Présentation projet COVID-19",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    
    #MainMenu {visibility: hidden;}       /* menu hamburger en haut à droite */
    header {visibility: hidden;}          /* barre blanche */

    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("images/DST.png", width="content")  # chemin local, centré automatiquement
        st.write("")  # petit espace

        st.sidebar.title("Navigation")
        pages = [
            "Titre",
            "Le projet",
            "Analyse des données",
            "Préprocessing",
            "Modèles testés",
            "Meilleur modèle",
            "Démo prédiction + Grad-CAM",
            "Conclusion",
            "Fin"
        ]
        choice = st.sidebar.radio(label="", options=pages)
    
     # --------- IMPORTANT: si on est sur "Analyse des données", afficher
    # le titre + les sous-pages directement ici (immédiatement après la radio)
    if choice == "Analyse des données":

        # radio des sous-pages — créé ici, donc il sera rendu juste sous le titre ci-dessus
        subpage = st.sidebar.radio(
            "Analyse des données menu ",
            [
                "Aperçu général",
                "Distribution des classes",
                "Analyse statistique approfondie",
                "Analyse qualité des images"
            ],
            key="data_analysis_subpage"
        )
    elif choice == "Modèles testés":

        # radio des sous-pages — créé ici, donc il sera rendu juste sous le titre ci-dessus
        subpage = st.sidebar.radio(
            "Modèles testés menu ",
            [
                "Machine Learning",
                "Deep learning maison",
                "Transfert learning",
                "Optimisation"
            ],
            key="tested_models_subpage"
        )
    elif choice == "Meilleur modèle":

        # radio des sous-pages — créé ici, donc il sera rendu juste sous le titre ci-dessus
        subpage = st.sidebar.radio(
            "Meilleur modèle menu ",
            [
                "GradCam",
                "Présentation du meilleur modèle"
            ],
            key="best_model_subpage"
        )
    elif choice == "Conclusion":

        # radio des sous-pages — créé ici, donc il sera rendu juste sous le titre ci-dessus
        subpage = st.sidebar.radio(
            "Conclusion menu ",
            [
                "Problématique rencontrées",
                "Bilan",
                "Suite du projet"
            ],
            key="Conclusion_subpage"
        )
    else:
        subpage = None
        
        # --- Appel des pages ---
    if choice == "Titre":
        page_title()
        
    elif choice == "Le projet":
        page_project()
        
    elif choice == "Analyse des données":
        page_data_analysis(subpage)

    elif choice == "Préprocessing":
        page_preprocessing()

    elif choice == "Modèles testés":
        page_models(subpage)

    elif choice == "Meilleur modèle":
        page_best_model(subpage)
        
    elif choice == "Démo prédiction + Grad-CAM":
        page_demo_predict()
        
    elif choice == "Conclusion":
        page_conclusion(subpage)
        
    elif choice == "Fin":
        page_end()

if __name__ == "__main__":
    main()