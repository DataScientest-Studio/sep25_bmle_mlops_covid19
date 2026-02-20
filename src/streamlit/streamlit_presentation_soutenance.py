"""
Présentation de soutenance — Projet COVID-19 Radiography (MLOps).
Application Streamlit structurée pour jury de fin d'études / professeur MLOps.
"""
import sys
import os
from pathlib import Path

import streamlit as st

# Racine projet et import des schémas (images matplotlib, même dossier que ce script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STREAMLIT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_DIR))

from diagram_images import (
    diagram_architecture,
    diagram_chargement_dataset,
    diagram_flux_donnees,
    diagram_pipeline,
    diagram_endpoints,
    diagram_boucle_feedback,
)


def _show_diagram(png_bytes: bytes):
    """Affiche une image de schéma en grande taille (pleine largeur, fond clair)."""
    # Compatibilite Streamlit: gestion de plusieurs signatures selon version.
    try:
        st.image(png_bytes, width="stretch")
    except TypeError:
        try:
            st.image(png_bytes, use_container_width=True)
        except TypeError:
            try:
                st.image(png_bytes, use_column_width=True)
            except TypeError:
                st.image(png_bytes)

# --- Config page ---
st.set_page_config(
    page_title="Soutenance MLOps — Analyse de radiographies pulmonaires Covid-19",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Style présentation (thème clair forcé pour lisibilité et contraste) ---
st.markdown("""
<style>
    /* Forcer thème clair sur toute l'app (fond blanc, texte noir) */
    [data-testid="stAppViewContainer"] { background-color: #ffffff !important; }
    [data-testid="stAppViewContainer"] main { background-color: #ffffff !important; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; }
    section[data-testid="stSidebar"] * { color: #1a1a1a !important; }
    .block-container { background-color: #ffffff !important; color: #1a1a1a !important; }
    .stMarkdown, .stMarkdown p, h1, h2, h3, li, .stRadio label { color: #1a1a1a !important; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }
    h1 { font-size: 1.85rem; margin-bottom: 0.5rem; color: #1a1a1a !important; }
    h2 { font-size: 1.35rem; margin-top: 1.25rem; margin-bottom: 0.5rem; color: #1a1a1a !important; }
    h3 { font-size: 1.1rem; margin-top: 0.75rem; color: #1a1a1a !important; }
    ul { margin-left: 1.25rem; margin-bottom: 0.75rem; }
    /* Blocs « schéma » en texte : fond gris clair, texte noir, grande taille */
    .flow-diagram {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1a1a1a !important;
    }
    .flow-diagram .flow-step { margin: 0.6rem 0; color: #1a1a1a !important; }
    .flow-diagram strong { color: #0d47a1 !important; }
    /* Schémas (images) en grande taille */
    [data-testid="stImage"] img { width: 100% !important; min-width: 100% !important; max-width: 100% !important; height: auto !important; }
</style>
""", unsafe_allow_html=True)

# --- Liste des slides ---
SLIDES = [
    "Titre",
    "Introduction",
    "Le projet",
    "Architecture technique Globale",
    "Données & Dataset",
    "Modèle & Architecture",
    "Pipeline d'entraînement",
    "API & Exposition du modèle",
    "Boucle de feedback & MLOps",
    "Stack technique",
    "Application clinique",
    "Démo clinique (prédiction)",
    "Conclusion",
    "Perspectives",
    "Fin"
]


def slide_titre():

    st.markdown("## Classification de radiographies thoraciques COVID-19 — Pipeline MLOps")
    st.markdown("---")
    cols = st.columns([1,4,1])
    with cols[1]:
        st.image("src/streamlit/images/MLops covid.png" if os.path.exists("src/streamlit/images/MLops covid.png") else None)
    st.markdown("""
Projet réalisé dans le cadre de la formation, portant sur l'**aide au diagnostic** de signes radiologiques du COVID-19 à partir de radiographies thoraciques, avec une mise en œuvre complète des pratiques **MLOps** : pipeline reproductible, API, application clinique et boucle de feedback.
    """)


def slide_introduction():
    st.markdown("# Introduction")
    st.markdown("---")
    st.markdown("### Contexte et problématique")
    st.markdown("""
Ce projet s'inscrit dans le cadre de l'aide au diagnostic médical à partir d'images. La **problématique** est la suivante : comment assister un médecin dans l'identification de signes radiologiques évocateurs du COVID-19 sur des radiographies thoraciques ? Pour y répondre, on s'appuie sur un **dataset public** (COVID-19 Radiography Dataset, disponible sur Kaggle), qui contient des radiographies déjà étiquetées « COVID » ou « Non-COVID » par des experts. L'objectif n'est pas de remplacer le médecin, mais de proposer une **prédiction du modèle** et une **visualisation explicative** (Grad-CAM) pour alimenter la décision clinique.

Du point de vue **MLOps**, le projet va au-delà du simple entraînement d'un modèle : il met en place une **chaîne reproductible** (données, entraînement, sauvegarde), une **API** pour exposer le modèle en production, une **application clinique** pour les utilisateurs, et une **boucle de feedback** permettant d'enregistrer les diagnostics des médecins et d'enrichir progressivement le dataset pour de futurs ré-entraînements.
    """)
    st.markdown("### Plan de la présentation")
    st.markdown("""
La présentation est structurée en **trois parties**. **(1) Introduction et cadrage** : objectifs du projet, sources et structuration des données, choix du modèle et de l'architecture (EfficientNetV2, Grad-CAM). **(2) Mise en œuvre MLOps** : pipeline d'entraînement automatisé, API REST (FastAPI) et ses endpoints, application clinique Streamlit, boucle de feedback avec enregistrement des images et des diagnostics en base (Supabase), stack technique et livrables. **(3) Conclusion et perspectives** : bilan des objectifs atteints, puis pistes d'évolution (CI/CD, monitoring, A/B testing, sécurisation).

Cette structure permet de passer du **cadrage** et des **choix techniques** à la **réalisation concrète** (pipeline, API, interface, feedback), puis au **bilan** et aux **évolutions** possibles.
    """)
    st.markdown("### Architecture globale du projet")
    _show_diagram(diagram_architecture())  

def slide_objectifs():
    st.markdown("# Présentation de l'équipe")
    st.markdown("---")
    st.markdown("""
    - Marouane : ingénieur spatial reconverti\n
    - Kevin : 19 ans ingénieur full-stack\n
    - Narifidy : compétences transverses, vient de la partie Business
    """)
    st.markdown("# Objectifs du projet")
    st.markdown("---")
    st.markdown("""
Les objectifs ont été définis pour couvrir à la fois la partie **machine learning** (données, modèle, explicabilité) et la partie **MLOps** (pipeline reproductible, API, déploiement, boucle de feedback). Ils structurent la suite de la présentation. Voici les cinq objectifs principaux.
    """)
    st.markdown("""
**1. Automatisation du pipeline de données.**  
On ne part pas de fichiers déjà prêts : le pipeline prend en charge le téléchargement du dataset depuis Kaggle (via l’API et un fichier de credentials), puis la structuration en dossiers train et test, avec des sous-dossiers par classe (0 = Non-COVID, 1 = COVID). En production, les images injectées sont persistées dans **Supabase (table images_dataset)** et les fichiers image peuvent être stockés dans **S3/Backblaze**, ce qui permet un stockage durable des cas terrain. Comme la classe COVID est souvent minoritaire, on applique un **oversampling proportionnel à la différence de classe** pour rééquilibrer l’apprentissage.
    """)
    st.markdown("""
**2. Entraînement reproductible.**  
L’entraînement repose sur un modèle **EfficientNetV2-B0** (TensorFlow/Keras), avec augmentation de données, callbacks pour l’early stopping et la réduction du learning rate, et une **sauvegarde horodatée** des modèles. Ainsi, chaque run est traçable et reproductible.
    """)
    st.markdown("""
**3. Explicabilité.**  
En contexte médical, il est important de comprendre *où* le modèle regarde dans l’image. On utilise **Grad-CAM** (Gradient-weighted Class Activation Mapping) pour produire des cartes de chaleur qui mettent en évidence les régions de la radiographie les plus influentes pour la prédiction.
    """)
    st.markdown("""
**4. Déploiement et API.**  
Le modèle n’est pas seulement sauvegardé localement : il est exposé via une **API REST** (FastAPI). Celle-ci permet d’envoyer une image et d’obtenir une prédiction, une prédiction avec Grad-CAM, ou même de déclencher un entraînement à distance. Cela ouvre la voie au déploiement en production et à l’orchestration (CI/CD, cron, etc.).
    """)
    st.markdown("""
**5. Boucle de feedback.**  
Enfin, une **interface clinique** (Streamlit) permet à un médecin de charger une image, de voir la prédiction et le Grad-CAM, puis de saisir son **propre diagnostic** et un commentaire. Ces retours sont enregistrés dans une base (Supabase), et le diagnostic peut être utilisé pour mettre à jour le label de l’image dans le dataset. Ainsi, le système s’enrichit au fil de l’usage et peut servir à de futurs ré-entraînements.
    """)

def slide_architecture():
    st.markdown("# Architecture technique globale")
    st.markdown("---")
    st.image("src/streamlit/images/Architecture technique.png" if os.path.exists("src/streamlit/images/Architecture technique.png") else None)

def slide_donnees():
    st.markdown("# Data Engineering & Governance")
    st.markdown("---")

    st.markdown("### Acquisition & Versionnement des données")
    st.markdown("""
    Les données initiales proviennent du **COVID-19** Radiography Dataset (Kaggle).
    Le téléchargement est automatisé via l’API **Kaggle** avec gestion sécurisée des credentials.

    Chaque run d’entraînement est associé à une configuration précise,
    et les métadonnées dataset sont tracées dans **MLflow** afin d’assurer
    la **reproductibilité** complète des expérimentations.
    """)

    st.markdown("### Pipeline de préparation reproductible")
    st.markdown("""
    Les données sont automatiquement structurées en train/test.

    Nous intégrons :
    - un oversampling contrôlé de la classe minoritaire,
    - de l’augmentation d’images,
    - une normalisation cohérente.

    Ces étapes sont versionnées et tracées dans **MLflow** :
    paramètres, métriques, artefacts.
    On garantit ainsi qu’un modèle peut être reconstruit à partir d’un run donné.
    """)

    st.markdown("### Données en production & Feedback Loop")
    st.markdown("""
    En production, les images et labels sont stockés dans Supabase.

    Chaque prédiction validée via l’endpoint **/feedback**
    alimente la base et déclenche potentiellement un ré-entraînement planifié.

    On met ainsi en place une boucle d’amélioration continue :
    **Data → Model → Feedback → Data → **.
    """)

    _show_diagram(diagram_chargement_dataset())
    _show_diagram(diagram_flux_donnees())


def slide_modele():
    st.markdown("# Model Engineering & Tracking")
    st.markdown("---")

    st.markdown("### Architecture & Standardisation")
    st.markdown("""
    Le modèle repose sur **EfficientNetV2-B0**, encapsulé dans une classe dédiée.

    Le prétraitement et l’augmentation sont intégrés au graphe,
    ce qui garantit une cohérence totale entre entraînement et inférence.
    """)

    st.markdown("### Tracking avec MLflow")
    st.markdown("""
    Chaque entraînement est loggé dans **MLflow** :

    - hyperparamètres,
    - métriques (accuracy, recall),
    - artefacts (modèle, courbes, metrics complet),
    - version du dataset.

    **MLflow** agit ici comme :
    - registre d’expérimentations,
    - outil de comparaison de **runs**,
    - Model **Registry** pour promotion en production.
    """)

    st.markdown("### Gouvernance & Versioning")
    st.markdown("""
    Les modèles sont versionnés automatiquement.

    Un modèle validé peut être promu dans le **Model Registry MLflow**,
    ce qui permet :
    - rollback,
    - AB test,
    - staging → production,
    - traçabilité complète.
    """)


def slide_pipeline_entrainement():
    st.markdown("# Orchestration & Automatisation")
    st.markdown("---")

    st.markdown("### Orchestration avec Airflow")
    st.markdown("""
    Le pipeline d’entraînement est orchestré via **Apache Airflow**.

    Un **DAG** structure les étapes :
    1. Extraction / préparation des données
    2. Entraînement
    3. Logging MLflow
    4. Validation
    5. Promotion éventuelle du modèle

    Cela permet :
    - planification automatique,
    - gestion des dépendances,
    - monitoring des jobs.
    """)

    st.markdown("### Containerisation avec Docker")
    st.markdown("""
    L’ensemble des composants (API, entraînement, MLflow, Airflow)
    est containerisé avec **Docker**.

    Cela garantit :
    - portabilité,
    - reproductibilité environnementale,
    - déploiement simplifié sur tout environnement.
    """)

    st.markdown("### Déclenchement")
    st.markdown("""
    Le pipeline peut être déclenché :
    - automatiquement via Airflow,
    - manuellement via l’endpoint **/train**.

    On est donc sur un système **automatisé**,
    et non sur un entraînement manuel type notebook.
    """)
    
    _show_diagram(diagram_pipeline())


def slide_api():
    st.markdown("# Model Serving & Observability")
    st.markdown("---")

    st.markdown("### API & Serving")
    st.markdown("""
    Le modèle est exposé via **FastAPI**,
    containerisé et déployable derrière un reverse proxy.

    Les endpoints couvrent :
    - prédiction,
    - Grad-CAM,
    - ré-entraînement,
    - feedback.
    """)

    st.markdown("### Monitoring avec Prometheus & Grafana")
    st.markdown("""
    Les métriques applicatives sont exposées à **Prometheus** :

    - latence des prédictions,
    - nombre de requêtes,
    - metrics d'entrainement
    - statut du modèle actif.

    **Grafana** permet de visualiser ces métriques
    et de surveiller la santé globale du système.

    On peut ainsi détecter :
    - dégradation des performances,
    - anomalies de trafic,
    - dérive potentielle.
    """)

    st.markdown("### Observabilité complète")
    st.markdown("""
    Nous avons donc trois niveaux de supervision :

    - **MLflow** → performance modèle
    - **Airflow** → santé des pipelines
    - **Prometheus/Grafana** → monitoring runtime

    Cela transforme le projet en système **MLOps** complet,
    prêt pour un environnement de production réel.
    """)

    st.markdown("**Client → API → Model → Monitoring Stack**")
    _show_diagram(diagram_endpoints())


def slide_app_clinique():
    st.markdown("# Application clinique")
    st.markdown("---")
    st.markdown("### Rôle de l’interface")
    st.markdown("""
L’**application clinique** est une interface Streamlit destinée aux **médecins** (ou utilisateurs pilotes). Elle permet de charger une radiographie, d’obtenir la **prédiction du modèle** et la **carte Grad-CAM**, puis de saisir le **diagnostic réel** et un commentaire. Elle sert donc à la fois d’outil d’**aide au diagnostic** et de **point de collecte** pour la boucle de feedback MLOps.
    """)
    st.markdown("### Parcours utilisateur")
    st.markdown("""
**(1)** L’utilisateur **uploade** une radiographie (format PNG ou JPG). **(2)** Il peut cocher « Enregistrer dans le dataset » et choisir le **diagnostic à enregistrer** (COVID ou Non-COVID) : c’est ce label qui sera stocké en base, et non la prédiction du modèle — le **diagnostic fait foi**. **(3)** Il clique sur « Lancer la prédiction » : l’application envoie l’image à l’API **/predict-with-gradcam** et affiche la **prédiction** (classe et probabilités) ainsi que l’**image avec la heatmap Grad-CAM** superposée. **(4)** Dans la barre latérale, il peut ajuster le **zoom** et l’**opacité** de la heatmap pour mieux analyser les zones d’attention. **(5)** En bas de page, un **formulaire de feedback** lui permet de sélectionner son **diagnostic** (COVID, Non-COVID ou Indécis), d’ajouter un **commentaire** optionnel, et d’envoyer le tout à l’API **/feedback**. Les données sont alors enregistrées dans les tables Supabase (feedback et mise à jour de images_dataset si une image a été enregistrée).
    """)
    st.markdown("### Lien avec la chaîne MLOps")
    st.markdown("""
Les **diagnostics** saisis par les médecins sont persistés et peuvent servir à **ré-enrichir le dataset** et à **déclencher ou paramétrer** de futurs ré-entraînements. Les **images** associées peuvent être stockées dans **Supabase Storage** (URL publique) ou, en secours, en **data URL** dans la colonne image_url, et sont référencées dans la table **images_dataset** avec leur label (diagnostic). Ainsi, l’application n’est pas qu’un démo : elle alimente activement la base qui servira aux prochaines versions du modèle.
    """)


def slide_demo_clinique():
    st.markdown("# Démo clinique (prédiction)")
    st.markdown("---")
    st.markdown("### Scénario de démonstration au jury")
    st.markdown("""
La démonstration est pensée comme un parcours utilisateur réel et court : **(1)** chargement d’une radiographie thoracique ; **(2)** appel de l’endpoint `/predict-with-gradcam` ; **(3)** affichage du résultat (classe et probabilités) ; **(4)** visualisation de l’overlay Grad-CAM avec réglage de l’opacité/zoom ; **(5)** saisie du diagnostic médecin et envoi du feedback ; **(6)** vérification de la persistance dans `images_dataset` et `feedback`. Cette séquence montre simultanément la valeur clinique immédiate et la logique d’amélioration continue MLOps.
    """)
    st.markdown("### Points à verbaliser pendant la démo")
    st.markdown("""
Pendant la démo, il est important de rappeler que le système est une **aide à la décision** : le diagnostic final appartient au médecin. Quand un écart apparaît entre prédiction et diagnostic, c’est le **diagnostic médecin** qui est conservé comme vérité métier (`class_type`). L’image peut être stockée côté **S3/Storage**, puis référencée en base pour réutilisation dans les prochains entraînements. On montre ainsi le cycle complet : **prédire, expliquer, corriger, apprendre**.
    """)
    st.markdown("### Références conteneurs pour la démo")
    st.markdown("""
La démo s’appuie sur trois composants dockerisés : **`predict_api`** (prédiction et Grad-CAM), **`streamlit_clinic_app`** (interface), et, hors démonstration interactive, **`train_model_api`** pour les runs d’entraînement/retrain. Cette séparation rend le dispositif plus robuste et plus lisible pour une architecture de production.
    """)


def slide_boucle_feedback():
    st.markdown("# Boucle de feedback & MLOps")
    st.markdown("---")
    st.markdown("### Description du flux")
    st.markdown("""
La **boucle de feedback** relie l'usage en conditions réelles au ré-entraînement du modèle. **(1)** En conditions réelles, une **prédiction** est effectuée (via l'application clinique ou l'API) sur une nouvelle radiographie ; l'image peut être **enregistrée** dans la table **images_dataset** avec le **diagnostic** choisi par l'utilisateur (COVID ou Non-COVID), qui fait foi. **(2)** Le médecin envoie ensuite son **feedback** (diagnostic et commentaire) via **POST /feedback** ; si un **image_id** est fourni, le champ **class_type** dans **images_dataset** est **mis à jour** selon ce diagnostic — le diagnostic écrase le label initial, ce qui permet de corriger les erreurs ou d'affiner les cas ambigus. **(3)** La base **images_dataset** contient alors des images et des labels issus du **terrain** ; elle est prête à servir de source pour un **prochain entraînement** en appelant **POST /train** avec `data_source=db`, ce qui charge ces données pour entraîner une nouvelle version du modèle. Le cycle prédiction → feedback → mise à jour → ré-entraînement peut ainsi se répéter pour faire évoluer le système.
    """)
    st.markdown("**Boucle de feedback**")
    _show_diagram(diagram_boucle_feedback())
    st.markdown("### Brique technique")
    st.markdown("""
Côté infrastructure : **Supabase** fournit la base PostgreSQL (accès via REST API), le **Storage** pour héberger les fichiers image (URLs publiques), et les politiques **RLS** ; l'API utilise la clé **service_role** pour les opérations côté serveur. Le module **DatabaseAccess** crée un **client HTTP par requête** (sans client partagé), ce qui évite l'erreur « client has been closed » ; les insert et update gèrent les champs **injection_date** et **class_type**. En **fallback**, si le Storage est indisponible, l'image peut être stockée en **data URL** dans la colonne **image_url**, ce qui préserve la traçabilité tout en permettant de continuer à enregistrer les cas.
    """)


def slide_stack():
    st.markdown("# Stack technique")
    st.markdown("---")
    st.markdown("""
Le projet repose sur une **stack** cohérente côté backend, ML, données et interface. Voici les technologies utilisées et les livrables livrés.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Backend & ML")
        st.markdown("""
- **Python 3**
- **TensorFlow / Keras** (EfficientNetV2)
- **FastAPI** (API REST)
- **httpx** (client HTTP async)
- **OpenCV, NumPy** (images, Grad-CAM)
- **MLflow** (orchestration/tracking d'entraînement)
        """)
    with col2:
        st.markdown("### Données & Front")
        st.markdown("""
- **Supabase** (PostgreSQL, Storage, REST)
- **Backblaze / S3** (stockage objets image)
- **Streamlit** (interface clinique + présentation)
- **Docker** (`train_model_api`, `predict_api`, `streamlit_clinic_app`)
- **YAML** (secrets, config)
        """)
    st.markdown("### Livrables")
    st.markdown("""
Le projet livre un **pipeline d'entraînement** reproductible (données, fit, save, evaluate, gradcam), une **API** déployable exposant prédiction, Grad-CAM, entraînement et feedback, une **application clinique** Streamlit pour l'aide au diagnostic et la collecte du feedback médecin, et une **boucle de feedback** intégrée à la base Supabase (images_dataset, feedback) pour alimenter de futurs ré-entraînements. L'ensemble est versionné, documenté et prêt à être étendu (CI/CD, monitoring, A/B testing).
    """)


def slide_conclusion():
    st.markdown("# Conclusion")
    st.markdown("---")
    st.markdown("### Bilan du projet")
    st.markdown("""
Le projet atteint les objectifs fixés en couvrant l’ensemble de la chaîne MLOps. Sur le **pipeline** : un enchaînement **données → entraînement → sauvegarde** est automatisé et reproductible (méthode `execute`, chargement depuis fichier Kaggle ou depuis la base Supabase, oversampling, callbacks). Sur l’**exposition** : le modèle est servi via une **API REST** (FastAPI) avec prédiction, Grad-CAM et entraînement déclenchable à distance. Sur l’**interface** : une **application clinique** Streamlit permet l’aide au diagnostic et la collecte du feedback médecin (diagnostic et commentaire). Sur la **boucle de données** : une **boucle MLOps** est en place — les images et les diagnostics sont enregistrés en base (tables images_dataset et feedback), ce qui permet d’alimenter de futurs ré-entraînements et de faire évoluer le système en continu.

En résumé, le livrable inclut un pipeline reproductible, une API déployable, une application utilisable par les cliniciens et un mécanisme de feedback intégré à la base, conformément aux objectifs de la soutenance.
    """)
    st.markdown("### Transition vers les perspectives")
    st.markdown("""
La suite logique du projet consiste à renforcer l’**automatisation** (CI/CD), le **monitoring** en production et la **sécurisation** des accès. Les pistes détaillées sont présentées dans la section **Perspectives**.
    """)


def slide_perspectives():
    st.markdown("# Perspectives")
    st.markdown("---")
    st.markdown("### Pistes d’évolution envisagées")
    st.markdown("""
Plusieurs évolutions permettraient d’aller plus loin en production et en MLOps.
    """)
    st.markdown("**CI/CD et ré-entraînement.** Déclencher l’entraînement automatiquement selon des critères définis : détection de **drift** des données (changement de distribution des entrées), atteinte d’un **volume de feedback** suffisant, ou exécution selon un **planning** (par exemple hebdomadaire). L’API **POST /train** est déjà prête à être appelée par un pipeline CI/CD (GitHub Actions, GitLab CI, Jenkins, etc.) pour produire et versionner de nouveaux modèles.")
    st.markdown("**Monitoring en production.** Mettre en place un suivi des **métriques opérationnelles** : latence des prédictions, taux d’erreur, distribution des classes prédites. En cas de dégradation (dérive des performances, pics de latence), des alertes permettraient d’investiguer ou de déclencher un ré-entraînement. Des outils comme **MLflow** ou des dashboards dédiés peuvent centraliser ces métriques.")
    st.markdown("**A/B testing.** Comparer en production **plusieurs versions** du modèle (par exemple ancien vs nouveau après ré-entraînement) sur une fraction du trafic, afin de valider les gains en précision ou en robustesse avant une bascule complète. Cela nécessite un routage côté API ou un mécanisme de feature flags.")
    st.markdown("**Sécurisation.** Renforcer la **sécurité** de l’API et de l’application clinique : **authentification** (tokens JWT, OAuth2), **audit** des accès et des appels, **contrôle des permissions** (rôles utilisateur, restriction des endpoints sensibles comme /train). Le fichier **secrets.yaml** et les variables d’environnement restent la base pour les credentials ; une intégration avec un coffre-fort (Vault, secrets cloud) est une piste pour la production.")
    st.image("src/streamlit/images/Life_Cycle.png" if os.path.exists("src/streamlit/images/Life_Cycle.png") else None)
    
def slide_fin():
    cols = st.columns([1,4,1])
    with cols[1]:
        st.image("src/streamlit/images/thanks.png" if os.path.exists("src/streamlit/images/thanks.png") else None)


# --- Dispatcher ---
RENDER = {
    "Titre": slide_titre,
    "Introduction": slide_introduction,
    "Le projet": slide_objectifs,
    "Architecture technique Globale":slide_architecture,
    "Données & Dataset": slide_donnees,
    "Modèle & Architecture": slide_modele,
    "Pipeline d'entraînement": slide_pipeline_entrainement,
    "API & Exposition du modèle": slide_api,
    "Application clinique": slide_app_clinique,
    "Démo clinique (prédiction)": slide_demo_clinique,
    "Boucle de feedback & MLOps": slide_boucle_feedback,
    "Stack technique": slide_stack,
    "Conclusion": slide_conclusion,
    "Perspectives": slide_perspectives,
    "Fin":slide_fin
}


def main():
    st.sidebar.image("src/streamlit/images/logo liora.png", width="stretch")
    st.sidebar.title("Soutenance MLOps")
    st.sidebar.markdown("Analyse de radiographies pulmonaires Covid-19")
    st.sidebar.markdown("---")
    choice = st.sidebar.radio(
        "Navigation",
        SLIDES,
        index=0,
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Sélectionnez une section pour naviguer dans la présentation.")

    RENDER[choice]()


if __name__ == "__main__":
    main()
