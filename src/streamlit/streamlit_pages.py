# app.py
"""
Streamlit app de présentation de projet (image classification) avec :
- Titre
- Présentation de l'équipe
- Présentation du projet
- Analyse des données (graphiques)
- Preprocessing + explication des choix
- Les différents modèles entraînés + stats
- Le meilleur modèle : justification
- Démo de predict sur N cas choisis aléatoirement (avec Grad-CAM)
- Analyse des résultats & "comment aurions-nous pu faire mieux"
- Conclusion

Dépendances (pip):
streamlit, pandas, numpy, matplotlib, pillow, scikit-learn, tensorflow or torch (selon tes modèles)
(ajoute: seaborn si tu veux, mais c'est optionnel)

Adapte DATA_PATH, MODEL_DIR, et la méthode grad_cam_apply() à ton code / modèle existant.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.tensor.models_to_test import Base
from src.utils.data_utils import select_sample_data

# --- CONFIG ---
#st.set_page_config(page_title="Présentation Projet - Streamlit", layout="wide")
DATA_PATH = "csv/dataset.csv"           # csv with at least: image_path,label (adapt to your format)
IMAGE_ROOT = "data/images"               # folder for images if image_path are relative
MODEL_DIR = "models"                     # folder where models & meta (json/csv) are stored
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(os.getcwd())  # racine du projet
DATASET_ROOT = PROJECT_ROOT.parent.parent / "data" # racine du dataset
DATASET_REP  = DATASET_ROOT / "COVID-19_Radiography_Dataset_init" # Répertoire contenant le dataset
MODELS_FOLDER = PROJECT_ROOT.parent / "models"

#bouchons
counts_image = pd.DataFrame()
counts_image["class"] = ["COVID", "Normal", "Viral_Pneumonia","Lung Opacity"]
counts_image["count"] = [3616, 10192, 1345, 6012]

imageEDA2 = pd.read_csv("csv/imageEDA2.csv", sep=";", index_col=0)
df_opacity = pd.read_csv("csv/opacity_analysis_results.csv", sep=',')

metrics_XGBoost = pd.read_csv("csv/metrics_XGBoost.csv", sep=";", index_col=0, dtype=str)
metrics_CNN_Custom = pd.read_csv("csv/metrics_CNN_Custom.csv", sep=";", index_col=0, dtype=str)
metrics_DenseNet121 = pd.read_csv("csv/metrics_DenseNet121.csv", sep=";", index_col=0, dtype=str)
metrics_EfficientNetV2B0 = pd.read_csv("csv/metrics_EfficientNetV2B0.csv", sep=";", index_col=0, dtype=str)
metrics_EfficientNetV2B3 = pd.read_csv("csv/metrics_EfficientNetV2B3.csv", sep=";", index_col=0, dtype=str)
metrics_EfficientNetB5 = pd.read_csv("csv/metrics_EfficientNetB5.csv", sep=";", index_col=0, dtype=str)
metrics_LeNet = pd.read_csv("csv/metrics_LeNet.csv", sep=";", index_col=0, dtype=str)
metrics_RestNet18 = pd.read_csv("csv/metrics_RestNet18.csv", sep=";", index_col=0, dtype=str)
metrics_VGG19 = pd.read_csv("csv/metrics_VGG19.csv", sep=";", index_col=0, dtype=str)

heatmap_XGBoost = pd.read_csv("csv/heatmap_XGBoost.csv", sep=";", index_col=0, dtype=str)
heatmap_LeNet = pd.read_csv("csv/heatmap_LeNet.csv", sep=";", index_col=0, dtype=str)
heatmap_CNN_Custom = pd.read_csv("csv/heatmap_CNN_Custom.csv", sep=";", index_col=0, dtype=str)
heatmap_DenseNet121 = pd.read_csv("csv/heatmap_DenseNet121.csv", sep=";", index_col=0, dtype=str)
heatmap_EfficientNetB5 = pd.read_csv("csv/heatmap_EfficientNetB5.csv", sep=";", index_col=0, dtype=str)
heatmap_EfficientNetV2B0 = pd.read_csv("csv/heatmap_EfficientNetV2B0.csv", sep=";", index_col=0, dtype=str)
heatmap_EfficientNetV2B3 = pd.read_csv("csv/heatmap_EfficientNetV2B3.csv", sep=";", index_col=0, dtype=str)
heatmap_RestNet18 = pd.read_csv("csv/heatmap_RestNet18.csv", sep=";", index_col=0, dtype=str)
heatmap_VGG19 = pd.read_csv("csv/heatmap_VGG19.csv", sep=";", index_col=0, dtype=str)

metrics_DenseNet121_opt = pd.read_csv("csv/metrics_DenseNet121_opt.csv", sep=";", index_col=0, dtype=str)
metrics_EfficientNetV2B0_opt = pd.read_csv("csv/metrics_EfficientNetV2B0_opt.csv", sep=";", index_col=0, dtype=str)
metrics_EfficientNetV2B3_opt = pd.read_csv("csv/metrics_EfficientNetV2B3_opt.csv", sep=";", index_col=0, dtype=str)
heatmap_DenseNet121_opt = pd.read_csv("csv/heatmap_DenseNet121_opt.csv", sep=";", index_col=0, dtype=str)
heatmap_EfficientNetV2B0_opt = pd.read_csv("csv/heatmap_EfficientNetV2B0_opt.csv", sep=";", index_col=0, dtype=str)
heatmap_EfficientNetV2B3_opt = pd.read_csv("csv/heatmap_EfficientNetV2B3_opt.csv", sep=";", index_col=0, dtype=str)

MODEL_DICT = {"EfficientNetv2B0":("EfficientNetv2B0_model_augmented_COVID_mask_full_new2_best", (299,299)),
              "EfficientNetv2B3":("EfficientNetv2B3_model_augmented_COVID_mask_full_new1", (299,299)),
              "DenseNet121":("DenseNet121_model_augmented_COVID_mask_full_new1", (224,224))}

CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

model_loaded = []
    
# ---------------------------------------------------------------------
# Pages (fonctions)
# ---------------------------------------------------------------------
def page_title():
    st.title("Analyse de radiographies pulmonaires Covid-19", text_alignment="center")
    cols = st.columns([1,4,1])
    with cols[1]:
        st.image("images/Projet covid.png" if os.path.exists("images/Projet covid.png") else None)
        
def page_project():
    
    st.title("Le projet")
    
    st.header("Présentation de l'équipe")
    st.markdown("""
    - Marouane : ingénieur spatial reconverti\n
    - Kevin : 19 ans ingénieur full-stack\n
    - Narifidy : compétences transverses, vient de la partie Business
    """)
    
    st.header("Présentation du projet")
    st.markdown("""
    La pandémie de Covid-19 a mis une forte pression sur les systèmes de santé, notamment sur la capacité à diagnostiquer rapidement les patients.
    Les tests PCR, bien que fiables, peuvent être coûteux, lents, ou indisponibles dans certaines situations.
    Dans ce contexte, l’analyse automatisée des radiographies pulmonaires s’est imposée comme une piste prometteuse : les images médicales sont déjà utilisées en routine et facilement accessibles dans la plupart des hôpitaux.
    L’objectif du projet est donc de développer un modèle de deep learning capable de classer des radiographies pulmonaires entre :
    - COVID-19
    - pneumonie virale
    - autres maladies pulmonaires
    - normale (pas de maladie)""")
    st.markdown("""Un tel modèle pourrait aider les professionnels de santé à accélérer la prise de décision clinique, surtout en situation d’urgence ou dans des régions à faible accès aux tests diagnostiques.""")

@st.cache_data
def page_data_analysis(subpage):
  
    st.title("Analyse des données", text_alignment="center")
  
    if subpage == "Aperçu général":
        st.header("Description des données")
        st.markdown("""
        Le projet s’appuie sur le dataset COVID-19 Radiography Database, contenant des radiographies pulmonaires annotées (Normal, Lung Opacity, Viral Pneumonia, COVID).
        Les données sont librement disponibles. Elles sont publiées sur Kaggle par Tawsifur Rahman et collaborateurs.
        Le dataset contient 21 165 images PNG réparties en quatre classes, un fichier metadata.csv, et des masks de segmentation des poumons.
        Répartition :
        - Normal : 10 192
            - Mask : 10 192
        - Lung Opacity : 6 012
            - Mask : 6 012
        - COVID : 3 616
            - Mask : 3 616
        - Viral Pneumonia : 1 345
            - Mask : 1 345
        """)
        st.header("Pertinence et limitation des variables")
        st.markdown("""
        Nos variables cibles vont être toutes les radiographies des différentes maladies ainsi que leurs masques (mask). 
        Le dataset comporte beaucoup d'images pour de l’imagerie médicale. 
        En revanche, le dataset est très déséquilibré. Les sources sont multiples donc la qualité des images est variable.
        De plus les radiographies sont de dimensions 299x299, alors que les mask associés sont de dimension 256x256.
        """)
      
    elif subpage == "Distribution des classes":
        st.header("Répartition des classes")
        cols = st.columns([1,4,1])
        with cols[1]:
            fig = plt.figure(figsize = (2, 2))
            
            # On calcule le décalage du secteur "Python", via une liste en intention.
            explode = [0.15 if c == "COVID" else 0 for c in counts_image["class"]]

            # Création du graphique circulaire
            plt.pie(counts_image["count"], 
                    labels=counts_image["class"],
                    explode=explode,               # Les décalages pour faire ressortir Python 
                    autopct='%.2f %%',             # Le format utilisé pour l'affichage des valeurs (2 chiffres après la vigule)
                    shadow={'ox': -0.01, 'edgecolor': 'none', 'shade': 0.9}, #forme de l'ombre
                    startangle=90,                 # L'angle de démarrage pour afficher la première valeur
                    counterclock=False)            # Dans quel sens on affiche les données

            st.pyplot(fig)
        st.markdown("""On constate un trés net déséquilibre de la classe COVID par rapport aux autres classes""")
    
    elif subpage == "Analyse statistique approfondie":
        st.header("Analyse statistique approfondie des distributions d’intensité ")

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        sns.kdeplot(data = imageEDA2, x = 'mean', hue = 'diag', ax=axes[0])
        axes[0].set_title('Distribution des moyennes des couleurs par classes\n', fontsize = 12);
        sns.kdeplot(data = imageEDA2, x = 'max', hue = 'diag', ax=axes[1])
        axes[1].set_title('\nDistribution des maximums des couleurs par classes\n', fontsize = 12);
        sns.kdeplot(data = imageEDA2, x = 'min', hue = 'diag', ax=axes[2])
        axes[2].set_title('\nDistribution des minimums des couleurs par classes\n', fontsize = 12);

        st.pyplot(fig)
      
        st.markdown("""
        Les graphiques montrent la distribution des valeurs moyennes de pixels pour chaque classe. 
        On observe que certaines classes présentent une distribution unimodale, tandis que d’autres, comme Normal, présentent une tendance bimodale, indiquant la présence de deux sous-populations d’images. 
        - Cela suggère des conditions d’acquisition différentes (machines, exposition) créant une hétérogénéité au sein même des classes.
        - Cette observation justifie pleinement les étapes de pré-processing (normalisation, homogénéisation de l’éclairage) pour éviter que le modèle n’apprenne des artéfacts techniques plutôt que des caractéristiques médicales.
        """)
      
    elif subpage == "Analyse qualité des images":
        st.header("Recherche d'artefacts")
        cols = st.columns([1,4,1])
        with cols[1]:
            st.image("images/artefacts.png" if os.path.exists("images/artefacts.png") else None)
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        st.subheader("Synthèse générale")
        st.markdown("""
        Le système présenté met en œuvre une stratégie robuste et pragmatique de détection d’artefacts dans les radiographies pulmonaires, fondée sur une combinaison raisonnée d’analyses fréquentielle et spatiale.<br>
        L’objectif n’est pas esthétique mais opérationnel : identifier automatiquement les perturbations artificielles susceptibles de biaiser les modèles de diagnostic assisté par apprentissage automatique.<br>
        L’approche est clairement pensée pour un contexte industriel ou clinique, avec des compromis maîtrisés entre précision, rappel et coût computationnel.
        """, unsafe_allow_html=True)
        st.subheader("Analyse fréquentielle : détection structurelle globale")
        st.markdown("""
        L’approche fréquentielle repose sur un principe fondamental du traitement du signal : toute structure spatiale régulière génère une signature identifiable dans le domaine des fréquences.<br>
        En exploitant la FFT 2D, le système isole efficacement les patterns répétitifs artificiels (grilles, textures de compression, bordures uniformes), absents ou très atténués dans les structures anatomiques naturelles.<br> 
        La détection s’appuie sur quatre indicateurs complémentaires — densité de pics spectraux, uniformité des bordures, variance locale élevée (annotations) et ratio d’énergie haute fréquence — dont la pondération hiérarchise l’information la plus discriminante.<br> 
        Cette stratégie multi-indicateurs limite les faux positifs et confère à la méthode une forte robustesse face aux variations d’intensité inter-images. 
        """, unsafe_allow_html=True)
        st.subheader("Analyse spatiale : détection locale et géométrique")
        st.markdown("""
        L’analyse spatiale cible les artefacts là où ils vivent vraiment : dans le pixel. Elle exploite deux leviers visuels majeurs : le contraste anormal et les contours géométriques non anatomiques.<br> 
        L’utilisation d’un colormap amplificateur suivie d’une segmentation dans l’espace HSV permet de faire ressortir les zones artificielles que l’œil humain détecte intuitivement mais que les modèles ignorent souvent.<br> 
        En parallèle, la détection de contours adaptative (Canny à seuils dynamiques) capture les structures linéaires ou rectilignes typiques des annotations et bordures.<br> 
        L’intersection stricte de ces deux masques agit comme un filtre de crédibilité : un artefact doit être visuellement suspect et structurellement cohérent pour être retenu.
        """, unsafe_allow_html=True)
        st.subheader("Approche hybride : stratégie orientée rappel")
        st.markdown("""
        L’approche hybride assume un positionnement clair : ne rien rater. En combinant les décisions fréquentielle et spatiale via un OU logique, le système maximise le rappel tout en conservant une explicabilité forte.<br> 
        Cette complémentarité est pertinente : les artefacts globaux et répétitifs sont captés par la FFT, tandis que les anomalies locales et non périodiques sont mieux détectées spatialement.<br> 
        Ce choix augmente mécaniquement le risque de faux positifs, mais il est parfaitement cohérent dans un pipeline de pré-filtrage ou de contrôle qualité en amont d’un modèle clinique.
        """, unsafe_allow_html=True)
        st.subheader("Fondements méthodologiques clés")
        st.markdown("""
        Le dispositif repose sur des principes solides et bien maîtrisés : masquage pulmonaire strict pour réduire le bruit anatomique hors zone d’intérêt, seuils adaptatifs basés sur les percentiles pour gérer l’hétérogénéité des acquisitions, validation multi-indicateurs pour éviter les décisions fragiles, et nettoyage morphologique pour produire des masques exploitables.<br> 
        L’analyse par composantes connexes, menée séparément par poumon, renforce encore la cohérence anatomique des détections.
        """, unsafe_allow_html=True)
        st.subheader("Performance et viabilité opérationnelle")
        st.markdown("""
        D’un point de vue algorithmique, la complexité est dominée par la FFT mais reste parfaitement compatible avec un usage à grande échelle.<br> 
        Les temps d’exécution observés confirment la viabilité industrielle de la solution, y compris sur des jeux de données de plusieurs dizaines de milliers d’images.<br> 
        Le système est donc prêt pour une intégration en production, sans dépendance lourde ni latence critique.
        """, unsafe_allow_html=True)
        st.subheader("Positionnement et perspectives")
        st.markdown("""
        Ce travail propose une solution explicable, modulaire et immédiatement exploitable, là où beaucoup de pipelines misent aveuglément sur le deep learning.<br> 
        Les pistes d’évolution — segmentation sémantique, calibration automatique, accélération GPU ou hybridation avec des CNN spécialisés — sont naturelles et crédibles.
        """, unsafe_allow_html=True)
      
        st.header("Cadrage et contrastes")

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        ax1 = axes[0, 0]
        for cls in CLASSES:
            cls_data = df_opacity[df_opacity['class'] == cls]['min']
            ax1.hist(cls_data, alpha=0.6, label=cls, bins=50)
        ax1.set_xlabel('Valeur Min')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution de la Valeur Min par Classe')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Distribution du Max par classe
        ax2 = axes[0, 1]
        for cls in CLASSES:
            cls_data = df_opacity[df_opacity['class'] == cls]['max']
            ax2.hist(cls_data, alpha=0.6, label=cls, bins=50)
        ax2.set_xlabel('Valeur Max')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution de la Valeur Max par Classe')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Boxplot Min par classe
        ax3 = axes[1, 0]
        df_opacity.boxplot(column='min', by='class', ax=ax3)
        ax3.set_xlabel('Classe')
        ax3.set_ylabel('Valeur Min')
        ax3.set_title('Boxplot de la Valeur Min par Classe')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)

        # 4. Boxplot Max par classe
        ax4 = axes[1, 1]
        df_opacity.boxplot(column='max', by='class', ax=ax4)
        ax4.set_xlabel('Classe')
        ax4.set_ylabel('Valeur Max')
        ax4.set_title('Boxplot de la Valeur Max par Classe')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
      
        st.markdown("""
    Ce graphique analyse les valeurs extrêmes d'opacité :
    - Distribution du Min (haut-gauche) : Les valeurs minimales représentent les zones les plus sombres (noirs profonds). Des valeurs trop élevées (>50) peuvent indiquer une sous-exposition ou un manque de zones réellement noires, ce qui réduit le contraste.
    - Distribution du Max (haut-droite) : Les valeurs maximales représentent les zones les plus claires (blancs purs). Des valeurs trop faibles (<200) suggèrent une sur-exposition ou un manque de zones réellement blanches, également problématique pour le contraste.
    - Boxplot du Min (bas-gauche) : Permet d'identifier les classes avec des valeurs minimales anormales. Les outliers vers le haut indiquent des images potentiellement sous-exposées.
    - Boxplot du Max (bas-droite) : Identifie les classes avec des valeurs maximales anormales. Les outliers vers le bas indiquent des images potentiellement sur-exposées.
    """)
        
        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Graphique en barres empilées
        ax1 = axes[0]
        quality_by_class_no_margin = pd.crosstab(df_opacity['class'], df_opacity['quality'])
        quality_by_class_no_margin.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
        ax1.set_xlabel('Classe')
        ax1.set_ylabel('Nombre d\'images')
        ax1.set_title('Volumétrie de Qualité par Classe (Absolu)')
        ax1.legend(title='Qualité', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Graphique en pourcentages
        ax2 = axes[1]
        quality_by_class_pct_no_margin = pd.crosstab(df_opacity['class'], df_opacity['quality'], normalize='index') * 100
        quality_by_class_pct_no_margin.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis')
        ax2.set_xlabel('Classe')
        ax2.set_ylabel('Pourcentage (%)')
        ax2.set_title('Volumétrie de Qualité par Classe (Pourcentage)')
        ax2.legend(title='Qualité', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        st.pyplot(fig)
      
        st.markdown("""
    Ce graphique présente la répartition de la qualité des images :
    - Graphique Absolu (gauche) : Nombre total d'images par classe et par niveau de qualité. Permet d'identifier rapidement les classes avec le plus d'images problématiques.
    - Graphique Pourcentage (droite) : Proportion relative de chaque niveau de qualité par classe. Plus informatif pour comparer les classes indépendamment de leur taille.
    \n
    Classification de qualité :
    - Excellente : Opacité optimale (moyenne 100-180, range >100, std >20)
    - Bonne : Opacité acceptable (moyenne 80-200, range >80, std >15)
    - Moyenne : Opacité sous-optimale mais utilisable (moyenne 60-220, range >60)
    - Faible : Images problématiques nécessitant une attention (sous/sur-exposition, faible contraste)
    """)

@st.cache_data
def page_preprocessing():
    st.title("Preprocessing & explication des choix", text_alignment="center")
    st.subheader("Application des masks :")
    st.markdown("""
    - Les masks étant plus petits que les radio, un redimensionnement est necessaire. Les masks étant binaires (0 ou 255), le choix d'un redimensionnement NEAREST est le plus judicieux.
    - Les masks sont ensuite appliqués sur les radio afin de limiter le risque que le modèle utilisent des données non pertinentes pour la détection.
    """)
    st.subheader("Analyse des artefacts :")
    st.markdown("""
    - La quantité d'image contenant des artefacts étant assez faible, le choix est fait de ne pas les utiliser lors de l'entrainement. 
    - La possibilité de "supprimer" les artefacts et les remplacer par de la donnée extrapolée a été envisagée mais le risque d'altération a été considéré comme trop élevé.
    - Le volume d'image "difficilement exploitables", c'est à dire trop sombre, trop claire ou trop mal cadré étant faible, il a été décidé de les garder pour l'entrainement.
    
    ➜ Au final, seul le masquage des radiographie sera appliqué.
    """)
    st.subheader("Paramètres choisis")
    st.write("""
    "resize": (299,299),\n
    "resizeMethod": "NEAREST_NEIGHBOR",\n
    "NOT IN" : liste_artefacts
    """)
    st.markdown("### Code (extrait)")
    st.code("""
    # Fonction de chargement des images
    def load_image(path, image_size=(299,299)):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, expand_animations=False, channels=3)
        img = tf.image.resize(img, image_size,image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.cast(img, tf.float32) / 255.0
            
        return img
    """, language="python")

@st.cache_data
def page_models(subpage):
    st.title("Les différents modèles entraînés", text_alignment="center")
  
    if subpage == "Machine Learning":
        st.title("Machine Learning")
        st.markdown("""En première approche, un modele XGBoost est testé sur les images bruts (array de représentation des pixels)""")
        cols = st.columns([2,1]) 
        with cols[0]:
            st.header("Modèle XGBoost")
            st.subheader("Rapport de classification")
            st.table(metrics_XGBoost.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_XGBoost.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        st.markdown("""Bien que relativement correct, ces résultats ne sont pas à la hauteur de ce que l’on pourrait espérer avec un modèle de deep learning""")
    elif subpage == "Deep learning maison":
        st.title("Deep learning maison")
        
        cols = st.columns([2,1])
        with cols[0]:
            st.header("Modèle CNN custom")
            st.subheader("Rapport de classification")
            st.table(metrics_CNN_Custom.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_CNN_Custom.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
           
        cols = st.columns([2,1]) 
        with cols[0]:
            st.header("Modèle LeNet-5 custom")
            st.subheader("Rapport de classification")
            st.table(metrics_LeNet.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_LeNet.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
            
        st.subheader("Conclusion")
        st.markdown("""Les résultats sont relativement interessant, mais le recall de la class COVID est trés faible. Il semble judicieux d'essayer d'entrainer des modèles plus complexes""")

    elif subpage == "Transfert learning":
        st.title("Transfert learning")
        st.markdown("""Les modèles maison n’ayant pas donné des résultats satisfaisants, l’utilisation de modèles par transfert learning est envisagé. 
        Les modèles seront testés dans les même conditions afin de pouvoir effectuer un classement de performance.""")
        
        cols = st.columns([2,1])
        with cols[0]:
            st.header("Modèle DenseNet121")
            st.subheader("Rapport de classification")
            st.table(metrics_DenseNet121.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_DenseNet121.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
           
        cols = st.columns([2,1]) 
        with cols[0]:
            st.header("Modèle RestNet18")
            st.subheader("Rapport de classification")
            st.table(metrics_RestNet18.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_RestNet18.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
            
        cols = st.columns([2,1])
        with cols[0]:
            st.header("Modèle VGG19")
            st.subheader("Rapport de classification")
            st.table(metrics_VGG19.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_VGG19.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
           
        cols = st.columns([2,1]) 
        with cols[0]:
            st.header("Modèle EfficientNetB5")
            st.subheader("Rapport de classification")
            st.table(metrics_EfficientNetB5.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_EfficientNetB5.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
            
        cols = st.columns([2,1])
        with cols[0]:
            st.header("Modèle EfficientNetV2B0")
            st.subheader("Rapport de classification")
            st.table(metrics_EfficientNetV2B0.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_EfficientNetV2B0.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
           
        cols = st.columns([2,1]) 
        with cols[0]:
            st.header("Modèle EfficientNetV2B3")
            st.subheader("Rapport de classification")
            st.table(metrics_EfficientNetV2B3.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_EfficientNetV2B3.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))

        st.title("Top 3 des meilleurs modèles d’après les performances", text_alignment="center")
        cols = st.columns([1,4,1])
        with cols[1]:
            st.image("images/podium.png" if os.path.exists("images/podium.png") else None)
            
    elif subpage == "Optimisation":
        st.title("Optimisation")
        st.markdown("""Les résultats de ces 3 modèles sont très satisfaisants, cependant, comme vu lors de l’analyse du dataset, la class COVID est très déséquilibrée. En partant des 3 modèles sélectionnés précédemment, un entrainement par oversampling est testé pour essayer d’améliorer le recall de la class COVID, métrique très important dans un contexte médical.
        \nLes radiographie de la class COVID du jeu d’entrainement sont dupliquées 4 fois (pour arriver à un équilibre avec la class non COVID) et chaque image subit une manipulation aléatoire (rotation, translation) pour limiter les doublons.
        """)
        
        cols = st.columns([2,1])
        with cols[0]:
            st.header("Modèle DenseNet121 + Oversampling")
            st.subheader("Rapport de classification")
            st.table(metrics_DenseNet121_opt.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_DenseNet121_opt.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
           
        cols = st.columns([2,1]) 
        with cols[0]:
            st.header("Modèle EfficientNetV2B0 + Oversampling")
            st.subheader("Rapport de classification")
            st.table(metrics_EfficientNetV2B0_opt.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_EfficientNetV2B0_opt.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
            
        cols = st.columns([2,1])
        with cols[0]:
            st.header("Modèle EfficientNetV2B3 + Oversampling")
            st.subheader("Rapport de classification")
            st.table(metrics_EfficientNetV2B3_opt.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
        with cols[1]:
            st.title(" ")
            st.subheader("Matrice de confusion")
            st.table(heatmap_EfficientNetV2B3_opt.style
                .set_properties(**{
                    "background-color": "white",
                    "color": "black",
                    "border-color": "#E2E8F0",
                    "font-size": "14px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F1F8FF"),
                            ("color", "black"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #000000"),
                            ("text-align", "center")
                        ]
                    }
                ]))
            
@st.cache_data
def page_best_model(subpage):
  
    st.title("Le choix du meilleur modèle", text_alignment="center")
  
    if subpage == "GradCam":
        st.title("GradCam")
        st.markdown("""Les métriques sont un bon indicateur de performance pour les modèles. Cependant, les données traitées étant des images, les pixels utilisés par le modèle pour justifier sa prédiction sont aussi à prendre en compte.
    \nLes 3 modèles font donc l’objet d’une analyse via GradCam afin de s’assurer que le modèle ne se base pas sur des zones non fiables :
    """)
        st.subheader("Echantillon de GradCam pour DenseNet121")
        st.image("images/gradDN121.png" if os.path.exists("images/gradDN121.png") else None)
        st.subheader("Echantillon de GradCam pour EfficientNetV2B0")
        st.image("images/gradENv2B0.png" if os.path.exists("images/gradENv2B0.png") else None)
        st.subheader("Echantillon de GradCam pour EfficientNetV2B3")
        st.image("images/gradENv2B3.png" if os.path.exists("images/gradENv2B3.png") else None)
        
        st.subheader("Conclusion")
        st.markdown("""On constate que les modèles EfficientNetV2 ont un GradCam beaucoup plus cohérent que DenseNet. 
    \nDe plus EfficientnetV2B0 a un recall sur la class COVID beaucoup plus faible. Le choix se portera donc sur EfficientNetV2B0.
    """)
    elif subpage == "Présentation du meilleur modèle":
        st.title("Présentation du meilleur modèle")
        st.header("Schéma d'architecture")
        st.image("images/schemaENv2B0.png" if os.path.exists("images/schemaENv2B0.png") else None)
        st.header("Description")
        st.markdown("""EfficientNetV2-B0 est structuré comme une succession organisée de blocs convolutifs optimisés pour l’efficacité. 
    Le réseau commence par une convolution initiale 3×3 qui réduit la taille spatiale de l’image et augmente le nombre de canaux à 24. 
    \nEnsuite, il enchaîne plusieurs étapes de blocs Fused-MBConv et MBConv : les Fused-MBConv occupent les premiers étages et combinent des convolutions 1×1 et 3×3 pour accélérer le calcul, tandis que les MBConv avec Squeeze-and-Excitation apparaissent plus profondément, permettant au réseau de moduler l’importance de chaque canal.
    \nChaque étape du réseau augmente progressivement le nombre de canaux et réduit la résolution spatiale, passant par des blocs avec 24, 48, 64, 128, 160, puis 256 canaux. 
    \nEn fin de réseau, une convolution 1×1 projette les caractéristiques à 1280 canaux avant un Global Average Pooling, suivi de la couche fully connected pour la classification. 
    \nCette organisation stage par stage permet de combiner efficacement extraction locale et compréhension globale, tout en limitant le nombre total de paramètres à environ 7 millions.
    """)

def page_demo_predict():
    
    st.title("Démo: prédiction sur N cas aléatoires (avec Grad-CAM)", text_alignment="center")

    choice = st.selectbox(
        "Choisissez le modèle",
        list(MODEL_DICT.keys()))
    
    model_name, img_size = MODEL_DICT[choice]
    #model_name = model_loaded[choice]

    sample_size = st.number_input(
        "Nombre de cas",
        min_value=1, 
        max_value=30,
        step=1
    )
    
    #check = st.checkbox("uniquement des cas COVID ?")
    check = st.radio("Type de sample:", ["COVID", "Non-COVID", "COVID/non-COVID"])
    
    if st.button("Prédire"):
        sample = select_sample_data(DATASET_REP, sample_size, check)

        model = Base(
            data_folder=DATASET_REP,
            save_model_folder=MODELS_FOLDER,
            model_name=model_name,
            img_size=img_size
            )

        model.load()
            
        grad_cam_result = model.generate_gradcam_2(sample)
              
        for grad_cam_title, grad_cam_image in grad_cam_result:
            st.image(grad_cam_image, caption=grad_cam_title, width=600)

@st.cache_data
def page_conclusion(subpage):
    st.title("Conclusion", text_alignment="center")
    if subpage == "Problématique rencontrées":
        st.header("Problématique rencontrées")
        st.subheader("Problématiques fonctionnelles")
        st.markdown("""L’analyse de radiographie nécessite des connaissances médicales très poussées que notre équipe ne possède pas. 
    \nCette problématique limite nos choix de preprocessing ainsi que l’analyse des GradCam.
    """)
        st.subheader("Problématiques techniques")
        st.markdown("""
    -	L'entraînement de modèles très complexes avec le dataset complet nécessite l’utilisation de GPU très puissants que notre équipe ne possède pas. 
    Cette problématique réduit les possibilités en termes de transfert learning (entraînement de modèle très complexes et peut-être plus performants) ainsi que la taille du dataset utilisé. 
    Pour ce point, l’utilisation d’un dataset physique avec des batch physiques ont permis d’augmenter nos limites techniques, mais en restant toutefois limité.
    -	Concernant le dataset, le déséquilibre des classes est assez gênant pour obtenir un modèle réellement performant. 
    L’utilisation d’oversampling a permis d’améliorer certains modèles, sans pour autant obtenir un résultat optimal. 
    Des recherches ont été faites pour ajouter des données extérieures au dataset mais sans succès.
    -	Le projet nécessitait des connaissances en deep learning qui n'étaient proposées que dans des sprints assez éloignés du début du projet. 
    L'anticipation de ces cours a été nécessaire. 
    -	La recherche d’artefacts dans les images nécessite des compétences en computer vision assez poussés. 
    Le module proposé dans la formation ne propose que des compétences de base. 
    Des recherches extérieures ont été nécessaires.
    """)
    elif subpage == "Bilan":
        st.header("Bilan")
        st.markdown("""L’objectif du projet était d'entraîner un modèle performant afin de détecter le COVID à partir de radiographies pulmonaires. 
    \nLe choix de l’équipe projet a donc été de ne prédire que deux classes non-COVID et COVID. 
    \nDe plus, le projet étant dans un contexte médical de détection de maladie, l’indicateur important est le recall de la classe COVID. 
    \nNotre modèle a prédit 15 faux négatifs pour 723 cas, ce qui est un résultat plus qu’honorable. 
    \nDe plus, le GradCam montre que le modèle utilise des zones cohérentes pour faire sa détection.
    """)
    elif subpage == "Suite du projet":
        st.header("Suite du projet")
        st.markdown("""
    \nL’apport de nouvelles radiographies permettrait d’améliorer les performances du modèle.
    \nUne machine plus performante avec des GPU plus puissants permettrait d'entraîner des modèles plus complexes avec un dataset plus conséquent et certainement améliorer les performances.
    \nSe rapprocher d'experts du domaine médical permettrait non pas d’améliorer le modèle, mais d’améliorer l’analyse des résultats pour valider le modèle.
    """)

def page_end():
    st.title("Merci pour votre attention", text_alignment="center")
    cols = st.columns([1,4,1])
    with cols[1]:
        st.image("images/thanks.png" if os.path.exists("images/thanks.png") else None)
        
    st.title("Avez-vous des questions ?", text_alignment="center")
