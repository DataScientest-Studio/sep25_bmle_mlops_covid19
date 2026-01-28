# COVID-19 Radiography — EfficientNetV2 Training Pipeline

Ce projet implémente un pipeline complet d’entraînement deep learning pour la classification d’images radiographiques COVID-19 à partir du dataset Kaggle COVID-19 Radiography Dataset.

Le pipeline automatise l’ensemble de la chaîne, depuis la récupération des données jusqu’à l’analyse des résultats du modèle.

## Fonctionnalités principales

Le pipeline prend en charge automatiquement :

    Le téléchargement du dataset Kaggle (si inexistant)
    La structuration du dataset en train / test (si inexistante)
    L’oversampling de la classe minoritaire
    L’entraînement d’un modèle EfficientNetV2-B0
    La sauvegarde, l’évaluation et l’interprétation du modèle

## Structure du projet
    project_root/
    │
    ├── data/
    │   ├── COVID-19_Radiography_Dataset/      # Dataset brut (Kaggle)
    │   └── structured_dataset_for_training/  # Dataset restructuré pour l'entraînement
    │
    ├── src/
    │   ├── features/
    │   │   └── tensor/
    │   │       └── models_to_test.py          # Contient EfficientNetv2B0_model_augmented
    │   │
    │   ├── models/                            # Modèles entraînés (sauvegardés ici)
    │   └── utils/
    │       ├── data_utils.py
    │       └── image_utils.py
    │
    └── main.py                               # Script principal

# Prérequis
**1 - Clé Kaggle**

Le téléchargement automatique du dataset nécessite la présence du fichier :

    project_root/.kaggle/kaggle.json

Ce fichier est généré depuis Kaggle :
Account → API → Create New API Token

**2 - Dépendances Python**

(en supposant que ton environnement TensorFlow / PyTorch est déjà configuré)

    pip install -r requirements.txt

**=> Exécution**

    python main.py


Le script va automatiquement :

    Télécharger le dataset Kaggle s’il n’existe pas
    Créer un dataset structuré :

        structured_dataset_for_training/
        ├── train/
        │   ├── 0/
        │   └── 1/
        └── test/
            ├── 0/
            └── 1/

    Appliquer un oversampling ×4 de la classe 1
    Entraîner EfficientNetV2-B0 pendant 200 epochs (avec callbacks d’arrêt)
    Sauvegarder le modèle dans src/models/

**Option** 
Forcer l’écrasement des dossiers existants
    
    python main.py --force


Cela force le re-téléchargement du dataset Kaggle et la recréation des données même si elles existent déjà.

## Modèle

Le modèle utilisé est :

    EfficientNetv2B0_model_augmented

Paramètres principaux :

    Augmentation de données

    Oversampling

    Batch size = 16

    big_dataset = True   # les batchs sont chargés depuis le disque plutôt que via la RAM

## Sauvegarde des modèles

Les modèles sont sauvegardés avec horodatage :

    EfficientNetv2B0_model_trained_2501-01-14-16-03-42
    Format : YYMM-MM-DD-HH-MM-SS

## Pipeline exécuté

La méthode :

    model.execute(epochs=200)

exécute automatiquement :

    load_data → fit → save → evaluate → predict → gradcam

Cela permet :

    D’évaluer les performances 
    De générer des prédictions
    De produire des cartes Grad-CAM pour l’interprétation du modèle

## Objectif du projet

Ce projet vise à fournir une chaîne complète, reproductible et exploitable pour l’entraînement d’un réseau CNN de diagnostic médical, avec un focus sur :

    Le déséquilibre de classes
    L’automatisation du pipeline de données
    L’explicabilité des modèles (Grad-CAM)