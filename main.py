import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil

from src.models.models_to_test import EfficientNetv2B0_model_augmented
from src.utils.data_utils import build_masked_dataset_by_classes, import_dataset
from src.utils.image_utils import ensure_dirs, oversample_train_class_1, split_dataset


def main(args):
    # Paths
    DATASET_DIR_NAME = "COVID-19_Radiography_Dataset"                        # nom du dossier contenant le dataset                            
    PROJECT_ROOT = Path(os.getcwd())                                         # racine du projet
    DATASET_ROOT = PROJECT_ROOT / "data"                                     # racine du dataset
    DATASET_REP  = DATASET_ROOT / DATASET_DIR_NAME                           # chemin vers le répertoire contenant le dataset 
    MODELS_FOLDER = PROJECT_ROOT / "src" / "models"                          # chemin du dossier de sauvegarde des modèles entrainés
    dataset_root = DATASET_ROOT / "structured_dataset_for_training"          # chemin du dataset structuré pour l'entrainement
    tmp = DATASET_ROOT / "tmp"

    # Téléchargement du dataset depuis kaggle, si celui-ci n'a as encore été téléchargé
    # necessite d'avoir un dossier .kaggle à la racine du projet, contenant un fichier json avec la clé kaggle (généré sur le site)
    if not os.path.isdir(DATASET_REP) or args.force:
        import_dataset(DATASET_DIR_NAME, DATASET_REP)
        
    # création du dataset structuré avec overrsampling x4 si non déjà créé
    if not os.path.isdir(dataset_root):
        # Repartition + masquage 
        build_masked_dataset_by_classes(DATASET_REP, tmp)
        # R&partition des images par classe et application du mask
        ensure_dirs(output_dir=dataset_root)
        # structuration du dataset en deux dossiers  train/test, eux même séparés en deux dossier 0/1 représentant les 2 classes à prédire
        # ratio de répartition train/test = 80%/20%   
        split_dataset(source_dir=tmp, output_dir=dataset_root, train_ratio=0.8)
        # oversampling de la class 1 x4
        oversample_train_class_1(output_dir=dataset_root, oversample_multiplier = 4.0)
        # suppréssion du dossier tmp
        shutil.rmtree(tmp)
        
    # récupération du timestamp courant pour l'horodatage du modèle
    now = datetime.now().strftime("%y%m-%m-%d-%H-%M-%S")

    # création de la classe du modèle à entrainer
    model = EfficientNetv2B0_model_augmented(
        data_folder=dataset_root,
        save_model_folder=MODELS_FOLDER,
        model_name=f"EfficientNetv2B0_model_trained_{now}",
        batch_size=16,
        big_dataset=True,
        oversampling=True
        )
    
    # Execution du pipeline d'entrainement du modèle
    # load_data/fit/save/evaluate/predict/gradcam    
    model.execute(epochs=200)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Écraser le fichier s'il existe déjà"
    )

    args = parser.parse_args()
    main(args)