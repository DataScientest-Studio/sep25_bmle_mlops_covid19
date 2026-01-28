import os
from pathlib import Path
import kagglehub
import random
import shutil
import cv2
import numpy as np

def load_dataset_by_classes(dataset_path: str) -> list:
    
    CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia', "Autre"]  # classes du dataset

    class_0 = []
    class_1 = []
    class_2 = []
    
    for cls in CLASSES:  # boucle classes
        class_path = dataset_path / cls  # répertoire cible
        if cls == "COVID":
            class_1.append(class_path)
        elif cls == "Autre":
            class_2.append(class_path)
        else:
            class_0.append(class_path)
                    
    return {"0":class_0, "1": class_1, "2": class_2}

# ----------------------------------------------------------------------
# Organisation d'un dataset avec plusieurs dossiers pour une classe
# ----------------------------------------------------------------------
def organize_custom_dataset(
        dataset_path: str,
        output_root: str,
        images_by_folder: dict,
        shuffle: bool = True,
        copy_mode: str = "copy",
        image_folder_name: str = "images",
        replace:bool = False
    ) -> dict:
        """
        Organise un dataset où plusieurs dossiers peuvent appartenir à une même classe.

        Args:
            output_root (str): dossier de sortie structuré.

            images_per_folder (dict):
                {
                    "folderA": 1200,
                    "folderB": 1200,
                    "folderC": 1200,
                    "folderD": None   # None = prendre toutes les images
                }

            shuffle (bool): mélanger les images dans chaque dossier.
            copy_mode (str): 'copy' ou 'move'
            replace (bool): Si le dossier existe, l'ecraser
        """

        if os.path.exists(output_root):
            if replace:
                shutil.rmtree(output_root)
                os.makedirs(output_root)
            else:
                raise FileExistsError("Le dossier existe déjà. Positionner replace à True si vous voulez l'écraser")
        else:
            os.makedirs(output_root)

        result = {}
        
        class_folders = load_dataset_by_classes(dataset_path)

        # Pour chaque classe
        for class_id, folder_list in class_folders.items():

            # Dossier de sortie de la classe
            class_output_dir = os.path.join(output_root, str(class_id))
            os.makedirs(class_output_dir, exist_ok=True)

            result[class_id] = 0

            # Pour chaque dossier de cette classe
            for folder_path in folder_list:
                image_path = folder_path / image_folder_name

                # Liste des images
                all_images = [
                    os.path.join(image_path, f)
                    for f in os.listdir(image_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif"))
                ]

                if not all_images:
                    continue

                # Mélange
                if shuffle:
                    random.shuffle(all_images)

                # Limite demandée
                limit = images_by_folder.get(os.path.basename(folder_path), None)

                if limit is not None:
                    all_images = all_images[:limit]

                # Copie ou déplacement
                for img_path in all_images:
                    dst = os.path.join(class_output_dir, os.path.basename(img_path))

                    if copy_mode == "copy":
                        shutil.copy(img_path, dst)
                    else:
                        shutil.move(img_path, dst)

                    result[class_id] += 1

        print(f"[INFO] Comptage par classe : {result}")

        return result, output_root
    
# permet d'importer les images depuis kaggle. Attention il faut avoir le json généré par kaggle avec le token dans un repertoire ".kaggle"
def import_dataset(rep_name: str, download_path: str):

    path = Path(download_path)
    # suppression du dossier cible si il existe déjà
    if path.exists():
        print(f"Suppression de {path}")
        shutil.rmtree(path)
    # création du dossier cible
    print(f"Création de {path}")
    path.mkdir(parents=True)
    
    print("telechargement du dataset")
    # Download latest version
    path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
    print(f"copie du dossier {path}\\{rep_name} vers {download_path}")
    move_command = f"xcopy {path}\\{rep_name} {download_path} /S /I /Y /q"
    print(move_command)
    sortie = os.popen(move_command)
    print(sortie.read())
    

def apply_masks_to_folder(path, replace=False):
    """
        Permet de créer un dossier contenant les images masquées à partir d'un dossier contenant
        un dossier images et un dossier masks
        Entrée :
            Path du dossier
            replace (bool): Si le dossier existe, l'ecraser
    """

    images_dir = path / "images"
    masks_dir = path / "masks"
    output_dir = path / "masked_images"

    if os.path.exists(output_dir):
        if replace:
           shutil.rmtree(output_dir)
           os.makedirs(output_dir)
        else:
            raise FileExistsError("Le dossier existe déjà. Positionner replace à True si vous voulez l'écraser")
    else:
        os.makedirs(output_dir)

    # Liste des fichiers d'images
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"{len(image_files)} images trouvées.")

    for filename in image_files:
        img_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, filename)

        # Vérification existence du mask
        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask absent pour {filename}, ignoré.")
            continue

        # Lecture image + mask
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[ERROR] Image illisible : {img_path}")
            continue

        if mask is None:
            print(f"[ERROR] Mask illisible : {mask_path}")
            continue


        # Taille mismatch ? → resize du mask à la taille de l'image
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Bitwise AND
        masked = cv2.bitwise_and(img, mask)

        # Sauvegarde
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, masked)

        #print(f"[OK] {filename} → masqué et sauvegardé.")

    print("\nTerminé ! Images masquées dans :", output_dir)
    
    
def select_sample_data(dataset_path: str, sample_size: int, sample_type: str) -> list:
    
    print(f"{sample_type = }")
    if sample_type == "COVID":
        nb_cls_1 = sample_size
        nb_cls_0 = 0
        nb_cls_2 = 0
    elif sample_type == "Non-COVID":
        nb_cls_1 = 0
        nb_cls_0 = sample_size
        nb_cls_2 = 0
    elif sample_type == "Autre":
        nb_cls_2 = sample_size
        nb_cls_0 = 0
        nb_cls_1 = 0
    else:    
        nb_cls_0 = sample_size // 2
        nb_cls_1 = sample_size - nb_cls_0
        nb_cls_2 = 0
    
    print(f"{nb_cls_0 = }, {nb_cls_1 = }, {nb_cls_2 = }")
    class_0_list = []
    class_1_list = []
    class_2_list = []
    
    results = []
    
    class_folders = load_dataset_by_classes(dataset_path)
    print(f"{class_folders = }")
    # Pour chaque classe
    for class_id, folder_list in class_folders.items():
        
        # Pour chaque dossier de cette classe
        for folder_path in folder_list:
            image_path = folder_path / "images"

            # Liste des images
            all_images = [
                (folder_path, f)
                for f in os.listdir(image_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif"))
            ]

            # Mélange
            random.shuffle(all_images)
            
            if class_id == "0":
                class_0_list.extend(all_images)
            elif class_id == '1':
                class_1_list.extend(all_images)
            else:
                class_2_list.extend(all_images)
    
    results.extend(random.sample(class_0_list, nb_cls_0))
    results.extend(random.sample(class_1_list, nb_cls_1))
    results.extend(random.sample(class_2_list, nb_cls_2))
    random.shuffle(results)
    
    return results

def build_masked_dataset_by_classes(
    input_root,
    output_root
):
    """
    Parameters
    ----------
    input_root : str or Path
        Dossier contenant les 4 classes (chacune avec images/ et masks/)
    output_root : str or Path
        Dossier de sortie avec 0/ et 1/
    """

    input_root = Path(input_root)
    output_root = Path(output_root)
    class_mapping = {
            "Normal": 0,
            "Lung_Opacity": 0,
            "COVID": 1,
            "Viral Pneumonia": 0
    }

    # Créer dossiers de sortie
    for c in ["0", "1"]:
        (output_root / c).mkdir(parents=True, exist_ok=True)

    for class_name, label in class_mapping.items():
        class_dir = input_root / class_name
        images_dir = class_dir / "images"
        masks_dir = class_dir / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise RuntimeError(f"Structure invalide dans {class_dir}")

        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in ".png":
                continue

            mask_path = masks_dir / img_path.name

            if not mask_path.exists():
                print(f"Mask manquant pour {img_path.name}, ignoré")
                continue

            # Lecture image + masque
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Binariser le masque
            mask = (mask > 0).astype(np.uint8)
            
            # Binariser le masque
            mask = cv2.resize(mask, (img.shape[0], img.shape[1]))

            # Appliquer le masque
            masked = img * mask[:, :, None]

            # Nom de sortie unique
            out_name = img_path.name
            out_path = output_root / str(label) / out_name

            cv2.imwrite(str(out_path), masked)

    print("Dataset binaire masqué créé avec succès.")
