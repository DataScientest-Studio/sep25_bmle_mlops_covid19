import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import random
import shutil
import itertools
from PIL import Image
from tensorflow import image

# Zone de fonctions externes

def image_loss(img_origin, img_cleaned):
      
    if img_origin.sum() >= img_cleaned.sum():  
        return img_origin - img_cleaned
    else:
        return img_cleaned - img_origin

def estimate_canny_minmax(img):

    # Histogramme (256 niveaux)
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()

    # Normalisation
    hist = hist / hist.sum()

    # Calcul du "spread" : distance entre les percentiles 5% et 95%
    p5 = np.searchsorted(np.cumsum(hist), 0.05)
    p95 = np.searchsorted(np.cumsum(hist), 0.95)

    spread = (p95 - p5) / 255.0   # ≈ 0.1 à 1.0

    # Mapping : plus l'image est contrastée → sigma petit
    sigma = 0.7 - 0.5 * spread
    sigma = float(np.clip(sigma, 0.3, 0.7))
    
    # Calcul de la médiane des pixels
    v = np.median(img)
    
    # calcul des mon max du canny de manière standardisée
    canny_min = int(max(0, (1.0 - sigma) * v))
    canny_max = int(min(255, (1.0 + sigma) * v))
    
    if v == 0:
       canny_min = -1
       canny_max = -1

    return canny_min, canny_max


# Fonction d'affichage d'une image
def show(title, img):
    
    plt.figure(figsize = (10,8))
    plt.imshow(img, cmap = 'gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    plt.show();
    
def find_image(repertoire: str) -> list:
    """
    Méthode parcourant un repertoire d'image et la chargeant dans une liste.

    Args:
        repertoire (str): Chemin du répertoire où se trouvent les images.

    Returns:
        list: Liste contenant les noms des images
    """
    image_liste = []
    for racine, repertoires, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            image_liste.append(fichier)

    return image_liste

# Fonction de chargement des images
def load_image(path, image_size=(299,299)):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, expand_animations=False, channels=3)
    img = tf.image.resize(img, image_size,image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32) / 255.0
        
    return img

def overlay_heatmap(img, heatmap, alpha=0.4):
    
    h, w, d = img.shape
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)

    heatmap_c = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
    
    img_uint8 = np.uint8(255 * img)
    
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_c, alpha, 0)

def overlay_gradcam_on_gray(gray_img, heatmap, alpha=0.3, colormap=cv2.COLORMAP_JET, threshold=0.3):
    """
    Superpose un heatmap Grad-CAM sur une image grayscale en conservant le gris d'origine.
    
    Args:
        gray_img (np.array): image grayscale (H, W) ou (H, W, 1), valeurs 0-255 ou 0-1
        heatmap (np.array): heatmap normalisée entre 0 et 1, shape (H, W)
        alpha (float): intensité du heatmap
        colormap: cv2 colormap pour coloriser le heatmap
        threshold (float): ne colorer que les zones où heatmap > threshold
        
    Returns:
        superposed (np.array): image finale RGB avec le heatmap superposé
    """
    # Taille differente entre gray_img et mask -> rezsize du mask
    if gray_img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (gray_img.shape[0], gray_img.shape[1]), interpolation=cv2.INTER_LANCZOS4)
    
    # Assurer que l'image est uint8 et en (H, W)
    if len(gray_img.shape) == 3 and gray_img.shape[2] == 1:
        gray_img = gray_img[..., 0]
    gray_img_uint8 = np.uint8(255 * gray_img) if gray_img.max() <= 1 else gray_img.astype(np.uint8)
    
    # Convertir en BGR pour superposition
    gray_bgr = cv2.cvtColor(gray_img_uint8, cv2.COLOR_GRAY2BGR)
    
    # Préparer le heatmap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Créer un masque pour ne superposer que les zones « chaudes »
    mask = heatmap > threshold  # booléen
    mask_3c = np.stack([mask]*3, axis=-1)  # (H, W, 3)
    
    # Faire le blend uniquement sur les zones chaudes
    superposed = gray_bgr.copy()
    superposed = gray_bgr * (1 - alpha * mask_3c) + heatmap_color * (alpha * mask_3c)
    superposed = superposed.astype(np.uint8)
    
    return superposed


def ensure_dirs(output_dir):
    print("ensure_dirs")
    for split in ["train", "test"]:
        for cls in ["0", "1"]:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)


def split_dataset(source_dir, output_dir, train_ratio=0.8, seed = 42):
    print("split_dataset")
    random.seed(seed)
    for cls in ["0", "1"]:
        src_cls_dir = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(src_cls_dir) if f.endswith(".png")]
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for img in train_imgs:
            shutil.copy(
                os.path.join(src_cls_dir, img),
                os.path.join(output_dir, "train", cls, img)
            )

        for img in test_imgs:
            shutil.copy(
                os.path.join(src_cls_dir, img),
                os.path.join(output_dir, "test", cls, img)
            )

        print(f"Classe {cls} → train: {len(train_imgs)}, test: {len(test_imgs)}")


def get_last_covid_index(files):
    print("get_last_covid_index")
    indices = []
    for f in files:
        if f.startswith("COVID-") and f.endswith(".png"):
            try:
                indices.append(int(f.split("-")[1].split(".")[0]))
            except ValueError:
                pass
    return max(indices) if indices else 0


def augment_image(img, max_rotation = 20, max_shift = 0.1):

    w, h = img.size
    
    # rotation
    angle = random.uniform(-max_rotation, max_rotation)
    img = img.rotate(angle, resample=Image.BILINEAR)

    # translation
    dx = random.uniform(-max_shift, max_shift) * w
    dy = random.uniform(-max_shift, max_shift) * h
    img = img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.BILINEAR
    )

    # flip horizontal
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


def oversample_train_class_1(output_dir, oversample_multiplier = 4.0):
    print("oversampling_train_class_1")
    train_class1_dir = os.path.join(output_dir, "train", "1")
    images = [f for f in os.listdir(train_class1_dir) if f.endswith(".png")]

    initial_count = len(images)
    target_count = int(initial_count * oversample_multiplier)

    current_index = get_last_covid_index(images)
    cycle = itertools.cycle(random.sample(images, len(images)))
    generated = 0

    print(f"Oversampling train/1 : {initial_count} → {target_count}")

    while initial_count + generated < target_count:
        img_name = next(cycle)
        img_path = os.path.join(train_class1_dir, img_name)

        img = Image.open(img_path)
        aug_img = augment_image(img)

        current_index += 1
        new_name = f"COVID-{current_index:04d}.png"
        aug_img.save(os.path.join(train_class1_dir, new_name), quality=95)

        generated += 1

    print(f"{generated} images augmentées générées dans train/1")

