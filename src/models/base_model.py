from abc import ABC, abstractmethod
import asyncio
import os
from pathlib import Path
import sys
import cv2

from src.settings import S3Settings
from src.utils.s3_utils import S3ImageLoader

# Désactive les logs TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Désactive les logs XLA
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models, Model
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.image_utils import load_image, overlay_gradcam_on_gray, overlay_heatmap, show
from src.utils.data_utils import build_metadata, build_tf_dataset
from src.utils.database_utils import fetch_dataset
from src.utils.modele_utils import find_last_conv_layer


class Base_model(ABC):
    """
    Classe mère générique dédiée à des projets de Deep Learning sur images.
    Elle gère :
        - chargement du dataset
        - split train/val/test
        - preprocessing standard
        - création/compilation du modèle
        - entraînement / évaluation / prédiction
        - sauvegarde / chargement
    Seule la méthode build_model() doit être définie dans les classes enfants.
    """

    def __init__(self,
                 data_folder,
                 save_model_folder,
                 model_name,
                 img_size=(299, 299),
                 gray=False,
                 batch_size=32,
                 big_dataset=False,
                 train_size=0.2,
                 random_state=42,
                 oversampling=False):

        print("init ok")
        # Propriétés générales
        self.data_folder = data_folder
        self.save_model_folder = save_model_folder
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.big_dataset = big_dataset
        self.train_size = train_size
        self.random_state = random_state
        self.oversampling = oversampling
        self.oversampling_ratio = 1
        
        # Images en nuance de gris?
        self.gray = gray
        
        # Placeholders
        self.train_gen = None
        self.test_gen = None
        self.history = None
        self.model = None
        self.callbacks = None
        
        # Evaluation
        self.evaluation = None
        
        # predictions
        self.predictions = None
        self.metrics = None
        self.nb_validation_data = 0
        self.nb_training_data = 0
        # Initialisation du modèle
        self.model = self.build_model()
        
        # résumé du model
        #self.summary()

    # ----------------------------------------------------------------------
    # Loading & preprocessing dataset
    # ----------------------------------------------------------------------
    def load_data(self):
        """Chargement du dataset à partir d’un dossier structuré (classes = sous-dossiers)."""

        if self.gray:
                train_ds = image_dataset_from_directory(
                    self.data_folder,
                    validation_split= 1 - self.train_size,
                    subset="training",
                    seed=self.random_state,
                    image_size=self.img_size,
                    batch_size=self.batch_size,
                    label_mode="categorical",
                    color_mode="grayscale",
                    shuffle=True  
                )

                val_ds = image_dataset_from_directory(
                    self.data_folder,
                    validation_split= 1 - self.train_size,
                    subset="validation",
                    seed=self.random_state,
                    image_size=self.img_size,
                    batch_size=self.batch_size,
                    label_mode="categorical",
                    color_mode="grayscale",
                    shuffle=False  
                )
        else:
                train_ds = image_dataset_from_directory(
                    self.data_folder,
                    validation_split=1 - self.train_size,
                    subset="training",
                    seed=self.random_state,
                    image_size=self.img_size,
                    batch_size=self.batch_size,
                    label_mode="categorical",
                    shuffle=True
                )

                val_ds = image_dataset_from_directory(
                    self.data_folder,
                    validation_split=1 - self.train_size,
                    subset="validation",
                    seed=self.random_state,
                    image_size=self.img_size,
                    batch_size=self.batch_size,
                    label_mode="categorical",
                    shuffle=False
                )
                
        self.nb_training_data = sum(1 for _ in train_ds.unbatch())
        self.nb_validation_data = sum(1 for _ in val_ds.unbatch())
        
        # Optimisation performances
        if self.big_dataset:
            self.train_gen = train_ds.cache("./cache_train").prefetch(buffer_size=tf.data.AUTOTUNE)
            self.test_gen = val_ds.cache("./cache_val").prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            self.train_gen = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            self.test_gen = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            
    def load_data_from_s3(self, cache_dir, nb_case=100):
        
        raw_data = asyncio.run(fetch_dataset())
        
        df = pd.DataFrame(raw_data)

        train_df, test_df, oversampling_ratio = build_metadata(df, 1 - self.train_size, self.random_state, nb_case=nb_case)
        
        train_df.to_csv("dataset/train_df.csv")
        test_df.to_csv("dataset/test_df.csv")
        
        self.nb_training_data = len(train_df)
        self.nb_validation_data = len(test_df)
        
        settings = S3Settings("secrets.yaml")
    
        bucket_name, access_key, secret_key, b2_endpoint = settings.s3_access
 
        loader = S3ImageLoader(
            endpoint=b2_endpoint,
            key_id=access_key,
            app_key=secret_key,
            bucket=bucket_name,
            cache_dir=cache_dir
        )

        train_ds = build_tf_dataset(
            train_df,
            loader,
            batch_size=self.batch_size,
            image_width=self.img_size[0],
            image_heigh=self.img_size[1]            
        )

        test_ds = build_tf_dataset(
            test_df,
            loader,
            batch_size=self.batch_size,
            image_width=self.img_size[0],
            image_heigh=self.img_size[1]
        )

        self.train_gen = train_ds
        self.test_gen = test_ds
        self.oversampling_ratio = oversampling_ratio
                    
    # ----------------------------------------------------------------------
    # Construction du modèle
    # ----------------------------------------------------------------------
    @abstractmethod
    def build_model(self) -> models.Sequential:
        """
        Doit être implémentée par les modèles enfants.
        """
        pass

    # ----------------------------------------------------------------------
    # Entraînement
    # ----------------------------------------------------------------------
    def fit(self, epochs=5):
        """Entraine le modèle."""

        if self.callbacks == None:
            self.history = self.model.fit(
                self.train_gen,
                validation_data=self.test_gen, 
                epochs=epochs
            )

        else:
            self.history = self.model.fit(
                self.train_gen,
                validation_data=self.test_gen,
                callbacks=self.callbacks, 
                epochs=epochs
            )

        
        #self.show_history()
        
    def show_history(self):
        """ affiche le graphe de l'history """
        print(self.history.history.keys())
        acc = self.history.history['accuracy']
        loss = self.history.history['loss']
        val_acc = self.history.history['val_accuracy']
        val_loss = self.history.history['val_loss']
        plt.plot(np.arange(1 , len(acc)+1, 1),
                acc,
                label='Training Accuracy',
                color='green')
        plt.plot(np.arange(1 , len(loss)+1, 1),
                loss,
                label='Training Loss',
                color='red')
        plt.plot(np.arange(1 , len(val_acc)+1, 1),
                val_acc,
                label='Validation Accuracy',
                color='blue')
        plt.plot(np.arange(1 , len(val_loss)+1, 1),
                val_loss,
                label='Validation Loss',
                color='yellow')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show();

    # ----------------------------------------------------------------------
    # Prédictions
    # ----------------------------------------------------------------------
    def predict(self, X_predict=[]):
        """Prédit les classes d'un batch d'images."""
        if len(X_predict) > 0:
            self.predictions = self.model.predict(X_predict)
        else:
            if self.test_gen:
                self.predictions = self.model.predict(self.test_gen)
                self.evaluate_predictions()
            else:
                raise 
    
    # ----------------------------------------------------------------------
    # Évaluation
    # ----------------------------------------------------------------------
    def evaluate(self):
        """Évalue sur les données test."""
        self.evaluation = self.model.evaluate(self.test_gen)
        print(f"loss = {self.evaluation[0]}, accuracy={self.evaluation[1]}")
    
    def evaluate_predictions(self):
        """
        Compare les prédictions du modèle avec les labels réels d'un dataset de test.
        
        Args:
            self
            
        Returns:
            dict contenant accuracy, matrice de confusion, classification report
        """
        
        # --- Extraction des images et labels ---
        y_true = []
        X = []

        for images, labels in self.test_gen:
            X.append(images)
            y_true.append(labels.numpy())

        X = np.concatenate(X)
        y_true = np.concatenate(y_true)
        
        # --- Conversion labels one-hot -> entiers ---
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        # --- Prédictions ---
        y_pred = np.argmax(self.predictions, axis=1)

        # --- Scores ---
        classes = [0, 1]
        acc = accuracy_score(y_true, y_pred)
        cm = pd.crosstab(y_true, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'], dropna=False).reindex(index=classes, columns=classes, fill_value=0)
        report = classification_report(
            y_true, 
            y_pred,
            output_dict=True
        )

        self.metrics = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report
        }

    # ----------------------------------------------------------------------
    # Sauvegarde / Chargement
    # ----------------------------------------------------------------------
    def save(self):
        output_path = self.save_model_folder / self.model_name
        self.model.save(output_path)
        
    def load(self, show_summary=False):
        model_path = self.save_model_folder / self.model_name
        self.model = models.load_model(model_path)
        if show_summary:
            self.summary()

    # ----------------------------------------------------------------------
    # Résumé
    # ----------------------------------------------------------------------
    def summary(self):
        if self.model is None:
            raise ValueError("Le modèle n'est pas construit.")
        self.model.summary()
        
    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def execute(self, epochs=5):
        """Pipeline complet."""
        print("Chargement des données…")
        self.load_data()

        print("Entraînement du modèle…")
        self.fit(epochs)
        
        print("sauvegarde du model")
        self.save()

        print("Évaluation…")
        self.evaluate()
        
        print("Prédictions")
        self.predict()
        
        print("Grad-cam")
        self.generate_gradcam(max_images=5, mode_errors=True)

        print("Pipeline terminé !")
 
        
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        
        last_conv_layer_name = find_last_conv_layer(self.model)
        
        grad_model = Model(
            [self.model.inputs],
            [
                self.model.get_layer(last_conv_layer_name).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)

            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]).numpy())

            class_channel = predictions[:, pred_index]
            grads = tape.gradient(class_channel, conv_outputs)


        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return np.array(heatmap), int(pred_index)
    
    # ----------------------------------------------------------------------
    # Grad-CAM (visualisation des zones importantes)
    # ----------------------------------------------------------------------
    def generate_gradcam(
        self,
        mode_errors=False,
        max_images=10, 
    ):
        generated = 0

        for batch_images, batch_labels in self.test_gen:
         
            self.predict(X_predict=batch_images)
            pred_classes = np.argmax(self.predictions, axis=1)
            true_classes = np.argmax(batch_labels, axis=1)

            for i in range(len(batch_images)):
                if generated >= max_images:
                    return

                # Sélection des erreurs uniquement
                if mode_errors and pred_classes[i] == true_classes[i]:
                    continue
                
                elif pred_classes[i] == 0 and true_classes[i] == 0:
                    continue

                # Préparation image
                img_array = tf.expand_dims(batch_images[i], axis=0)

                heatmap, predicted_class = self.make_gradcam_heatmap(
                    img_array
                )

                # Sauvegarde
                filename = f"{generated} true : {true_classes[i]} - pred1 : {pred_classes[i]} - pred2 : {predicted_class}"

                superposed = overlay_heatmap(np.array(batch_images[i]), heatmap)
                show(filename, superposed)

                generated += 1
                
    def generate_gradcam_2(
        self,
        image_list: list
    ) -> list:
        
        """Similaire à generate_gradcam, mais prend en entrée des images complète et applique le masque"""
        predict_list = []
        img_name_list = []
        img_array_list = []
        img_origin_list = []
        results = []
       
        for file_path, file_name in image_list:
                    
            images_dir = file_path / "images"
            masks_dir = file_path / "masks"

            img_path = os.path.join(images_dir, file_name)
            mask_path = os.path.join(masks_dir, file_name)

            # Vérification existence du mask
            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask absent pour {file_name}, ignoré.")
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


            # Taille differente entre image et mask -> rezsize du mask
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_NEAREST)

            # Bitwise AND
            masked = cv2.bitwise_and(img, mask)
            
            # Taille differente entre l'image et la taille demandée -> resize de l'image masquée et de l'image d'origine pour la superposition
            masked = tf.expand_dims(masked, axis=-1)
            masked = tf.image.resize(masked,size=(self.img_size[0], self.img_size[1]),method=tf.image.ResizeMethod.BILINEAR)
            img = cv2.resize(img,(self.img_size[0], self.img_size[1]))

            masked = tf.image.grayscale_to_rgb(masked)
                        
            # Sauvegarde
            img_name_list.append(file_name)
            img_array_list.append(masked)
            img_origin_list.append(img)
            
        predict_list.append((img_name_list, img_array_list, img_origin_list))

        for img_name, img_array, img_origin in predict_list:
            
            batch_images = np.stack(img_array, axis=0)
            batch_images = batch_images.astype('float32')

            for i in range(len(batch_images)):
                heatmap = None
                predicted_class = None
                predict_label = None
                # Préparation image
                img_to_grad = tf.expand_dims(batch_images[i], axis=0)

                heatmap, predicted_class = self.make_gradcam_heatmap(
                    img_to_grad
                )
                if predicted_class == 1:
                    predict_label = "COVID"
                else:
                    predict_label = "Non COVID"
                    
                # Sauvegarde
                case_name = f"name : {img_name[i]} - pred : {predict_label}"

                superposed = overlay_gradcam_on_gray(img_origin[i], heatmap, alpha=0.2)
                results.append((case_name, superposed))
                #show(case_name, superposed)
        
        return results

            
    def make_predict_dataset(self, folder, batch_size=16):
        """Charge toutes les images d'un dossier dans un tf.data.Dataset optimisé pour la prédiction."""
        
        # Liste des chemins d'images
        file_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png"))
        ]
        file_paths.sort()  

        # Création dataset des paths
        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)

        # Pipeline optimisé
        ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return ds, file_paths

    def predict_folders(self, folders, output_csv="predictions.csv"):
        rows = []

        for folder in folders:
            print(f"Traitement du dossier : {folder}")

            dataset, file_paths = self.make_predict_dataset(folder=folder)

            # Prédiction du batch complet
            self.predict(X_predict=dataset)
            
            # Gestion automatique des modèles binaire ou softmax
            prob1 = self.predictions[:, 0]
            prob0 = 1 - prob1

            # Construction des lignes pour le CSV
            for path, p0, p1 in zip(file_paths, prob0, prob1):
                rows.append({
                    "folder": os.path.basename(folder),
                    "filename": os.path.basename(path),
                    "prob_class0": float(p0),
                    "prob_class1": float(p1),
                })

        # Sauvegarde CSV final
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)

        print(f"\nCSV généré : {output_csv}")
        return df