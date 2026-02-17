import asyncio
import random
import shutil
import sys
from pathlib import Path
from xmlrpc.client import boolean
from matplotlib import pyplot as plt
import seaborn as sns
import mlflow
from datetime import datetime
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.models_to_test import EfficientNetv2B0_model_augmented
from src.utils.data_utils import plot_classification_report
from src.utils.database_utils import get_parameters, post_metrics, get_metrics_model_by_stage, update_stage
from src.utils.mlflow_utils import  log_training_parameters


def train_model_mlflow():
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%f")
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"COVID_19")
    
    training_parameters = asyncio.run(get_parameters())
    
    model_name = f"model_{now}.keras"
            
    model = create_model_instance(training_parameters, model_name)
    
    cache_dir = "./image_cache"
    cache_dir_path = Path(cache_dir)
    
    metrics_dir = Path("./metrics")
    # création du dossier temporaire pour les images metrics
    metrics_dir.mkdir(parents=True, exist_ok=True)
        
    with mlflow.start_run(run_name=f"Training_{now}") as run:
        
        client = mlflow.MlflowClient()
        ########################## éléments pour simuler un scheduleur avec augmantation du dataset ##############
        nb_case = 100
        
        with open("training_dataset_size.txt", "r") as f:
            nb_case = f.read()
        
        print(f"{nb_case = }")
        epochs = 10
        
        # tirage aléatoire pour savoir si on augmente le nombre de cas ou le nombre d'epochs
        x = random.randint(0, 1)
        if x == 0:
            nb_case = int(nb_case) + 100
            with open("training_dataset_size.txt", "w") as f:
                f.write(str(nb_case))
        else:
            epochs = epochs + 2
        
        
        if training_parameters:
            # alimentation des paramêtres d'entrainements
            log_training_parameters(training_parameters)
            
            model.load_data_from_s3(cache_dir)
            
            mlflow.log_param("oversampling_strategy", "metadata_duplication")
            mlflow.log_param("augmentation_on_oversample_only", True)
            mlflow.log_param("oversample_ratio", model.oversampling_ratio)
            mlflow.log_param("augmentation_ops", "flip,rotate,zoom,brightness")
            
            model.fit(epochs=epochs)
            
            model.predict()
            classif = model.metrics["classification_report"]
            conf = model.metrics["confusion_matrix"]
            
            training_log_prod = asyncio.run(get_metrics_model_by_stage("prod"))
            
            prod_run = client.search_runs(
                experiment_ids=["1"],
                filter_string='tags.stage = "prod"'
            )
            
            candidate_run = client.search_runs(
                experiment_ids=["1"],
                filter_string='tags.stage = "candidate"'
            )
            
            if training_log_prod:
                if classif["1"]["recall"] > training_log_prod["class_1_recall"]:
                    stage = "candidate"
                elif classif["1"]["recall"] == training_log_prod["class_1_recall"] \
                    and model.metrics["accuracy"] > training_log_prod["accuracy"]:
                    stage = "candidate"
                else:
                    stage = "rejected"
            else:
                stage = "prod"
                
                
            # mise à jour des tags de prod et candidate
            if stage == "candidate" and candidate_run is None:
                pass
            elif stage == "candidate" and candidate_run is not None:
                client.set_tag(prod_run[0].info.run_id, "stage", "archived")
                client.set_tag(candidate_run[0].info.run_id, "stage", "prod")
                # mise à jour en base
                asyncio.run(update_stage(prod_run[0].info.run_id, "archived"))
                asyncio.run(update_stage(candidate_run[0].info.run_id, "prod"))
                # modification du registry
                prod_model = client.search_model_versions(f"run_id='{prod_run[0].info.run_id}'")
                candidate_model = client.search_model_versions(f"run_id='{candidate_run[0].info.run_id}'")
                client.transition_model_version_stage(
                    name=prod_model[0].name,
                    version=prod_model[0].version,
                    stage="Archived"
                )
                client.transition_model_version_stage(
                    name=candidate_model[0].name,
                    version=candidate_model[0].version,
                    stage="Production"
                )
            elif stage == "rejected" and candidate_run is not None:
                client.set_tag(prod_run[0].info.run_id, "stage", "archived")
                client.set_tag(candidate_run[0].info.run_id, "stage", "prod")
                # mise à jour en base
                asyncio.run(update_stage(prod_run[0].info.run_id, "archived"))
                asyncio.run(update_stage(candidate_run[0].info.run_id, "prod"))
                
                # modification du registry
                prod_model = client.search_model_versions(f"run_id='{prod_run[0].info.run_id}'")
                candidate_model = client.search_model_versions(f"run_id='{candidate_run[0].info.run_id}'")

                client.transition_model_version_stage(
                    name=prod_model[0].name,
                    version=prod_model[0].version,
                    stage="Archived"
                )
                client.transition_model_version_stage(
                    name=candidate_model[0].name,
                    version=candidate_model[0].version,
                    stage="Production"
                )

            # Alimentation des metrics d'entrainement dans la base de données
            training_log = {"training_date": datetime.strptime(now, "%Y-%m-%d-%H-%M-%f").isoformat(),
                    "model_name":model_name,
                    "run_id":run.info.run_id,
                    "stage":stage,
                    "training_size": model.nb_training_data,
                    "validation_size": model.nb_validation_data,
                    "epochs_number": len(model.history.history["loss"]),
                    "accuracy": float(model.metrics["accuracy"]),
                    "class_0_precision": float(classif["0"]["precision"]),
                    "class_0_recall": float(classif["0"]["recall"]),
                    "class_0_f1": float(classif["0"]["f1-score"]),
                    "class_1_precision": float(classif["1"]["precision"]),
                    "class_1_recall": float(classif["1"]["recall"]),
                    "class_1_f1": float(classif["1"]["f1-score"]),
                    "true_class_0": int(conf.loc[0,0]),
                    "false_class_0": int(conf.loc[0,1]),
                    "true_class_1": int(conf.loc[1,1]),
                    "false_class_1": int(conf.loc[1,0]),
                    "modification_date": datetime.strptime(now, "%Y-%m-%d-%H-%M-%f").isoformat()
                    }
            
            # log des metrics 
            mlflow.log_metric("class_1_recall", float(classif["1"]["recall"]))
            mlflow.log_metric("accuracy",float(model.metrics["accuracy"]))
            
            # log du graphique d'entrainement
            plt.figure()
            plt.plot(model.history.history['loss'], label='train_loss')
            plt.plot(model.history.history.get('val_loss', []), label='val_loss')
            plt.legend()
            plt.title("Training Loss")
            plt.savefig("./metrics/training_loss.png")
            mlflow.log_artifact("./metrics/training_loss.png", artifact_path="training_history")
            
            # conversion du batch pour la signature
            X_test = []
            y_test = []

            for batch_x, batch_y in model.test_gen:
                X_test.append(batch_x.numpy())
                y_test.append(batch_y.numpy())

            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)
            
            signature = infer_signature(X_test, y_test)
            
            # log des metrics en csv
            pd.DataFrame(classif).to_csv("./metrics/classification_report.csv")
            pd.DataFrame(conf).to_csv("./metrics/confusion_matrix.csv")

            mlflow.log_artifact("./metrics/classification_report.csv", artifact_path="classification_report")
            mlflow.log_artifact("./metrics/confusion_matrix.csv", artifact_path="confusion_matrix")
            
            # log des metrics en graphique
            plt.figure(figsize=(5,4))
            sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.savefig("./metrics/confusion_matrix.png")
            mlflow.log_artifact("./metrics/confusion_matrix.png", artifact_path="confusion_matrix")
            
            fig = plot_classification_report(classif)
            fig.savefig("./metrics/classification_report_plot.png")
            mlflow.log_artifact("./metrics/classification_report_plot.png", artifact_path="classification_report")
            plt.close(fig)
            
            # on enregistre les metrics
            asyncio.run(post_metrics(training_log))
            
            # sauvegarde du model
            model.save()
            
            print("sauvegarde du model ok")
            mlflow.log_param("dvc_path", f"./models/{model_name}")
            mlflow.log_param("dvc_models_hash", get_dvc_hash(f"models.dvc"))
            mlflow.log_param("dvc_dataset_hash", get_dvc_hash(f"dataset.dvc"))
            mlflow.log_param("dvc_metrics_hash", get_dvc_hash(f"metrics.dvc"))
            
            # Alimentation du model dans MLFlow
            mlflow.set_tag("type", "model")
            mlflow.set_tag("status", stage)
            mlflow.keras.log_model(model.model, name="model", signature=signature)
            model_uri = "runs:/{}/model".format(run.info.run_id)
            model_version = mlflow.register_model(model_uri, model_name)
            
            if stage == "prod":
                                
                client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Production",
                        archive_existing_versions=True
                    )
                
            if  stage == "candidate":
                client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Staging",
                        archive_existing_versions=True
                    )
 
            if cache_dir_path.exists():
                shutil.rmtree(cache_dir_path)

        else:
            raise ValueError("Aucun paamêtre trouvé dans la table parameters")
        
def create_model_instance(training_parameters, model_name):
    
    # Entrainement du modèle
    model = EfficientNetv2B0_model_augmented(   
                    data_folder="",
                    save_model_folder=Path("models"),
                    model_name=model_name,
                    img_size=(int(training_parameters["img_width"]), int(training_parameters["img_height"])),
                    gray=boolean(training_parameters["gray_mode"]),
                    batch_size=int(training_parameters["batch_size"]),
                    big_dataset=False,
                    train_size=float(training_parameters["train_size"]),
                    random_state=int(training_parameters["random_state"]),
                    oversampling=True,
                    nb_layer_to_freeze=int(training_parameters["nb_layer_to_freeze"]),
                    es_patience=int(training_parameters["es_patience"]),
                    es_min_delta=float(training_parameters["es_min_delta"]),
                    es_mode=training_parameters["es_mode"],
                    es_monitor=training_parameters["es_monitor"],
                    rlrop_monitor=training_parameters["rlrop_monitor"],
                    rlrop_patience=int(training_parameters["rlrop_patience"]),
                    rlrop_min_delta=float(training_parameters["rlrop_min_delta"]),
                    rlrop_factor=float(training_parameters["rlrop_factor"]),
                    rlrop_cooldown=int(training_parameters["rlrop_cooldown"]),
                    loss_cat=training_parameters["loss_cat"],
                    optimizer_name=training_parameters["optimizer_name"],
                    metrics=[training_parameters["metrics"]]
    )
            
    return model

def get_dvc_hash(dvc_file: str) -> str:
    with open(dvc_file, "r") as f:
        dvc_data = yaml.safe_load(f)
    return dvc_data["outs"][0]["md5"]