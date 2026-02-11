import asyncio
import os
from pathlib import Path
from xmlrpc.client import boolean
from matplotlib import pyplot as plt
import seaborn as sns
import mlflow
from datetime import datetime
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature
from src.models.models_to_test import EfficientNetv2B0_model_augmented
from src.utils.data_utils import plot_classification_report
from src.utils.database_utils import get_parameters, post_metrics, get_metrics_model_by_stage
from src.utils.mlflow_utils import  log_training_parameters


def main_mlflow():
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%f")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(f"COVID_19_test2")
    
    # Paths
    PROJECT_ROOT = Path(os.getcwd())  # racine du projet\n",
    DATASET_ROOT = PROJECT_ROOT / "data" # racine du dataset
    MODELS_FOLDER = PROJECT_ROOT / "models"
    dataset_root = DATASET_ROOT / "structured_dataset"

    training_parameters = asyncio.run(get_parameters())
        
    with mlflow.start_run(run_name=f"Training_{now}") as run:
        
        if training_parameters:
            # alimentation des paramêtres d'entrainements
            log_training_parameters(training_parameters)

            model_name = f"model_{now}.keras"
            
            # Entrainement du modèle
            model = EfficientNetv2B0_model_augmented(   
                    data_folder=dataset_root,
                    save_model_folder=MODELS_FOLDER,
                    model_name=model_name,
                    img_size=(int(training_parameters["img_width"]), int(training_parameters["img_height"])),
                    gray=boolean(training_parameters["gray_mode"]),
                    batch_size=int(training_parameters["batch_size"]),
                    big_dataset=False,
                    train_size=float(training_parameters["train_size"]),
                    random_state=int(training_parameters["random_state"]),
                    oversampling=False,
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
            
            model.load_data()
            
            model.fit(epochs=5)
            
            model.predict()
            classif = model.metrics["classification_report"]
            conf = model.metrics["confusion_matrix"]
            
            training_log_prod = asyncio.run(get_metrics_model_by_stage("prod"))
            
            if training_log_prod:
                if model.metrics["accuracy"] >= training_log_prod["accuracy"] and classif["1"]["recall"] >= training_log_prod["class_1_recall"]:
                    stage = "candidate"
                else:
                    stage = "rejected"
            else:
                stage = "prod"

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
                    "false_class_1": int(conf.loc[1,0])
                    }
            
            # log du graphique d'entrainement
            plt.figure()
            plt.plot(model.history.history['loss'], label='train_loss')
            plt.plot(model.history.history.get('val_loss', []), label='val_loss')
            plt.legend()
            plt.title("Training Loss")
            plt.savefig("training_loss.png")
            mlflow.log_artifact("training_loss.png", artifact_path="training_history")
            
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
            pd.DataFrame(classif).to_csv("classification_report.csv")
            pd.DataFrame(conf).to_csv("confusion_matrix.csv")

            mlflow.log_artifact("classification_report.csv", artifact_path="classification_reports")
            mlflow.log_artifact("confusion_matrix.csv", artifact_path="confusion_matrix")
            
            # log des metrics en graphique
            plt.figure(figsize=(5,4))
            sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png", artifact_path="confusion_matrix")
            
            fig = plot_classification_report(classif)
            fig.savefig("classification_report_plot.png")
            mlflow.log_artifact("classification_report_plot.png", artifact_path="classification_report")
            plt.close(fig)
            
            # on enregistre les metrics
            asyncio.run(post_metrics(training_log))
            
            # Alimentation du model dans MLFlow
            mlflow.set_tag("type", "model")
            mlflow.set_tag("status", stage)
            mlflow.keras.log_model(model.model, name="model", signature=signature)
 
            #log_metrics_from_dict(metrics)
        else:
            raise ValueError("Aucun paamêtre trouvé dans la table parameters")
        
        
if __name__ == "__main__":
    
    main_mlflow()