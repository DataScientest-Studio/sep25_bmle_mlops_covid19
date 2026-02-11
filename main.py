import asyncio
import os
from pathlib import Path
from xmlrpc.client import boolean

import mlflow
from datetime import datetime

import mlflow.models
import numpy as np
from src.models.models_to_test import EfficientNetv2B0_model_augmented
from src.utils.database_utils import get_parameters, post_metrics, get_metrics_prod_model
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
            
            model.fit(epochs=1)
            
            model.predict()
            classif = model.metrics["classification_report"]
            conf = model.metrics["confusion_matrix"]
            
            training_log = asyncio.run(get_metrics_prod_model())
            
            if training_log:
                if model.metrics["accuracy"] >= training_log["accuracy"] and classif["1"]["recall"] >= training_log["class_1_recall"]:
                    stage = "candidate"
                else:
                    stage = "rejected"
            else:
                stage = "prod"

            # Alimentation des metrics d'entrainement dans MLFlow
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
            
            # on enregistre les metrics
            asyncio.run(post_metrics(training_log))
            
            # Alimentation du model dans MLFlow
            mlflow.set_tag("type", "model")
            mlflow.set_tag("status", stage)
            mlflow.keras.log_model(model.model, name="model")
            
            # conversion du dataset de validation
            X_test = []
            y_test = []

            for batch_x, batch_y in model.test_gen:
                X_test.append(batch_x.numpy())
                y_test.append(batch_y.numpy())

            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)
            
            #log_metrics_from_dict(metrics)
            model_uri = f"runs:/{run.info.run_id}/model"
            eval_result = mlflow.models.evaluate(
                model=model_uri,
                data=X_test,
                targets=y_test,
                model_type="keras",
                evaluator_config={
                    "explainability": True  
                }
            )
        else:
            raise ValueError("Aucun paamêtre trouvé dans la table parameters")
        
        
if __name__ == "__main__":
    
    main_mlflow()