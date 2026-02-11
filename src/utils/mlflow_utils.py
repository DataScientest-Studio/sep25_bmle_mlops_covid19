import asyncio
import mlflow
import sys
from pathlib import Path
from typing import Dict, Union

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.database_utils import get_parameters


def log_training_parameters(params: dict):
    
    mlflow.set_tag("type", "parameters")
    mlflow.log_params({
        k: str(v) if not isinstance(v, (int, float, bool)) else v
        for k, v in params.items()
    })
    
def log_parameters_to_mlflow():
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("COVID 19")
    
    with mlflow.start_run(run_name="traning_parameters_update") as run:
        training_parameters = asyncio.run(get_parameters())
        
        log_training_parameters(training_parameters)
        
def log_metrics_from_dict(
    metrics: Dict[str, Union[int, float]],
) -> None:
    """
    Log des métriques MLflow à partir d'un dictionnaire.

    Args:
        metrics: dictionnaire {nom_metric: valeur}
    """

    for name, value in metrics.items():

        if value is None:
            continue

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Metric '{name}' doivent être int ou float"
            )
        mlflow.set_tag("type", "metrics")
        mlflow.log_metric(name, float(value))