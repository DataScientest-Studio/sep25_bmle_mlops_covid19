import asyncio
import mlflow
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.database_utils import get_parameters


def log_training_parameters(params: dict):
    
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