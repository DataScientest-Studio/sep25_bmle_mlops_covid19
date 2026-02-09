import asyncio

import mlflow

from src.utils.database_utils import get_parameters
from src.utils.mlflow_utils import log_training_parameters


def main_mlflow():
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Default")

    training_parameters = asyncio.run(get_parameters())
        
    with mlflow.start_run(run_name="Training_parameters"):
        log_training_parameters(training_parameters)
        
        
if __name__ == "__main__":
    
    main_mlflow()