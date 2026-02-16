from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests

def trigger_training():
    response = requests.post("http://localhost:8001/train")
    print(response.json())

with DAG(
    dag_id="covid_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(hours=4), 
    catchup=False
) as dag:

    train_task = PythonOperator(
        task_id="trigger_train_api",
        python_callable=trigger_training
    )