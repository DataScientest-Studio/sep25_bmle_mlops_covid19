from src.models.train_model_mlflow import train_model_mlflow
from src.settings import S3Settings
from src.utils.s3_utils import upload_dataset_and_generate_csv


if __name__ == "__main__":
    
    train_model_mlflow()
    
    """settings = S3Settings("secrets.yaml")
    
    bucket_name, access_key, secret_key, b2_endpoint = settings.s3_access

    upload_dataset_and_generate_csv(
        bucket_name=bucket_name,
        access_key=access_key,
        secret_key=secret_key,
        local_root="./data/COVID-19_Radiography_Dataset",
        s3_prefix="dataset",
        output_csv="dataset.csv"
    )"""