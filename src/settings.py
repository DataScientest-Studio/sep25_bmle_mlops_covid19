import yaml
from pathlib import Path

class DatabaseSettings:
    
    def __init__(self, secrets_path: str = "../secrets.yaml"):
        secrets_file = Path(secrets_path)

        if not secrets_file.exists():
            raise FileNotFoundError("secrets.yaml not found")

        with open(secrets_file, "r") as f:
            self._secrets = yaml.safe_load(f)

    @property
    def database_url(self) -> tuple:
        db = self._secrets["database"]
        return db['host'], db['password']
        
class S3Settings:
    
    def __init__(self, secrets_path: str = "secrets.yaml"):
        secrets_file = Path(secrets_path)

        if not secrets_file.exists():
            raise FileNotFoundError("secrets.yaml not found")

        with open(secrets_file, "r") as f:
            self._secrets = yaml.safe_load(f)

    @property
    def s3_access(self) -> tuple:
        s3 = self._secrets["S3"]
        return s3["bucket_name"], s3["access_key"], s3["secret_key"], s3["b2_endpoint"]
