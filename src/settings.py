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
        host = (db.get("host") or "").strip()
        password = (db.get("password") or "").strip()
        # Si l'URL n'a pas de protocole, construire l'URL REST Supabase
        if host and not host.startswith(("http://", "https://")):
            if host.startswith("db.") and "supabase.co" in host:
                # db.PROJECT.supabase.co -> https://PROJECT.supabase.co/rest/v1
                project = host[3:].split(".")[0]  # "hkjhkruzfvglrgcomqzj"
                host = f"https://{project}.supabase.co/rest/v1"
            else:
                host = f"https://{host}"
        return host, password
        
class S3Settings:
    
    def __init__(self, secrets_path: str = "secrets.yaml"):
        secrets_file = Path(secrets_path)

        if not secrets_file.exists():
            raise FileNotFoundError("secrets.yaml not found")

        with open(secrets_file, "r") as f:
            self._secrets = yaml.safe_load(f)

    @property
    def s3_access(self) -> tuple:
        # Accepter "S3" ou "s3" (sensible à la casse dans le YAML)
        print(f"{self._secrets = }")
        s3 = self._secrets.get("S3") or self._secrets.get("s3")
        print(f"{s3 = }")
        if not s3:
            raise KeyError(
                "Cle 'S3' ou 's3' manquante dans le fichier de secrets. "
                "Ajoutez une section S3: avec bucket_name, access_key, secret_key, b2_endpoint."
            )
        # Nettoyer espaces/newlines (copier-coller, YAML)
        bucket = str(s3.get("bucket_name", "")).strip()
        access = str(s3.get("access_key", "")).strip()
        secret = str(s3.get("secret_key", "")).strip()
        endpoint = str(s3.get("b2_endpoint", "")).strip()
        return bucket, access, secret, endpoint
