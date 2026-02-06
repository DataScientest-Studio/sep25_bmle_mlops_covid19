import yaml
from pathlib import Path


class Settings:
    def __init__(self, secrets_path: str = "secrets.yaml"):
        secrets_file = Path(secrets_path)

        if not secrets_file.exists():
            raise FileNotFoundError("secrets.yaml not found")

        with open(secrets_file, "r") as f:
            self._secrets = yaml.safe_load(f)

    @property
    def database_url(self) -> str:
        db = self._secrets["database"]
        return (
            f"postgresql+asyncpg://"
            f"{db['user']}:{db['password']}"
            f"@{db['host']}:{db['port']}/{db['name']}"
        )
