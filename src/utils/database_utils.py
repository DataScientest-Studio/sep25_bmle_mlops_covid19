import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.database_access import DatabaseAccess
from src.settings import DatabaseSettings

async def get_parameters():
    
    root = Path(__file__).resolve().parent.parent.parent
    secrets_path = root / "secrets.yaml"

    settings = DatabaseSettings(str(secrets_path))
    url, key = settings.database_url
    db = DatabaseAccess(api_url=url, api_key=key)

    params = {"order": "validity_date.desc",
            "limit": 1
    }

    param = await db.select("parameters", params=params)

    return param[0]

async def post_parameters(data):

    root = Path(__file__).resolve().parent.parent.parent
    secrets_path = root / "secrets.yaml"
    
    settings = DatabaseSettings(str(secrets_path))
    url, key = settings.database_url
    db = DatabaseAccess(api_url=url, api_key=key)

    response = await db.insert("parameters", data=data)
    

