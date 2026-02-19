from datetime import datetime
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

    if param:
        return param[0]
    else:
        return None

async def post_parameters(data):

    root = Path(__file__).resolve().parent.parent.parent
    secrets_path = root / "secrets.yaml"
    
    settings = DatabaseSettings(str(secrets_path))
    url, key = settings.database_url
    db = DatabaseAccess(api_url=url, api_key=key)

    response = await db.insert("parameters", data=data)
    
async def get_metrics_model_by_stage(stage):
    
    root = Path(__file__).resolve().parent.parent.parent
    secrets_path = root / "secrets.yaml"

    settings = DatabaseSettings(str(secrets_path))
    url, key = settings.database_url
    db = DatabaseAccess(api_url=url, api_key=key)

    params = {"stage": f"eq.{stage}"}

    training_log = await db.select("training_log", params=params)

    if training_log:
        return training_log[0]
    else:
        return None
    
async def post_metrics(data):

    root = Path(__file__).resolve().parent.parent.parent
    secrets_path = root / "secrets.yaml"
    
    settings = DatabaseSettings(str(secrets_path))
    url, key = settings.database_url

    db = DatabaseAccess(api_url=url, api_key=key)

    response = await db.insert("training_log", data=data)
    
async def fetch_dataset():

    root = Path(__file__).resolve().parent.parent.parent
    secrets_path = root / "secrets.yaml"
    
    settings = DatabaseSettings(str(secrets_path))
    url, key = settings.database_url

    db = DatabaseAccess(api_url=url, api_key=key)

    response = await db.fetch_all("images_dataset")
    
    return response

async def update_stage(run_id, new_stage):

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%f")

    root = Path(__file__).resolve().parent.parent.parent
    secrets_path = root / "secrets.yaml"
    
    settings = DatabaseSettings(str(secrets_path))
    url, key = settings.database_url

    db = DatabaseAccess(api_url=url, api_key=key)
    
    data = {"stage":new_stage,
            "modification_date": datetime.strptime(now, "%Y-%m-%d-%H-%M-%f").isoformat()
            }
    
    match = {"run_id":run_id}

    response = await db.update("training_log", data=data, match=match)
    
    return response
    

