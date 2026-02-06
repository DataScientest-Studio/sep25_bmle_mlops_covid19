import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path().resolve()))
from src.data.database_access import DatabaseAccess
from src.settings import DatabaseSettings



async def get_parameters():
    settings = DatabaseSettings("secrets.yaml")
    url, key = settings.database_url
    db = DatabaseAccess(api_url=url, api_key=key)

    params = {"order": "validity_date.desc",
            "limit": 1
    }

    param = await db.select("parameters", params=params)

    return param
    
import asyncio
print(asyncio.run(get_parameters()))

