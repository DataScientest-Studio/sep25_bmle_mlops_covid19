import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
from src.data.database_access import DatabaseAccess
from src.data.image_dataset import ImageDataset
from src.settings import Settings


async def get_parameters():
    settings = Settings("secrets.yaml")
    print(f"{settings.database_url = }")
    db = DatabaseAccess(settings.database_url)
    print(f"{db = }")
    image = ImageDataset(
        image_url="https://test",
        mask_url="https://mask",
        class_type="class_1",
        injection_date=None,
        created_at=None,
    )

    await db.insert(image)
    
    
import asyncio
asyncio.run(get_parameters())

