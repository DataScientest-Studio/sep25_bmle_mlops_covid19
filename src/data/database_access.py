from typing import Any, Dict, Optional, AsyncIterator
from contextlib import asynccontextmanager
import httpx

class DatabaseAccess:
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/") + "/"  # s'assure qu'il y a un slash final
        self.headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(headers=self.headers)

    @asynccontextmanager
    async def get_session(self) -> AsyncIterator[httpx.AsyncClient]:
        """
        Fournit un client HTTP asynchrone pour faire des requêtes à la Data API.
        """
        try:
            yield self.client
        finally:
            await self.client.aclose()

    # -------- CREATE --------

    async def select(self, table: str, params: Optional[Dict[str, Any]] = None):
        """
        Récupérer des lignes d'une table.
        """
        url = f"{self.api_url}{table}"
        async with self.get_session() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def insert(self, table: str, data: Dict[str, Any]):
        """
        Insérer une ligne dans une table.
        """
        url = f"{self.api_url}{table}"
        async with self.get_session() as client:
            response = await client.post(url, json=data)
            response.raise_for_status()
            return response.json()

    async def update(self, table: str, data: Dict[str, Any], match: Dict[str, Any]):
        """
        Mettre à jour des lignes d'une table.
        """
        url = f"{self.api_url}{table}"
        async with self.get_session() as client:
            response = await client.patch(url, json=data, params=match)
            response.raise_for_status()
            return response.json()

    async def delete(self, table: str, match: Dict[str, Any]):
        """
        Supprimer des lignes d'une table.
        """
        url = f"{self.api_url}{table}"
        async with self.get_session() as client:
            response = await client.delete(url, params=match)
            response.raise_for_status()
            return response.json()