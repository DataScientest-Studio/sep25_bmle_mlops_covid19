from typing import Any, Dict, Optional
import httpx


class DatabaseAccess:
    """
    Accès à la Data API Supabase. Utilise un nouveau client HTTP par opération
    pour éviter "Cannot send a request, as the client has been closed" quand
    on enchaîne plusieurs appels (ex. update puis insert).
    """

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/") + "/"
        self.headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self.headers)

    async def select(self, table: str, params: Optional[Dict[str, Any]] = None):
        """Récupérer des lignes d'une table."""
        url = f"{self.api_url}{table}"
        async with self._client() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def insert(
        self,
        table: str,
        data: Dict[str, Any],
        *,
        return_representation: bool = False,
    ):
        """
        Insérer une ligne dans une table.
        Si return_representation=True, envoie Prefer: return=representation pour récupérer la ligne insérée (ex. id).
        """
        url = f"{self.api_url}{table}"
        headers = dict(self.headers)
        if return_representation:
            headers["Prefer"] = "return=representation"
        async with self._client() as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            body = (response.text or "").strip()
            if not body:
                return {}
            try:
                out = response.json()
                if return_representation and isinstance(out, list) and len(out) == 1:
                    return out[0]
                return out
            except Exception:
                return {}

    async def update(self, table: str, data: Dict[str, Any], match: Dict[str, Any]):
        """Mettre à jour des lignes d'une table. Retourne les lignes mises à jour (PostgREST)."""
        url = f"{self.api_url}{table}"
        headers = {**self.headers, "Prefer": "return=representation"}
        async with self._client() as client:
            response = await client.patch(url, json=data, params=match, headers=headers)
            response.raise_for_status()
            return response.json()

    async def delete(self, table: str, match: Dict[str, Any]):
        """Supprimer des lignes d'une table."""
        url = f"{self.api_url}{table}"
        async with self._client() as client:
            response = await client.delete(url, params=match)
            response.raise_for_status()
            return response.json()