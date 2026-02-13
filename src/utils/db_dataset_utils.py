import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from src.data.database_access import DatabaseAccess
from src.settings import DatabaseSettings


def _normalize_class(value: Any) -> str:
    if value in (1, "1", True, "COVID", "covid", "Covid"):
        return "1"
    return "0"


def _safe_filename(url: str, index: int) -> str:
    basename = os.path.basename(urlsplit(url).path)
    if not basename:
        basename = f"image_{index}.png"
    name, ext = os.path.splitext(basename)
    if ext.lower() != ".png":
        ext = ".png"
    return f"{index:08d}_{name}{ext}"


async def _select_page(
    table: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    settings = DatabaseSettings("secrets.yaml")
    api_url, api_key = settings.database_url
    db = DatabaseAccess(api_url=api_url, api_key=api_key)
    return await db.select(table, params=params)


async def fetch_image_rows(
    table: str = "images_dataset",
    limit: Optional[int] = None,
    batch_size: int = 1000,
    order: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0

    while True:
        page_size = batch_size
        if limit is not None:
            remaining = limit - len(rows)
            if remaining <= 0:
                break
            page_size = min(batch_size, remaining)

        params: Dict[str, Any] = {
            "select": "image_url,mask_url,class_type",
            "limit": page_size,
            "offset": offset,
        }
        if order:
            params["order"] = order

        page = await _select_page(table, params=params)
        if not page:
            break

        rows.extend(page)
        offset += len(page)

        if len(page) < page_size:
            break

    return rows


def fetch_image_rows_sync(
    table: str = "images_dataset",
    limit: Optional[int] = None,
    batch_size: int = 1000,
    order: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return asyncio.run(
        fetch_image_rows(
            table=table,
            limit=limit,
            batch_size=batch_size,
            order=order,
        )
    )


async def fetch_image_row_by_id(
    image_id: int,
    table: str = "images_dataset",
) -> Dict[str, Any]:
    params = {
        "select": "image_url,mask_url,class_type",
        "id": f"eq.{image_id}",
        "limit": 1,
    }
    rows = await _select_page(table, params=params)
    if not rows:
        raise ValueError(f"No record found for id={image_id}")
    return rows[0]


def download_dataset_from_rows(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    apply_masks: bool = True,
    timeout: float = 20.0,
    replace: bool = True,
) -> int:
    if replace and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in ["0", "1"]:
        os.makedirs(output_dir / cls, exist_ok=True)

    saved = 0
    with httpx.Client(timeout=timeout) as client:
        for idx, row in enumerate(rows):
            image_url = row.get("image_url")
            if not image_url:
                continue

            try:
                resp = client.get(image_url)
                resp.raise_for_status()
            except httpx.HTTPError:
                continue

            image = cv2.imdecode(
                np.frombuffer(resp.content, np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                continue

            if apply_masks and row.get("mask_url"):
                try:
                    mask_resp = client.get(row["mask_url"])
                    mask_resp.raise_for_status()
                except httpx.HTTPError:
                    mask_resp = None

                if mask_resp is not None:
                    mask = cv2.imdecode(
                        np.frombuffer(mask_resp.content, np.uint8),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    if mask is not None:
                        if image.shape[:2] != mask.shape[:2]:
                            mask = cv2.resize(
                                mask,
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        mask = (mask > 0).astype(np.uint8)
                        image = image * mask[:, :, None]

            class_label = _normalize_class(row.get("class_type"))
            filename = _safe_filename(image_url, idx)
            out_path = output_dir / class_label / filename
            cv2.imwrite(str(out_path), image)
            saved += 1

    return saved
