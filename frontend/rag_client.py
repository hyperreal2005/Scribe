import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8001")


async def ingest_text(
    text: str,
    *,
    source: Optional[str] = None,
    title: Optional[str] = None,
    path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not text.strip():
        return {"chunks": 0, "document_id": None}
    payload = {
        "text": text,
        "source": source,
        "title": title,
        "path": path,
        "metadata": metadata or {},
    }
    async with httpx.AsyncClient(base_url=RAG_BACKEND_URL, timeout=60.0) as client:
        resp = await client.post("/ingest/text", json=payload)
        resp.raise_for_status()
        return resp.json()


async def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient(base_url=RAG_BACKEND_URL, timeout=60.0) as client:
        resp = await client.post("/retrieve", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
