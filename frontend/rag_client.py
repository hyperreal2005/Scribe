import os
from typing import Any, Dict, List, Optional

import httpx

RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8001")


async def ingest_text(
    text: str,
    *,
    source: Optional[str] = None,
    title: Optional[str] = None,
    path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    scope_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not text.strip():
        return {"chunks": 0, "document_id": None}
    payload = {
        "text": text,
        "source": source,
        "title": title,
        "path": path,
        "metadata": metadata or {},
        "scope_id": scope_id,
    }
    async with httpx.AsyncClient(base_url=RAG_BACKEND_URL, timeout=60.0) as client:
        resp = await client.post("/ingest/text", json=payload)
        resp.raise_for_status()
        return resp.json()


async def retrieve(
    query: str,
    top_k: int = 5,
    *,
    scope_id: Optional[str] = None,
    include_global: bool = True,
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    payload = {
        "query": query,
        "top_k": top_k,
        "scope_id": scope_id,
        "include_global": include_global,
    }
    async with httpx.AsyncClient(base_url=RAG_BACKEND_URL, timeout=60.0) as client:
        resp = await client.post("/retrieve", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])


async def chat(
    message: str,
    *,
    top_k: int = 5,
    scope_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    uploaded_text: Optional[str] = None,
    use_web_search: bool = False,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    payload = {
        "message": message,
        "top_k": top_k,
        "scope_id": scope_id,
        "system_prompt": system_prompt,
        "uploaded_text": uploaded_text,
        "use_web_search": use_web_search,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(base_url=RAG_BACKEND_URL, timeout=120.0) as client:
        resp = await client.post("/chat", json=payload)
        resp.raise_for_status()
        return resp.json()
