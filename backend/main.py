import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import RagService

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Scribe RAG Backend", version="0.2.0")
rag = RagService()


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: Optional[str] = None
    title: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    scope_id: Optional[str] = None


class IngestFolderRequest(BaseModel):
    path: str
    recursive: bool = True


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    scope_id: Optional[str] = None
    include_global: bool = True


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    scope_id: Optional[str] = None
    system_prompt: Optional[str] = None
    uploaded_text: Optional[str] = None
    use_web_search: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


@app.on_event("startup")
async def on_startup() -> None:
    await rag.startup()
    docs_path = os.getenv("RAG_DOCS_PATH")
    if docs_path:
        logger.info("Scheduling folder ingest: %s", docs_path)
        # Fire-and-forget to avoid blocking startup
        import asyncio

        asyncio.create_task(rag.ingest_folder(docs_path))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await rag.shutdown()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/ingest/text")
async def ingest_text(req: IngestTextRequest) -> Dict[str, Any]:
    try:
        return await rag.ingest_text(
            req.text,
            source=req.source,
            title=req.title,
            path=req.path,
            metadata=req.metadata,
            scope_id=req.scope_id,
        )
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest/folder")
async def ingest_folder(req: IngestFolderRequest) -> Dict[str, Any]:
    try:
        return await rag.ingest_folder(req.path, recursive=req.recursive)
    except Exception as exc:
        logger.exception("Folder ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest) -> Dict[str, Any]:
    try:
        chunks = await rag.retrieve(
            req.query,
            top_k=req.top_k,
            scope_id=req.scope_id,
            include_global=req.include_global,
        )
        return {
            "results": [
                {"text": c.text, "score": c.score, "payload": c.payload} for c in chunks
            ]
        }
    except Exception as exc:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    try:
        return await rag.chat(
            req.message,
            system_prompt=req.system_prompt,
            uploaded_text=req.uploaded_text,
            scope_id=req.scope_id,
            top_k=req.top_k,
            use_web_search=req.use_web_search,
            temperature=req.temperature,
        )
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
