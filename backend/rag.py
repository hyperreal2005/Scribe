import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from google import genai
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .db import RagDb, get_db
from .ingest import extract_text_from_path, scan_folder
from .logger import get_ai_logger

logger = logging.getLogger(__name__)
ai_logger = get_ai_logger()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: List[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            space = cleaned.rfind(" ", start, end)
            if space > start + int(chunk_size * 0.6):
                end = space
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, end)
        if start == end:
            start += 1
    return chunks


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RetrievedChunk:
    text: str
    score: float
    payload: Dict[str, Any]


class RagService:
    def __init__(self) -> None:
        self.embed_model = os.getenv("RAG_EMBED_MODEL", "text-embedding-004")
        self.collection = os.getenv("RAG_COLLECTION", "rag_chunks")
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.docs_path = os.getenv("RAG_DOCS_PATH")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.genai_client = genai.Client(api_key=api_key)
        self.qdrant = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.db: RagDb = get_db()

    async def startup(self) -> None:
        await self.db.connect()
        await self.db.init_schema()

    async def shutdown(self) -> None:
        await self.db.close()
        await self.qdrant.close()

    async def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        await self._log_ai_call(
            purpose="embed_content",
            model=self.embed_model,
            request=texts,
            response=None,
        )
        response = await self.genai_client.aio.models.embed_content(
            model=self.embed_model,
            contents=list(texts),
        )
        embeddings = response.embeddings or []
        vectors: List[List[float]] = []
        for emb in embeddings:
            if not emb.values:
                raise RuntimeError("Embedding returned empty values")
            vectors.append(list(emb.values))
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding count mismatch")
        await self._log_ai_call(
            purpose="embed_content_result",
            model=self.embed_model,
            request=None,
            response=f"{len(vectors)} embeddings",
        )
        return vectors

    async def _ensure_collection(self, dim: int) -> None:
        try:
            info = await self.qdrant.get_collection(self.collection)
            existing = info.config.params.vectors.size  # type: ignore[attr-defined]
            if existing != dim:
                logger.warning(
                    "Qdrant collection dimension mismatch (%s != %s).",
                    existing,
                    dim,
                )
            return
        except Exception:
            await self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def ingest_text(
        self,
        text: str,
        *,
        source: Optional[str] = None,
        title: Optional[str] = None,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not text.strip():
            return {"chunks": 0, "document_id": None}
        doc_hash = _hash_text(text)
        document_id = await self.db.insert_document(
            source=source,
            title=title,
            path=path,
            doc_hash=doc_hash,
            metadata=metadata or {},
        )
        chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return {"chunks": 0, "document_id": document_id}
        vectors = await self._embed_texts(chunks)
        await self._ensure_collection(len(vectors[0]))

        points: List[PointStruct] = []
        chunk_records: List[tuple[str, int, str, Dict[str, Any]]] = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            qdrant_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "doc_id": document_id,
                "title": title,
                "source": source,
                "path": path,
                "chunk_index": i,
                "metadata": metadata or {},
            }
            points.append(PointStruct(id=qdrant_id, vector=vec, payload=payload))
            chunk_records.append((qdrant_id, i, chunk, payload))

        await self.qdrant.upsert(collection_name=self.collection, points=points)
        await self.db.insert_chunks(document_id, chunk_records)

        return {"chunks": len(chunks), "document_id": document_id}

    async def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not query.strip():
            return []
        vectors = await self._embed_texts([query])
        vector = vectors[0]
        await self._ensure_collection(len(vector))
        response = await self.qdrant.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        results = response.points
        chunks: List[RetrievedChunk] = []
        for r in results:
            payload = r.payload or {}
            chunks.append(
                RetrievedChunk(
                    text=str(payload.get("text", "")),
                    score=float(r.score),
                    payload=dict(payload),
                )
            )
        return chunks

    async def ingest_folder(self, folder: str, recursive: bool = True) -> Dict[str, Any]:
        path = os.path.abspath(folder)
        files = scan_folder(Path(path), recursive=recursive)
        ingested = 0
        for file_path in files:
            text = extract_text_from_path(file_path)
            if not text:
                continue
            await self.ingest_text(
                text,
                source="folder",
                title=file_path.name,
                path=str(file_path),
                metadata={"ext": file_path.suffix.lower()},
            )
            ingested += 1
        return {"files_ingested": ingested}

    async def _log_ai_call(
        self,
        *,
        purpose: str,
        model: str,
        request: Optional[Any],
        response: Optional[Any],
    ) -> None:
        try:
            prompt_tokens = None
            if request is not None:
                try:
                    count = await self.genai_client.aio.models.count_tokens(
                        model=model,
                        contents=request,
                    )
                    prompt_tokens = getattr(count, "total_tokens", None)
                except Exception:
                    prompt_tokens = None

            response_tokens = None
            if response is not None:
                try:
                    count = await self.genai_client.aio.models.count_tokens(
                        model=model,
                        contents=str(response),
                    )
                    response_tokens = getattr(count, "total_tokens", None)
                except Exception:
                    response_tokens = None

            total_tokens = None
            if prompt_tokens is not None or response_tokens is not None:
                total_tokens = (prompt_tokens or 0) + (response_tokens or 0)

            logger.info(
                "AI_CALL purpose=%s model=%s prompt_tokens=%s response_tokens=%s total_tokens=%s request=%s response=%s",
                purpose,
                model,
                prompt_tokens,
                response_tokens,
                total_tokens,
                _preview(request),
                _preview(response),
            )
            ai_logger.info(
                "AI_CALL purpose=%s model=%s prompt_tokens=%s response_tokens=%s total_tokens=%s request=%s response=%s",
                purpose,
                model,
                prompt_tokens,
                response_tokens,
                total_tokens,
                _preview(request),
                _preview(response),
            )
        except Exception as exc:
            logger.warning("AI_CALL logging failed: %s", exc)


def _preview(value: Any, limit: int = 1200) -> str:
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value[:5]:
            parts.append(_preview(item, limit=300))
        extra = "" if len(value) <= 5 else f"...(+{len(value)-5} more)"
        return "[" + ", ".join(parts) + f"]{extra}"
    text = str(value)
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text
