import asyncio
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from google import genai
from google.genai import types as genai_types
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from .db import RagDb, get_db
from .ingest import extract_text_from_path, scan_folder
from .logger import get_ai_logger

logger = logging.getLogger(__name__)
ai_logger = get_ai_logger()

DEFAULT_SCOPE_ID = "global"
DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are Scribe, a helpful educational assistant. "
    "Ground your answers in retrieved context when available, "
    "and explicitly say when the answer is not present in the provided context."
)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

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

        # Preserve overlap for better semantic continuity between adjacent chunks.
        next_start = max(0, end - overlap)

        # Avoid starting inside a word after applying overlap.
        if next_start > 0 and cleaned[next_start - 1] != " ":
            next_space = cleaned.find(" ", next_start)
            if next_space != -1:
                next_start = next_space + 1

        start = next_start if next_start > start else end
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
        self.embed_model = os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001")
        self.chat_model = os.getenv("RAG_CHAT_MODEL", "gemini-2.5-flash")
        self.collection = os.getenv("RAG_COLLECTION", "rag_chunks")
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.docs_path = os.getenv("RAG_DOCS_PATH")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_timeout_seconds = int(os.getenv("RAG_QDRANT_TIMEOUT_SECONDS", "90"))
        self.qdrant_upsert_batch_size = int(os.getenv("RAG_QDRANT_UPSERT_BATCH_SIZE", "64"))
        self.recreate_collection_on_dim_mismatch = (
            os.getenv("RAG_RECREATE_COLLECTION_ON_DIM_MISMATCH", "true").strip().lower()
            in {"1", "true", "yes", "y"}
        )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self.genai_client = genai.Client(api_key=api_key)
        self.qdrant = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            check_compatibility=False,
            timeout=self.qdrant_timeout_seconds,
        )
        self.db: RagDb = get_db()
        self._embed_model_checked = False

    async def startup(self) -> None:
        await self.db.connect()
        await self.db.init_schema()
        await self._resolve_embed_model()

    async def shutdown(self) -> None:
        await self.db.close()
        await self.qdrant.close()

    async def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        await self._resolve_embed_model()
        await self._log_ai_call(
            purpose="embed_content",
            model=self.embed_model,
            request=texts,
            response=None,
        )
        try:
            response = await self.genai_client.aio.models.embed_content(
                model=self.embed_model,
                contents=list(texts),
            )
        except Exception as exc:
            switched = await self._maybe_switch_embed_model(exc)
            if not switched:
                raise
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

    async def _resolve_embed_model(self, force: bool = False) -> str:
        if self._embed_model_checked and not force:
            return self.embed_model
        try:
            available = await self._list_embedding_models()
        except Exception as exc:
            logger.warning("Could not list embedding models; using configured model '%s'. Error: %s", self.embed_model, exc)
            return self.embed_model

        if not available:
            logger.warning("No embedding models reported by the Gemini API. Using configured model '%s'.", self.embed_model)
            self._embed_model_checked = True
            return self.embed_model

        configured = self._normalize_model_name(self.embed_model)
        if configured in available:
            if configured != self.embed_model:
                self.embed_model = configured
            self._embed_model_checked = True
            return self.embed_model

        preferred = "gemini-embedding-001"
        selected = preferred if preferred in available else available[0]
        logger.warning(
            "Configured embedding model '%s' is unavailable. Falling back to '%s'. Available models: %s",
            self.embed_model,
            selected,
            ", ".join(available),
        )
        self.embed_model = selected
        self._embed_model_checked = True
        return self.embed_model

    async def _maybe_switch_embed_model(self, exc: Exception) -> bool:
        message = str(exc).lower()
        if "embedcontent" not in message and "embed_content" not in message:
            return False
        if "not found" not in message and "not supported" not in message:
            return False

        old_model = self.embed_model
        await self._resolve_embed_model(force=True)
        if self.embed_model == old_model:
            return False

        logger.warning("Retrying embedding with fallback model '%s' after '%s' failed.", self.embed_model, old_model)
        return True

    async def _list_embedding_models(self) -> List[str]:
        def _fetch() -> List[str]:
            discovered: List[str] = []
            for model in self.genai_client.models.list():
                actions = [a.lower() for a in (model.supported_actions or [])]
                if "embedcontent" in actions:
                    discovered.append(self._normalize_model_name(model.name))
            # Preserve order while removing duplicates.
            return list(dict.fromkeys(discovered))

        return await asyncio.to_thread(_fetch)

    def _normalize_model_name(self, model_name: str) -> str:
        return model_name.replace("models/", "", 1).strip()

    async def _ensure_payload_indexes(self) -> None:
        for field_name in ("scope_id", "doc_id"):
            try:
                await self.qdrant.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" in message:
                    continue
                logger.warning("Failed to ensure payload index '%s': %s", field_name, exc)

    async def _ensure_collection(self, dim: int) -> None:
        try:
            info = await self.qdrant.get_collection(self.collection)
            vectors_config = info.config.params.vectors  # type: ignore[attr-defined]
            if hasattr(vectors_config, "size"):
                existing = int(vectors_config.size)
            elif isinstance(vectors_config, dict) and vectors_config:
                first_cfg = next(iter(vectors_config.values()))
                existing = int(getattr(first_cfg, "size"))
            else:
                raise RuntimeError("Unable to determine Qdrant vector dimension")

            if existing != dim:
                message = (
                    f"Qdrant collection dimension mismatch ({existing} != {dim}) for '{self.collection}'."
                )
                if self.recreate_collection_on_dim_mismatch:
                    logger.warning("%s Recreating collection due to mismatch.", message)
                    await self.qdrant.recreate_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                    await self._ensure_payload_indexes()
                else:
                    raise RuntimeError(
                        message
                        + " Set RAG_RECREATE_COLLECTION_ON_DIM_MISMATCH=true or use a new RAG_COLLECTION."
                    )
            else:
                await self._ensure_payload_indexes()
            return
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code not in (None, 404):
                raise
            if status_code is None and "not found" not in str(exc).lower():
                raise
            await self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            await self._ensure_payload_indexes()

    async def _delete_points_by_ids(self, point_ids: Sequence[str]) -> None:
        ids = [pid for pid in point_ids if pid]
        if not ids:
            return
        await self.qdrant.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=list(ids)),
            wait=True,
        )

    def _build_scope_filter(
        self,
        scope_id: Optional[str],
        include_global: bool,
    ) -> Optional[Filter]:
        normalized_scope = (scope_id or "").strip() or DEFAULT_SCOPE_ID
        should: List[FieldCondition] = [
            FieldCondition(key="scope_id", match=MatchValue(value=normalized_scope))
        ]
        if include_global and normalized_scope != DEFAULT_SCOPE_ID:
            should.append(
                FieldCondition(key="scope_id", match=MatchValue(value=DEFAULT_SCOPE_ID))
            )
        return Filter(should=should)

    async def ingest_text(
        self,
        text: str,
        *,
        source: Optional[str] = None,
        title: Optional[str] = None,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not text.strip():
            return {"chunks": 0, "document_id": None, "deduplicated": False}

        normalized_scope = (scope_id or "").strip() or DEFAULT_SCOPE_ID
        doc_hash = _hash_text(text)
        document_id, should_index, stale_qdrant_ids = await self.db.prepare_document_for_ingest(
            scope_id=normalized_scope,
            source=source,
            title=title,
            path=path,
            doc_hash=doc_hash,
            metadata=metadata or {},
        )

        if not should_index:
            return {
                "chunks": 0,
                "document_id": document_id,
                "deduplicated": True,
                "scope_id": normalized_scope,
            }

        if stale_qdrant_ids:
            try:
                await self._delete_points_by_ids(stale_qdrant_ids)
            except Exception as cleanup_exc:
                logger.warning("Failed to cleanup stale Qdrant points: %s", cleanup_exc)

        chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            await self.db.mark_document_ready(document_id)
            return {
                "chunks": 0,
                "document_id": document_id,
                "deduplicated": False,
                "scope_id": normalized_scope,
            }

        vectors = await self._embed_texts(chunks)
        await self._ensure_collection(len(vectors[0]))

        points: List[PointStruct] = []
        chunk_records: List[tuple[str, int, str, Dict[str, Any]]] = []
        point_ids: List[str] = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            qdrant_id = str(uuid.uuid4())
            point_ids.append(qdrant_id)
            payload = {
                "text": chunk,
                "doc_id": document_id,
                "title": title,
                "source": source,
                "path": path,
                "scope_id": normalized_scope,
                "chunk_index": i,
                "metadata": metadata or {},
            }
            points.append(PointStruct(id=qdrant_id, vector=vec, payload=payload))
            chunk_records.append((qdrant_id, i, chunk, payload))

        try:
            batch_size = max(1, self.qdrant_upsert_batch_size)
            for start in range(0, len(points), batch_size):
                await self.qdrant.upsert(
                    collection_name=self.collection,
                    points=points[start : start + batch_size],
                    wait=True,
                )
            await self.db.insert_chunks(document_id, normalized_scope, chunk_records)
            await self.db.mark_document_ready(document_id)
        except Exception as exc:
            try:
                await self._delete_points_by_ids(point_ids)
            except Exception as cleanup_exc:
                logger.warning("Failed to rollback Qdrant points: %s", cleanup_exc)
            await self.db.mark_document_failed(document_id, str(exc))
            raise

        return {
            "chunks": len(chunks),
            "document_id": document_id,
            "deduplicated": False,
            "scope_id": normalized_scope,
        }

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        scope_id: Optional[str] = None,
        include_global: bool = True,
    ) -> List[RetrievedChunk]:
        if not query.strip():
            return []
        vectors = await self._embed_texts([query])
        vector = vectors[0]
        await self._ensure_collection(len(vector))

        scope_filter = self._build_scope_filter(scope_id, include_global)
        response = await self.qdrant.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=scope_filter,
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

    async def chat(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        uploaded_text: Optional[str] = None,
        scope_id: Optional[str] = None,
        top_k: int = 5,
        use_web_search: bool = False,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        chunks = await self.retrieve(
            message,
            top_k=top_k,
            scope_id=scope_id,
            include_global=True,
        )

        content_parts: List[str] = [
            system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT,
            f"Current Time: {datetime.utcnow().isoformat()}Z",
        ]

        if uploaded_text and uploaded_text.strip():
            content_parts.append(f"User uploaded documents:\n{uploaded_text.strip()}")

        if chunks:
            context_blocks = []
            for i, c in enumerate(chunks, start=1):
                text = c.text.strip()
                if text:
                    context_blocks.append(f"[{i}] {text}")
            if context_blocks:
                content_parts.append("Retrieved context:\n" + "\n\n".join(context_blocks))

        content_parts.append(message)

        tools = []
        if use_web_search:
            tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))

        config = genai_types.GenerateContentConfig(
            tools=tools,
            temperature=temperature,
            response_modalities=["TEXT"],
        )

        response = await self.genai_client.aio.models.generate_content(
            model=self.chat_model,
            contents=content_parts,
            config=config,
        )
        answer = (response.text or "").strip()

        await self._log_ai_call(
            purpose="chat_response",
            model=self.chat_model,
            request=content_parts,
            response=answer,
        )

        return {
            "answer": answer,
            "results": [
                {"text": c.text, "score": c.score, "payload": c.payload} for c in chunks
            ],
            "scope_id": (scope_id or "").strip() or DEFAULT_SCOPE_ID,
        }

    async def ingest_folder(self, folder: str, recursive: bool = True) -> Dict[str, Any]:
        path = os.path.abspath(folder)
        files = scan_folder(Path(path), recursive=recursive)
        ingested = 0
        deduplicated = 0
        for file_path in files:
            text = extract_text_from_path(file_path)
            if not text:
                continue
            result = await self.ingest_text(
                text,
                source="folder",
                title=file_path.name,
                path=str(file_path),
                metadata={"ext": file_path.suffix.lower()},
                scope_id=DEFAULT_SCOPE_ID,
            )
            if result.get("deduplicated"):
                deduplicated += 1
            else:
                ingested += 1
        return {"files_ingested": ingested, "files_deduplicated": deduplicated}

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
