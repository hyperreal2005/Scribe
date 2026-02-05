import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import asyncpg

logger = logging.getLogger(__name__)


class RagDb:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def init_schema(self) -> None:
        if self._pool is None:
            raise RuntimeError("DB not connected")
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            await conn.execute("CREATE SCHEMA IF NOT EXISTS rag;")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag.documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source TEXT,
                    title TEXT,
                    path TEXT,
                    hash TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag.chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
                    qdrant_id TEXT,
                    chunk_index INT,
                    text TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx ON rag.chunks(document_id);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_qdrant_id_idx ON rag.chunks(qdrant_id);"
            )

    async def insert_document(
        self,
        source: Optional[str],
        title: Optional[str],
        path: Optional[str],
        doc_hash: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        if self._pool is None:
            raise RuntimeError("DB not connected")
        meta_json = json.dumps(metadata or {})
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO rag.documents (source, title, path, hash, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                RETURNING id::text;
                """,
                source,
                title,
                path,
                doc_hash,
                meta_json,
            )
        return row["id"]

    async def insert_chunks(
        self,
        document_id: str,
        chunks: Iterable[Tuple[str, int, str, Dict[str, Any]]],
    ) -> None:
        if self._pool is None:
            raise RuntimeError("DB not connected")
        records: List[Tuple[str, str, int, str, str]] = []
        for qdrant_id, chunk_index, text, metadata in chunks:
            records.append(
                (
                    document_id,
                    qdrant_id,
                    chunk_index,
                    text,
                    json.dumps(metadata),
                )
            )
        if not records:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO rag.chunks (document_id, qdrant_id, chunk_index, text, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb);
                """,
                records,
            )


def get_db() -> RagDb:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")
    return RagDb(dsn)
