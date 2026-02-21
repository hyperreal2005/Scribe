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
                    scope_id TEXT NOT NULL DEFAULT 'global',
                    source TEXT,
                    title TEXT,
                    path TEXT,
                    hash TEXT,
                    metadata JSONB,
                    status TEXT NOT NULL DEFAULT 'pending',
                    ingest_error TEXT,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag.chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
                    scope_id TEXT NOT NULL DEFAULT 'global',
                    qdrant_id TEXT,
                    chunk_index INT,
                    text TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
            await conn.execute(
                "ALTER TABLE rag.documents ADD COLUMN IF NOT EXISTS scope_id TEXT NOT NULL DEFAULT 'global';"
            )
            await conn.execute(
                "ALTER TABLE rag.documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending';"
            )
            await conn.execute(
                "ALTER TABLE rag.documents ADD COLUMN IF NOT EXISTS ingest_error TEXT;"
            )
            await conn.execute(
                "ALTER TABLE rag.documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();"
            )
            await conn.execute(
                "ALTER TABLE rag.chunks ADD COLUMN IF NOT EXISTS scope_id TEXT NOT NULL DEFAULT 'global';"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_documents_scope_hash_idx ON rag.documents(scope_id, hash);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_documents_status_idx ON rag.documents(status);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx ON rag.chunks(document_id);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_qdrant_id_idx ON rag.chunks(qdrant_id);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_scope_id_idx ON rag.chunks(scope_id);"
            )

    async def prepare_document_for_ingest(
        self,
        scope_id: str,
        source: Optional[str],
        title: Optional[str],
        path: Optional[str],
        doc_hash: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[str, bool, List[str]]:
        if self._pool is None:
            raise RuntimeError("DB not connected")

        meta_json = json.dumps(metadata or {})
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2));",
                    scope_id,
                    doc_hash,
                )
                rows = await conn.fetch(
                    """
                    SELECT id::text AS id, status
                    FROM rag.documents
                    WHERE scope_id = $1 AND hash = $2
                    ORDER BY created_at DESC;
                    """,
                    scope_id,
                    doc_hash,
                )

                for row in rows:
                    if row["status"] == "ready":
                        return row["id"], False, []

                if rows:
                    document_id = rows[0]["id"]
                    stale_rows = await conn.fetch(
                        """
                        SELECT qdrant_id
                        FROM rag.chunks
                        WHERE document_id = $1 AND qdrant_id IS NOT NULL;
                        """,
                        document_id,
                    )
                    stale_qdrant_ids = [r["qdrant_id"] for r in stale_rows]
                    await conn.execute(
                        "DELETE FROM rag.chunks WHERE document_id = $1;",
                        document_id,
                    )
                    await conn.execute(
                        """
                        UPDATE rag.documents
                        SET scope_id = $2,
                            source = $3,
                            title = $4,
                            path = $5,
                            metadata = $6::jsonb,
                            status = 'pending',
                            ingest_error = NULL,
                            updated_at = now()
                        WHERE id = $1;
                        """,
                        document_id,
                        scope_id,
                        source,
                        title,
                        path,
                        meta_json,
                    )
                    return document_id, True, stale_qdrant_ids

                row = await conn.fetchrow(
                    """
                    INSERT INTO rag.documents (
                        scope_id, source, title, path, hash, metadata, status, ingest_error
                    )
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, 'pending', NULL)
                    RETURNING id::text;
                    """,
                    scope_id,
                    source,
                    title,
                    path,
                    doc_hash,
                    meta_json,
                )
                return row["id"], True, []

    async def insert_chunks(
        self,
        document_id: str,
        scope_id: str,
        chunks: Iterable[Tuple[str, int, str, Dict[str, Any]]],
    ) -> None:
        if self._pool is None:
            raise RuntimeError("DB not connected")
        records: List[Tuple[str, str, str, int, str, str]] = []
        for qdrant_id, chunk_index, text, metadata in chunks:
            records.append(
                (
                    document_id,
                    scope_id,
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
                INSERT INTO rag.chunks (document_id, scope_id, qdrant_id, chunk_index, text, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb);
                """,
                records,
            )

    async def mark_document_ready(self, document_id: str) -> None:
        if self._pool is None:
            raise RuntimeError("DB not connected")
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE rag.documents
                SET status = 'ready',
                    ingest_error = NULL,
                    updated_at = now()
                WHERE id = $1::uuid;
                """,
                document_id,
            )

    async def mark_document_failed(self, document_id: str, error: str) -> None:
        if self._pool is None:
            raise RuntimeError("DB not connected")
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE rag.documents
                SET status = 'failed',
                    ingest_error = $2,
                    updated_at = now()
                WHERE id = $1::uuid;
                """,
                document_id,
                error[:2000],
            )


def get_db() -> RagDb:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")
    return RagDb(dsn)
