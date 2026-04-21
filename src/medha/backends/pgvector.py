"""PgVectorBackend — PostgreSQL + pgvector storage backend."""

import logging
import re
import uuid
from typing import Any

from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

try:
    import asyncpg
    import pgvector.asyncpg
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False


class PgVectorBackend(VectorStorageBackend):
    """PostgreSQL + pgvector backend. Requires asyncpg and pgvector packages."""

    def __init__(self, settings: Any = None) -> None:
        if not HAS_PGVECTOR:
            raise ConfigurationError(
                "pgvector backend requires 'asyncpg' and 'pgvector'. "
                "Install with: pip install medha-archai[pgvector]"
            )
        # Import here to avoid NameError when HAS_PGVECTOR=False
        from medha.config import Settings
        self._settings = settings or Settings()
        self._pool: asyncpg.Pool | None = None
        self._initialized_tables: set[str] = set()

    def _table_name(self, collection_name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", collection_name)
        return f"{self._settings.pg_table_prefix}_{safe}"

    async def connect(self) -> None:
        try:
            kwargs = dict(
                min_size=self._settings.pg_pool_min_size,
                max_size=self._settings.pg_pool_max_size,
                init=pgvector.asyncpg.register_vector,
            )
            if self._settings.pg_dsn:
                self._pool = await asyncpg.create_pool(dsn=self._settings.pg_dsn, **kwargs)
            else:
                self._pool = await asyncpg.create_pool(
                    host=self._settings.pg_host,
                    port=self._settings.pg_port,
                    database=self._settings.pg_database,
                    user=self._settings.pg_user,
                    password=self._settings.pg_password.get_secret_value(),
                    **kwargs,
                )
        except Exception as e:
            raise StorageInitializationError(f"Failed to connect to PostgreSQL: {e}") from e

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        if collection_name in self._initialized_tables:
            return

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        try:
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        id                   UUID        PRIMARY KEY,
                        vector               vector({dimension}) NOT NULL,
                        original_question    TEXT NOT NULL DEFAULT '',
                        normalized_question  TEXT NOT NULL DEFAULT '',
                        generated_query      TEXT NOT NULL DEFAULT '',
                        query_hash           TEXT NOT NULL DEFAULT '',
                        response_summary     TEXT,
                        template_id          TEXT,
                        usage_count          INTEGER NOT NULL DEFAULT 1,
                        created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_vector_hnsw_idx
                        ON {schema}.{table}
                        USING hnsw (vector vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_query_hash_idx
                        ON {schema}.{table} (query_hash)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_template_id_idx
                        ON {schema}.{table} (template_id)
                        WHERE template_id IS NOT NULL
                """)

                await conn.execute(f"""
                    ALTER TABLE {schema}.{table}
                        ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_expires_at_idx
                        ON {schema}.{table} (expires_at)
                        WHERE expires_at IS NOT NULL
                """)

        except asyncpg.PostgresError as e:
            raise StorageInitializationError(
                f"Failed to initialize collection '{collection_name}': {e}"
            ) from e

        self._initialized_tables.add(collection_name)

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        sql = f"""
            SELECT
                id::text,
                original_question,
                normalized_question,
                generated_query,
                query_hash,
                response_summary,
                template_id,
                usage_count,
                created_at,
                expires_at,
                (1 - (vector <=> $1::vector))::float AS score
            FROM {schema}.{table}
            WHERE (1 - (vector <=> $1::vector)) >= $2
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY vector <=> $1::vector
            LIMIT $3
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, vector, score_threshold, limit)
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

        return [_row_to_cache_result(row) for row in rows]

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        if not entries:
            return

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        sql = f"""
            INSERT INTO {schema}.{table} (
                id, vector, original_question, normalized_question,
                generated_query, query_hash, response_summary,
                template_id, usage_count, created_at, expires_at
            )
            VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id) DO UPDATE SET
                vector              = EXCLUDED.vector,
                original_question   = EXCLUDED.original_question,
                normalized_question = EXCLUDED.normalized_question,
                generated_query     = EXCLUDED.generated_query,
                query_hash          = EXCLUDED.query_hash,
                response_summary    = EXCLUDED.response_summary,
                template_id         = EXCLUDED.template_id,
                usage_count         = EXCLUDED.usage_count,
                created_at          = EXCLUDED.created_at,
                expires_at          = EXCLUDED.expires_at
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(sql, [
                    (
                        uuid.UUID(entry.id),
                        entry.vector,
                        entry.original_question,
                        entry.normalized_question,
                        entry.generated_query,
                        entry.query_hash,
                        entry.response_summary,
                        entry.template_id,
                        entry.usage_count,
                        entry.created_at,
                        entry.expires_at,
                    )
                    for entry in entries
                ])
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        int_offset = int(offset) if offset is not None else 0

        vector_col = ", vector" if with_vectors else ""
        sql = f"""
            SELECT id::text, original_question, normalized_question, generated_query,
                   query_hash, response_summary, template_id, usage_count, created_at{vector_col}
            FROM {schema}.{table}
            ORDER BY created_at ASC, id ASC
            LIMIT $1 OFFSET $2
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, limit, int_offset)
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

        results = [_row_to_cache_result(row, score=1.0) for row in rows]
        next_offset = str(int_offset + limit) if len(rows) == limit else None
        return results, next_offset

    async def count(self, collection_name: str) -> int:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(f"SELECT COUNT(*) FROM {schema}.{table}")
                return int(row[0])
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        if not ids:
            return

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        uuid_ids = [uuid.UUID(id_) for id_ in ids]

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {schema}.{table} WHERE id = ANY($1::uuid[])",
                    uuid_ids,
                )
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

    async def find_expired(self, collection_name: str) -> list[str]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        sql = f"""
            SELECT id::text FROM {schema}.{table}
            WHERE expires_at IS NOT NULL AND expires_at < NOW()
            LIMIT 10000
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql)
            return [row["id"] for row in rows]
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector find_expired failed on '{collection_name}': {e}") from e

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized_tables.clear()

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        sql = f"""
            SELECT id::text, original_question, normalized_question, generated_query,
                   query_hash, response_summary, template_id, usage_count, created_at, expires_at
            FROM {schema}.{table}
            WHERE normalized_question = $1
            LIMIT 1
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, normalized_question)
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e
        if row is None:
            return None
        return _row_to_cache_result(row, score=1.0)

    async def find_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> list[str]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        sql = f"SELECT id::text FROM {schema}.{table} WHERE query_hash = $1"
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, query_hash)
            return [row["id"] for row in rows]
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

    async def find_by_template_id(
        self, collection_name: str, template_id: str
    ) -> list[str]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        sql = f"SELECT id::text FROM {schema}.{table} WHERE template_id = $1"
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, template_id)
            return [row["id"] for row in rows]
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

    async def drop_collection(self, collection_name: str) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
            self._initialized_tables.discard(collection_name)
            logger.info("Dropped table '%s.%s'", schema, table)
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector drop_collection failed on '{collection_name}': {e}") from e

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        sql = f"""
            SELECT id::text, original_question, normalized_question, generated_query,
                   query_hash, response_summary, template_id, usage_count, created_at
            FROM {schema}.{table}
            WHERE query_hash = $1
            LIMIT 1
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, query_hash)
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

        if row is None:
            return None
        return _row_to_cache_result(row, score=1.0)

    async def update_usage_count(self, collection_name: str, point_id: str) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    f"UPDATE {schema}.{table} SET usage_count = usage_count + 1 WHERE id = $1::uuid",
                    uuid.UUID(point_id),
                )
        except asyncpg.PostgresError as e:
            raise StorageError(f"PgVector operation failed on '{collection_name}': {e}") from e

        # asyncpg returns "UPDATE N" as string
        updated = int(result.split()[-1]) if result else 0
        if updated == 0:
            logger.warning(
                "update_usage_count: id '%s' not found in collection '%s'",
                point_id,
                collection_name,
            )


def _row_to_cache_result(row: Any, score: float | None = None) -> CacheResult:
    row_score = score if score is not None else max(0.0, min(1.0, float(row["score"])))
    return CacheResult(
        id=row["id"],
        score=max(0.0, min(1.0, row_score)),
        original_question=row["original_question"],
        normalized_question=row["normalized_question"],
        generated_query=row["generated_query"],
        query_hash=row["query_hash"],
        response_summary=row.get("response_summary"),
        template_id=row.get("template_id"),
        usage_count=row.get("usage_count", 0),
        created_at=row.get("created_at"),
        expires_at=row.get("expires_at"),
    )
