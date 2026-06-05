"""PostgreSQL + pgvector backend for vector search.

Implements the full ``BaseBackend`` interface using psycopg's async driver
and PostgreSQL's pgvector extension.

Does NOT implement ``ChangeStreamCapable`` or ``AutoEmbeddingCapable``.
Postgres has no change streams (use LISTEN/NOTIFY in future phases).
Postgres has no server-side embedding (always requires ``query_vector``).
"""

import asyncio
import json
import os
import uuid
from typing import Any, Optional

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.backends.postgres.config import (
    PgDistanceMetric,
    PgIndexType,
    PostgresBackendConfig,
)
from semantic_vector_router.backends.postgres.filters import (
    translate_filters,
    validate_field_name,
)
from semantic_vector_router.exceptions import ConnectionError, SearchError
from semantic_vector_router.models.backend import IndexStatus, PartitionStorageResult
from semantic_vector_router.models.partition import PartitionInfo
from semantic_vector_router.models.svr_config import SVRConfig
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class PostgresBackend(BaseBackend):
    """PostgreSQL + pgvector backend for vector search.

    Uses a single table with a ``partition_name`` column for logical
    partitioning. Vectors are stored in a pgvector ``vector`` column
    and indexed with HNSW or IVFFlat.
    """

    def __init__(self, config: SVRConfig):
        super().__init__(config)
        self._pool: Optional[AsyncConnectionPool] = None
        self._pg_config: PostgresBackendConfig = self._resolve_pg_config(config)
        self._table_name: str = f"{self._pg_config.table_prefix}vectors"
        self._schema: str = self._pg_config.schema_name
        self._fq_table: str = f"{self._schema}.{self._table_name}"
        self._dimensions: int = (
            self._pg_config.vector_dimensions or config.vector_search.dimensions
        )

    # ── Connection lifecycle ──────────────────────────────────────

    async def connect(self) -> None:
        """Create async connection pool and ensure schema + pgvector."""
        conninfo = os.environ.get(self._pg_config.connection_string_env, "")
        if not conninfo:
            raise ConnectionError(
                f"Environment variable '{self._pg_config.connection_string_env}' "
                f"is not set or empty"
            )
        self._pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=self._pg_config.pool_min_size,
            max_size=self._pg_config.pool_max_size,
            open=False,
        )
        await self._pool.open()
        await self._ensure_pgvector()
        await self._ensure_table()
        logger.info(
            f"Connected to PostgreSQL, schema={self._schema}, "
            f"table={self._table_name}, dimensions={self._dimensions}"
        )

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from PostgreSQL")

    async def is_connected(self) -> bool:
        """Check if pool is alive and can execute a query."""
        if self._pool is None:
            return False
        try:
            async with self._pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def health_check(self) -> bool:
        """Extended health check: verify pool + pgvector extension."""
        if not await self.is_connected():
            return False
        try:
            async with self._pool.connection() as conn:  # type: ignore[union-attr]
                await conn.execute("SELECT vector_dims('[1,0]'::vector)")
            return True
        except Exception:
            return False

    # ── Internal setup ────────────────────────────────────────────

    async def _ensure_pgvector(self) -> None:
        """Ensure pgvector extension is installed."""
        assert self._pool is not None
        async with self._pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    async def _ensure_table(self) -> None:
        """Create the vectors table if it does not exist."""
        assert self._pool is not None
        schema_id = sql.Identifier(self._schema)
        table_id = sql.Identifier(self._schema, self._table_name)
        partition_idx_id = sql.Identifier(
            f"{self._table_name}_partition_idx"
        )

        async with self._pool.connection() as conn:
            await conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema_id)
            )
            await conn.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} ("
                    "id TEXT PRIMARY KEY, "
                    "partition_name TEXT NOT NULL, "
                    "embedding vector({dimensions}), "
                    "content JSONB NOT NULL DEFAULT '{{}}'::jsonb, "
                    "created_at TIMESTAMPTZ DEFAULT NOW(), "
                    "updated_at TIMESTAMPTZ DEFAULT NOW()"
                    ")"
                ).format(table_id, dimensions=sql.Literal(self._dimensions))
            )
            # B-tree index on partition_name for filter efficiency
            await conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} (partition_name)"
                ).format(partition_idx_id, table_id)
            )

    # ── Partition storage ─────────────────────────────────────────

    async def create_partition_storage(
        self,
        partition: PartitionInfo,
        config: SVRConfig,
    ) -> PartitionStorageResult:
        """No physical storage created — partitions are WHERE clauses."""
        return PartitionStorageResult(
            storage_name=self._table_name,
            storage_type="table",
            metadata={
                "schema": self._schema,
                "partition_column": "partition_name",
                "partition_value": partition.filter_value or partition.name,
            },
        )

    async def delete_partition_storage(self, partition: PartitionInfo) -> None:
        """Delete all rows for this partition."""
        assert self._pool is not None
        pval = partition.filter_value or partition.name
        table_id = sql.Identifier(self._schema, self._table_name)
        async with self._pool.connection() as conn:
            await conn.execute(
                sql.SQL("DELETE FROM {} WHERE partition_name = %s").format(
                    table_id
                ),
                (pval,),
            )

    async def partition_storage_exists(self, partition: PartitionInfo) -> bool:
        """Check if the vectors table exists (always true after connect)."""
        assert self._pool is not None
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s)",
                (self._schema, self._table_name),
            )
            row = await cur.fetchone()
            return bool(row and row[0])

    # ── Index lifecycle ───────────────────────────────────────────

    async def create_partition_index(
        self,
        partition: PartitionInfo,
        config: SVRConfig,
    ) -> str:
        """Create HNSW or IVFFlat index on the embedding column.

        This is a table-level index, not per-partition. The index is
        created once and shared across all partitions (with
        ``partition_name`` in WHERE). If the index already exists,
        this is a no-op.
        """
        idx_name = f"{self._table_name}_embedding_idx"
        assert self._pool is not None

        # Check if index already exists
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
                (self._schema, idx_name),
            )
            if await cur.fetchone():
                return idx_name

        # Build the index using safe identifier interpolation
        idx_id = sql.Identifier(idx_name)
        table_id = sql.Identifier(self._schema, self._table_name)
        ops_class = sql.SQL(self._get_ops_class())

        if self._pg_config.index_type == PgIndexType.HNSW:
            hnsw = self._pg_config.hnsw
            async with self._pool.connection() as conn:
                await conn.execute(
                    sql.SQL(
                        "CREATE INDEX {} ON {} "
                        "USING hnsw (embedding {}) "
                        "WITH (m = {m}, ef_construction = {ef_construction})"
                    ).format(
                        idx_id,
                        table_id,
                        ops_class,
                        m=sql.Literal(hnsw.m),
                        ef_construction=sql.Literal(hnsw.ef_construction),
                    )
                )
        else:  # IVFFlat
            ivf = self._pg_config.ivfflat
            async with self._pool.connection() as conn:
                await conn.execute(
                    sql.SQL(
                        "CREATE INDEX {} ON {} "
                        "USING ivfflat (embedding {}) "
                        "WITH (lists = {lists})"
                    ).format(
                        idx_id,
                        table_id,
                        ops_class,
                        lists=sql.Literal(ivf.lists),
                    )
                )

        logger.info(
            f"Created {self._pg_config.index_type.value} index '{idx_name}' "
            f"on {self._fq_table}"
        )
        return idx_name

    async def delete_partition_index(self, partition: PartitionInfo) -> None:
        """No-op for Postgres. The shared index is not deleted per-partition."""
        pass

    async def get_partition_index_status(
        self,
        partition: PartitionInfo,
    ) -> IndexStatus:
        """Postgres indexes are synchronous — always READY or NOT_FOUND."""
        idx_name = f"{self._table_name}_embedding_idx"
        assert self._pool is not None
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
                (self._schema, idx_name),
            )
            if await cur.fetchone():
                return IndexStatus.READY
            return IndexStatus.NOT_FOUND

    async def wait_for_index_ready(
        self,
        partition: PartitionInfo,
        timeout_s: float = 300.0,
        poll_interval_s: float = 5.0,
    ) -> bool:
        """Override: Postgres index builds are synchronous. Return immediately."""
        status = await self.get_partition_index_status(partition)
        return status == IndexStatus.READY

    # ── Search ────────────────────────────────────────────────────

    async def execute_search(
        self,
        partition: PartitionInfo,
        query_vector: list[float],
        limit: int,
        num_candidates: int,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
        pre_native: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Execute vector similarity search using pgvector distance operators.

        ``query_vector`` is required (Postgres has no auto-embedding).

        Args:
            partition: Partition to search.
            query_vector: Query embedding vector.
            limit: Maximum results.
            num_candidates: ANN candidates (not used per-query in pgvector).
            filters: SVR portable filters translated to SQL WHERE.
            exact: If True, disable index scan (brute-force sequential).
                Uses ``SET LOCAL enable_indexscan = off`` scoped to a
                savepoint to prevent setting bleed.
            post_native: Raw SQL string appended as CTE consumer. The core
                query is wrapped in ``WITH svr_results AS (...)`` and
                post_native becomes the outer SELECT. Trusted application
                code only. **Runs per-partition before merge.**
            pre_native: Raw SQL string with additional WHERE conditions,
                AND-joined with translated filters. Parenthesized to
                prevent operator precedence issues. Trusted application
                code only.
        """
        assert self._pool is not None

        # Build WHERE clause
        where_parts = ["partition_name = %s"]
        params: list[Any] = [partition.filter_value or partition.name]

        # Translate user-provided filters to SQL
        if filters:
            sql_fragment, filter_params = translate_filters(filters)
            if sql_fragment:
                where_parts.append(sql_fragment)
                params.extend(filter_params)

        # Pre-native: raw Postgres WHERE conditions (trusted application code)
        if pre_native:
            where_parts.append(f"({pre_native})")

        where_clause = " AND ".join(where_parts)

        # Distance operator and score expression based on metric
        distance_op = self._get_distance_operator()
        score_expr = self._get_score_expression()
        vector_literal = "[" + ",".join(str(v) for v in query_vector) + "]"

        table_id = sql.Identifier(self._schema, self._table_name)

        # Core vector search query
        core_query = sql.SQL(
            "SELECT id, content, partition_name, "
            "{score_expr} AS score "
            "FROM {table} "
            "WHERE {where} "
            "ORDER BY embedding {dist_op} %s::vector "
            "LIMIT %s"
        ).format(
            score_expr=sql.SQL(score_expr),
            table=table_id,
            where=sql.SQL(where_clause),
            dist_op=sql.SQL(distance_op),
        )

        # If post_native: wrap in CTE so user SQL can reference svr_results
        if post_native:
            full_query = sql.SQL(
                "WITH svr_results AS ({core}) {post}"
            ).format(
                core=core_query,
                post=sql.SQL(post_native),
            )
        else:
            full_query = core_query

        async with self._pool.connection() as conn:
            if exact:
                await conn.execute("SET LOCAL enable_indexscan = off")
            else:
                search_config_sql = self._get_search_config_sql()
                if search_config_sql:
                    await conn.execute(search_config_sql)

            cur = await conn.execute(
                full_query,
                (vector_literal, *params, vector_literal, limit),
            )
            rows = await cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            doc = row[1] if isinstance(row[1], dict) else {}
            doc["_id"] = row[0]
            results.append(
                {
                    **doc,
                    "_svr_score": float(row[3]),
                    "_svr_partition": partition.name,
                }
            )

        return results

    # ── Data operations ───────────────────────────────────────────

    async def get_distinct_values(
        self,
        field: str,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """Get distinct values for a field.

        For ``partition_name``, queries the column directly.
        For other fields, queries inside the JSONB ``content`` column.
        """
        assert self._pool is not None

        if field == "partition_name":
            col_ref = "partition_name"
        else:
            validate_field_name(field)
            col_ref = f"content->>'{field}'"

        table_id = sql.Identifier(self._schema, self._table_name)

        where = ""
        params: list[Any] = []
        if filter_expression:
            sql_frag, filt_params = translate_filters(filter_expression)
            if sql_frag:
                where = f"WHERE {sql_frag}"
                params = filt_params

        query = sql.SQL("SELECT DISTINCT {col} FROM {table} {where}").format(
            col=sql.SQL(col_ref),
            table=table_id,
            where=sql.SQL(where),
        )

        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()
        return [row[0] for row in rows if row[0] is not None]

    async def count_documents(
        self,
        collection_name: Optional[str] = None,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> int:
        """Count documents, optionally filtered."""
        assert self._pool is not None

        table_id = sql.Identifier(self._schema, self._table_name)

        where = ""
        params: list[Any] = []
        if filter_expression:
            sql_frag, filt_params = translate_filters(filter_expression)
            if sql_frag:
                where = f"WHERE {sql_frag}"
                params = filt_params

        query = sql.SQL("SELECT COUNT(*) FROM {table} {where}").format(
            table=table_id,
            where=sql.SQL(where),
        )

        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            row = await cur.fetchone()
        return row[0] if row else 0

    async def get_collection_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get table statistics."""
        assert self._pool is not None
        table_id = sql.Identifier(self._schema, self._table_name)

        query = sql.SQL(
            "SELECT COUNT(*), pg_total_relation_size({fq_table_literal}) "
            "FROM {table}"
        ).format(
            fq_table_literal=sql.Literal(self._fq_table),
            table=table_id,
        )

        async with self._pool.connection() as conn:
            cur = await conn.execute(query)
            row = await cur.fetchone()
            count = row[0] if row else 0
            size = row[1] if row else 0

        return {
            "name": self._fq_table,
            "count": count,
            "size": size,
            "backend": "postgres",
        }

    async def get_partition_document_counts(
        self,
        field: str,
    ) -> dict[str, int]:
        """Get document counts grouped by the specified field.

        If ``field`` is ``"partition_name"``, groups by the top-level
        ``partition_name`` column. Otherwise, groups by the JSONB
        ``content->>'{field}'`` expression.
        """
        assert self._pool is not None
        table_id = sql.Identifier(self._schema, self._table_name)

        if field == "partition_name":
            col_ref = "partition_name"
        else:
            validate_field_name(field)
            col_ref = f"content->>'{field}'"

        query = sql.SQL(
            "SELECT {col}, COUNT(*) FROM {table} "
            "GROUP BY {col} ORDER BY COUNT(*) DESC"
        ).format(
            col=sql.SQL(col_ref),
            table=table_id,
        )

        async with self._pool.connection() as conn:
            cur = await conn.execute(query)
            rows = await cur.fetchall()
        return {row[0]: row[1] for row in rows if row[0] is not None}

    # ── search_partitions (required by client.py) ──────────────────

    async def search_partitions(
        self,
        partitions: list[PartitionInfo],
        limit: int,
        query_vector: Optional[list[float]] = None,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        exact: bool = False,
        post_native: Optional[Any] = None,
        pre_native: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Fan-out search across partitions.

        Postgres has no auto-embedding, so ``query`` is ignored.
        ``query_vector`` is required.
        ``post_native`` and ``pre_native`` run per-partition before merging.
        """
        if query_vector is None:
            raise SearchError(
                "PostgresBackend requires query_vector "
                "(no auto-embedding support)",
                details={},
            )

        if not partitions:
            return []

        num_candidates = limit * self.config.vector_search.num_candidates_multiplier
        tasks = [
            self.execute_search(
                p, query_vector, limit, num_candidates, filters,
                exact=exact, post_native=post_native, pre_native=pre_native,
            )
            for p in partitions
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined: list[dict[str, Any]] = []
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.error(
                    f"Search failed on partition {partitions[i].name}: {r}"
                )
                continue
            combined.extend(r)
        return combined

    # ── Document writes ─────────────────────────────────────────

    async def insert_documents(
        self,
        documents: list[dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> int:
        """Insert or upsert documents into the vectors table.

        Each document dict may contain:
        - ``_id``: used as the primary key. If missing, a UUID is generated.
        - ``partition_name``: used directly. Falls back to the config's
          partitioning field value in the document, or ``"default"``.
        - ``embedding``: the vector. Stored in the ``embedding`` column.
        - All remaining keys are stored in the ``content`` JSONB column.

        Uses ``INSERT ... ON CONFLICT (id) DO UPDATE`` for upsert semantics.

        Args:
            documents: List of document dicts to insert.
            collection_name: Ignored (Postgres uses a single table).

        Returns:
            Number of documents successfully inserted/updated.
        """
        if not documents:
            return 0

        assert self._pool is not None
        table_id = sql.Identifier(self._schema, self._table_name)

        partition_field = self.config.partitioning.field

        rows: list[tuple[str, str, Optional[str], str]] = []
        for doc in documents:
            doc = dict(doc)  # shallow copy to avoid mutating caller's data

            # Extract id
            doc_id = str(doc.pop("_id", None) or uuid.uuid4())

            # Extract partition_name
            partition_name = doc.pop("partition_name", None)
            if partition_name is None:
                partition_name = doc.pop(partition_field, None) if partition_field else None
            else:
                # Also remove partitioning field from content if present
                doc.pop(partition_field, None) if partition_field else None
            partition_name = str(partition_name) if partition_name is not None else "default"

            # Extract embedding
            embedding = doc.pop("embedding", None)
            vector_literal: Optional[str] = None
            if embedding is not None:
                vector_literal = "[" + ",".join(str(v) for v in embedding) + "]"

            # Build structured embedding text from source_fields if configured
            source_fields = self.config.embedding.source_fields
            if source_fields:
                embedding_text_obj = {
                    field: doc.get(field)
                    for field in source_fields
                    if doc.get(field) is not None
                }
                if embedding_text_obj:
                    doc["_svr_embedding_text"] = embedding_text_obj

            # Remaining fields go into content JSONB
            content_json = json.dumps(doc)

            rows.append((doc_id, partition_name, vector_literal, content_json))

        upsert_query = sql.SQL(
            "INSERT INTO {table} (id, partition_name, embedding, content, updated_at) "
            "VALUES (%s, %s, %s::vector, %s::jsonb, NOW()) "
            "ON CONFLICT (id) DO UPDATE SET "
            "partition_name = EXCLUDED.partition_name, "
            "embedding = EXCLUDED.embedding, "
            "content = EXCLUDED.content, "
            "updated_at = NOW()"
        ).format(table=table_id)

        # Performance note: Per-row INSERT is correct but not optimized for bulk.
        # COPY-based bulk ingestion planned for a future phase.
        count = 0
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                for row in rows:
                    await cur.execute(upsert_query, row)
                    count += cur.rowcount

        logger.info(f"Upserted {count} documents into {self._fq_table}")
        return count

    # ── Filter translation ────────────────────────────────────────

    def translate_filters(
        self, svr_filters: Optional[dict[str, Any]]
    ) -> Any:
        """Translate SVR filters to SQL WHERE clause + params."""
        if svr_filters is None:
            return None
        return translate_filters(svr_filters)

    # ── Helper methods ────────────────────────────────────────────

    def _get_ops_class(self) -> str:
        """Get pgvector operator class for index creation."""
        mapping = {
            PgDistanceMetric.COSINE: "vector_cosine_ops",
            PgDistanceMetric.L2: "vector_l2_ops",
            PgDistanceMetric.INNER_PRODUCT: "vector_ip_ops",
        }
        return mapping[self._pg_config.distance_metric]

    def _get_distance_operator(self) -> str:
        """Get pgvector distance operator for queries."""
        mapping = {
            PgDistanceMetric.COSINE: "<=>",
            PgDistanceMetric.L2: "<->",
            PgDistanceMetric.INNER_PRODUCT: "<#>",
        }
        return mapping[self._pg_config.distance_metric]

    def _get_score_expression(self) -> str:
        """Get the SQL score expression for the configured distance metric.

        Returns a SQL fragment that computes a similarity score from the
        pgvector distance operator. The ``%s::vector`` placeholder must
        be filled with the query vector literal.

        - Cosine (``<=>``): ``1 - (distance)`` — range [0, 1] for
          normalized vectors.
        - L2 (``<->``): ``1.0 / (1.0 + distance)`` — always in [0, 1],
          1 = identical.
        - Inner Product (``<#>``): ``-(negated dot product)`` — pgvector
          stores the negated inner product, so we negate it back.
        """
        mapping = {
            PgDistanceMetric.COSINE: "1 - (embedding <=> %s::vector)",
            PgDistanceMetric.L2: "1.0 / (1.0 + (embedding <-> %s::vector))",
            PgDistanceMetric.INNER_PRODUCT: "-(embedding <#> %s::vector)",
        }
        return mapping[self._pg_config.distance_metric]

    def _get_search_config_sql(self) -> Optional[sql.Composed]:
        """Get SET statements for index-specific search parameters."""
        if self._pg_config.index_type == PgIndexType.HNSW:
            return sql.SQL("SET hnsw.ef_search = {}").format(
                sql.Literal(self._pg_config.hnsw.ef_search)
            )
        elif self._pg_config.index_type == PgIndexType.IVFFLAT:
            return sql.SQL("SET ivfflat.probes = {}").format(
                sql.Literal(self._pg_config.ivfflat.probes)
            )
        return None

    def _resolve_pg_config(self, config: SVRConfig) -> PostgresBackendConfig:
        """Extract PostgresBackendConfig from SVRConfig."""
        if hasattr(config, "postgres") and config.postgres is not None:
            return config.postgres
        return PostgresBackendConfig()
