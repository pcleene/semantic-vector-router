"""Integration tests for PostgresBackend against local PostgreSQL + pgvector.

Validates the full backend lifecycle:
- Connection management (connect/disconnect/health)
- Partition storage (create/exists/delete)
- Index lifecycle (create HNSW, status, wait_for_ready)
- Vector search (execute_search, search_partitions)
- Data operations (count, stats, distinct values, partition counts)
- Filter translation (JSONB content filters in search)

Requires:
- PostgreSQL 17 running locally
- pgvector 0.8.1 extension installed
- Database ``svr_test`` at ``postgresql://<user>:<password>@<host>:5432/<db>
"""

import os

import pytest

from semantic_vector_router.backends.factory import create_backend
from semantic_vector_router.backends.postgres.backend import PostgresBackend
from semantic_vector_router.backends.postgres.config import PostgresBackendConfig
from semantic_vector_router.models.backend import IndexStatus, PartitionStorageResult
from semantic_vector_router.models.partition import PartitionInfo
from semantic_vector_router.models.svr_config import SVRConfig

pytestmark = pytest.mark.asyncio(loop_scope="module")

PG_DSN = "postgresql://<user>:<password>@<host>:5432/<db>"


def _make_config() -> SVRConfig:
    """Create a minimal SVRConfig for Postgres backend with 3D vectors."""
    return SVRConfig(
        database={
            "backend": "postgres",
            "database": "svr_test",
            "source_collection": "docs",
        },
        partitioning={"field": "category"},
        postgres=PostgresBackendConfig(connection_string_env="POSTGRES_URI"),
        vector_search={"dimensions": 3},
    )


@pytest.fixture(scope="module")
async def backend():
    """Create, connect, and yield a PostgresBackend; clean up after tests.

    - Sets the POSTGRES_URI env var
    - Creates a fresh SVRConfig with backend="postgres" and 3D vectors
    - Creates + connects the backend via create_backend()
    - After all tests: drops the test table and disconnects
    """
    os.environ["POSTGRES_URI"] = PG_DSN

    config = _make_config()
    be = create_backend(config)
    assert isinstance(be, PostgresBackend)

    try:
        await be.connect()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")

    yield be

    # Cleanup: drop the vectors table so tests are idempotent
    try:
        if be._pool is not None:
            async with be._pool.connection() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {be._fq_table} CASCADE")
    except Exception:
        pass

    await be.disconnect()


@pytest.fixture(scope="module")
async def seeded_backend(backend: PostgresBackend):
    """Insert test vectors into the backend table for search tests.

    Inserts 5 documents across 2 partitions (sports, music) with
    3-dimensional vectors for predictable cosine-distance results.
    """
    assert backend._pool is not None

    # Clear any leftover data
    async with backend._pool.connection() as conn:
        await conn.execute(f"DELETE FROM {backend._fq_table}")

    # Insert test data
    test_docs = [
        ("doc1", "sports", "[1,0,0]", '{"category": "sports", "title": "Football"}'),
        ("doc2", "sports", "[0.9,0.1,0]", '{"category": "sports", "title": "Basketball"}'),
        ("doc3", "music", "[0,1,0]", '{"category": "music", "title": "Jazz"}'),
        ("doc4", "music", "[0,0.9,0.1]", '{"category": "music", "title": "Rock"}'),
        ("doc5", "sports", "[0,0,1]", '{"category": "sports", "title": "Swimming"}'),
    ]

    async with backend._pool.connection() as conn:
        for doc_id, partition, embedding, content in test_docs:
            await conn.execute(
                f"INSERT INTO {backend._fq_table} "
                f"(id, partition_name, embedding, content) "
                f"VALUES (%s, %s, %s::vector, %s::jsonb)",
                (doc_id, partition, embedding, content),
            )

    yield backend

    # Cleanup inserted data after search tests
    async with backend._pool.connection() as conn:
        await conn.execute(f"DELETE FROM {backend._fq_table}")


# ── Connection lifecycle ──────────────────────────────────────────


async def test_backend_is_postgres_instance(backend: PostgresBackend):
    """create_backend() with backend='postgres' returns PostgresBackend."""
    assert isinstance(backend, PostgresBackend)


async def test_is_connected_after_connect(backend: PostgresBackend):
    """is_connected() returns True after successful connect()."""
    assert await backend.is_connected() is True


async def test_health_check_passes(backend: PostgresBackend):
    """health_check() returns True when pgvector is installed."""
    assert await backend.health_check() is True


# ── Partition storage ─────────────────────────────────────────────


async def test_create_partition_storage(backend: PostgresBackend):
    """create_partition_storage() returns PartitionStorageResult with storage_type='table'."""
    partition = PartitionInfo(name="sports", filter_value="sports")
    config = _make_config()

    result = await backend.create_partition_storage(partition, config)

    assert isinstance(result, PartitionStorageResult)
    assert result.storage_type == "table"
    assert result.storage_name == "svr_vectors"
    assert result.metadata["schema"] == "public"
    assert result.metadata["partition_column"] == "partition_name"
    assert result.metadata["partition_value"] == "sports"


async def test_partition_storage_exists(backend: PostgresBackend):
    """partition_storage_exists() returns True after connect (table exists)."""
    partition = PartitionInfo(name="sports", filter_value="sports")
    assert await backend.partition_storage_exists(partition) is True


# ── Index lifecycle ───────────────────────────────────────────────


async def test_create_partition_index_hnsw(backend: PostgresBackend):
    """create_partition_index() creates an HNSW index and returns the index name."""
    partition = PartitionInfo(name="sports", filter_value="sports")
    config = _make_config()

    # Drop any existing index first for idempotency
    assert backend._pool is not None
    async with backend._pool.connection() as conn:
        await conn.execute("DROP INDEX IF EXISTS public.svr_vectors_embedding_idx")

    idx_name = await backend.create_partition_index(partition, config)

    assert idx_name == "svr_vectors_embedding_idx"

    # Verify the index exists in pg_indexes
    async with backend._pool.connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = %s",
            (idx_name,),
        )
        row = await cur.fetchone()
        assert row is not None


async def test_create_partition_index_idempotent(backend: PostgresBackend):
    """Calling create_partition_index() twice is a no-op (returns same name)."""
    partition = PartitionInfo(name="sports", filter_value="sports")
    config = _make_config()

    idx_name = await backend.create_partition_index(partition, config)
    assert idx_name == "svr_vectors_embedding_idx"


async def test_get_partition_index_status_ready(backend: PostgresBackend):
    """get_partition_index_status() returns READY after index creation."""
    partition = PartitionInfo(name="sports", filter_value="sports")
    status = await backend.get_partition_index_status(partition)
    assert status == IndexStatus.READY


async def test_wait_for_index_ready(backend: PostgresBackend):
    """wait_for_index_ready() returns True immediately for Postgres."""
    partition = PartitionInfo(name="sports", filter_value="sports")
    result = await backend.wait_for_index_ready(partition)
    assert result is True


async def test_get_partition_index_status_not_found(backend: PostgresBackend):
    """get_partition_index_status() returns NOT_FOUND when index does not exist."""
    partition = PartitionInfo(name="sports", filter_value="sports")

    # Drop the index
    assert backend._pool is not None
    async with backend._pool.connection() as conn:
        await conn.execute("DROP INDEX IF EXISTS public.svr_vectors_embedding_idx")

    status = await backend.get_partition_index_status(partition)
    assert status == IndexStatus.NOT_FOUND

    # Re-create for subsequent tests
    config = _make_config()
    await backend.create_partition_index(partition, config)


# ── Search ────────────────────────────────────────────────────────


async def test_execute_search_basic(seeded_backend: PostgresBackend):
    """execute_search() returns results ordered by cosine similarity."""
    partition = PartitionInfo(name="sports", filter_value="sports")

    results = await seeded_backend.execute_search(
        partition=partition,
        query_vector=[1.0, 0.0, 0.0],
        limit=3,
        num_candidates=30,
    )

    assert len(results) == 3
    # doc1 ([1,0,0]) should be the top result (exact match, score ~1.0)
    assert results[0]["_id"] == "doc1"
    assert results[0]["_svr_score"] == pytest.approx(1.0, abs=0.001)
    assert results[0]["_svr_partition"] == "sports"
    # doc2 ([0.9,0.1,0]) should be second
    assert results[1]["_id"] == "doc2"
    assert results[1]["_svr_score"] > 0.9


async def test_execute_search_with_limit(seeded_backend: PostgresBackend):
    """execute_search() respects the limit parameter."""
    partition = PartitionInfo(name="sports", filter_value="sports")

    results = await seeded_backend.execute_search(
        partition=partition,
        query_vector=[1.0, 0.0, 0.0],
        limit=1,
        num_candidates=10,
    )

    assert len(results) == 1
    assert results[0]["_id"] == "doc1"


async def test_execute_search_different_partition(seeded_backend: PostgresBackend):
    """execute_search() filters by partition_name correctly."""
    partition = PartitionInfo(name="music", filter_value="music")

    results = await seeded_backend.execute_search(
        partition=partition,
        query_vector=[0.0, 1.0, 0.0],
        limit=3,
        num_candidates=30,
    )

    assert len(results) == 2  # only 2 music docs
    # doc3 ([0,1,0]) is exact match for [0,1,0]
    assert results[0]["_id"] == "doc3"
    assert results[0]["_svr_score"] == pytest.approx(1.0, abs=0.001)
    # All results are from the music partition
    for r in results:
        assert r["_svr_partition"] == "music"


async def test_execute_search_with_content_filter(seeded_backend: PostgresBackend):
    """execute_search() applies JSONB content filters."""
    partition = PartitionInfo(name="sports", filter_value="sports")

    results = await seeded_backend.execute_search(
        partition=partition,
        query_vector=[1.0, 0.0, 0.0],
        limit=10,
        num_candidates=30,
        filters={"title": "Football"},
    )

    assert len(results) == 1
    assert results[0]["_id"] == "doc1"
    assert results[0]["title"] == "Football"


async def test_search_partitions_fan_out(seeded_backend: PostgresBackend):
    """search_partitions() fans out across multiple partitions and merges results."""
    partitions = [
        PartitionInfo(name="sports", filter_value="sports"),
        PartitionInfo(name="music", filter_value="music"),
    ]

    results = await seeded_backend.search_partitions(
        partitions=partitions,
        limit=10,
        query_vector=[1.0, 0.0, 0.0],
    )

    # Should contain results from both partitions
    assert len(results) == 5  # 3 sports + 2 music
    partition_names = {r["_svr_partition"] for r in results}
    assert "sports" in partition_names
    assert "music" in partition_names


async def test_search_partitions_empty_list(seeded_backend: PostgresBackend):
    """search_partitions() returns empty list for empty partition list."""
    results = await seeded_backend.search_partitions(
        partitions=[],
        limit=10,
        query_vector=[1.0, 0.0, 0.0],
    )
    assert results == []


async def test_search_partitions_requires_query_vector(seeded_backend: PostgresBackend):
    """search_partitions() raises SearchError when query_vector is None."""
    from semantic_vector_router.exceptions import SearchError

    partitions = [PartitionInfo(name="sports", filter_value="sports")]

    with pytest.raises(SearchError, match="requires query_vector"):
        await seeded_backend.search_partitions(
            partitions=partitions,
            limit=10,
            query_vector=None,
        )


# ── Data operations ───────────────────────────────────────────────


async def test_get_distinct_values_partition_name(seeded_backend: PostgresBackend):
    """get_distinct_values('partition_name') returns all partition names."""
    values = await seeded_backend.get_distinct_values("partition_name")
    assert set(values) == {"sports", "music"}


async def test_get_distinct_values_jsonb_field(seeded_backend: PostgresBackend):
    """get_distinct_values() works for JSONB content fields."""
    values = await seeded_backend.get_distinct_values("category")
    assert set(values) == {"sports", "music"}


async def test_count_documents_total(seeded_backend: PostgresBackend):
    """count_documents() returns total row count."""
    count = await seeded_backend.count_documents()
    assert count == 5


async def test_count_documents_with_filter(seeded_backend: PostgresBackend):
    """count_documents() respects filter expressions."""
    count = await seeded_backend.count_documents(
        filter_expression={"partition_name": "sports"}
    )
    assert count == 3


async def test_get_collection_stats(seeded_backend: PostgresBackend):
    """get_collection_stats() returns dict with count, size, and backend."""
    stats = await seeded_backend.get_collection_stats()

    assert isinstance(stats, dict)
    assert stats["count"] == 5
    assert stats["size"] > 0
    assert stats["backend"] == "postgres"
    assert "public.svr_vectors" in stats["name"]


async def test_get_partition_document_counts(seeded_backend: PostgresBackend):
    """get_partition_document_counts() returns per-partition counts."""
    counts = await seeded_backend.get_partition_document_counts("category")

    assert isinstance(counts, dict)
    assert counts["sports"] == 3
    assert counts["music"] == 2


# ── Delete partition storage ──────────────────────────────────────


async def test_delete_partition_storage(backend: PostgresBackend):
    """delete_partition_storage() removes all rows for the given partition."""
    assert backend._pool is not None

    # Insert some rows to delete
    async with backend._pool.connection() as conn:
        await conn.execute(
            f"INSERT INTO {backend._fq_table} "
            f"(id, partition_name, embedding, content) VALUES "
            f"(%s, %s, %s::vector, %s::jsonb)",
            ("del1", "temp_partition", "[1,0,0]", '{"x": 1}'),
        )
        await conn.execute(
            f"INSERT INTO {backend._fq_table} "
            f"(id, partition_name, embedding, content) VALUES "
            f"(%s, %s, %s::vector, %s::jsonb)",
            ("del2", "temp_partition", "[0,1,0]", '{"x": 2}'),
        )
        await conn.execute(
            f"INSERT INTO {backend._fq_table} "
            f"(id, partition_name, embedding, content) VALUES "
            f"(%s, %s, %s::vector, %s::jsonb)",
            ("del3", "other_partition", "[0,0,1]", '{"x": 3}'),
        )

    partition = PartitionInfo(name="temp_partition", filter_value="temp_partition")
    await backend.delete_partition_storage(partition)

    # Verify temp_partition rows are gone
    async with backend._pool.connection() as conn:
        cur = await conn.execute(
            f"SELECT COUNT(*) FROM {backend._fq_table} WHERE partition_name = %s",
            ("temp_partition",),
        )
        row = await cur.fetchone()
        assert row[0] == 0

    # Verify other_partition rows are still there
    async with backend._pool.connection() as conn:
        cur = await conn.execute(
            f"SELECT COUNT(*) FROM {backend._fq_table} WHERE partition_name = %s",
            ("other_partition",),
        )
        row = await cur.fetchone()
        assert row[0] == 1

    # Clean up
    async with backend._pool.connection() as conn:
        await conn.execute(
            f"DELETE FROM {backend._fq_table} WHERE partition_name = %s",
            ("other_partition",),
        )


# ── Disconnect lifecycle ──────────────────────────────────────────


async def test_disconnect_and_reconnect(backend: PostgresBackend):
    """After disconnect(), is_connected() returns False; reconnect restores it."""
    # Disconnect
    await backend.disconnect()
    assert await backend.is_connected() is False

    # Reconnect for remaining test cleanup
    await backend.connect()
    assert await backend.is_connected() is True
