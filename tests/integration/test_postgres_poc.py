"""Proof-of-concept: PostgreSQL + pgvector connectivity.

Validates that:
1. Local PostgreSQL is reachable
2. pgvector extension works (vector type, distance operators)
3. The Phase 12 abstraction layer interfaces are sound for future Postgres backend

This test uses raw psycopg — a full PostgresBackend implementation
is Phase 13 work. This test proves the infrastructure is ready.
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

# Connection string for local PostgreSQL
PG_DSN = "postgresql://<user>:<password>@<host>:5432/<db>"


@pytest.fixture(scope="module")
async def pg_conn():
    """Create an async PostgreSQL connection."""
    try:
        import psycopg

        conn = await psycopg.AsyncConnection.connect(PG_DSN, autocommit=True)
        yield conn
        await conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


async def test_postgres_connection(pg_conn):
    """PostgreSQL should be reachable and accept queries."""
    cur = await pg_conn.execute("SELECT version()")
    row = await cur.fetchone()
    assert row is not None
    assert "PostgreSQL" in row[0]


async def test_pgvector_extension(pg_conn):
    """pgvector extension should be installed and functional."""
    cur = await pg_conn.execute("SELECT vector_dims('[1,2,3]'::vector)")
    row = await cur.fetchone()
    assert row[0] == 3


async def test_vector_table_crud(pg_conn):
    """Create table, insert vectors, query by cosine distance."""
    # Setup
    await pg_conn.execute("DROP TABLE IF EXISTS svr_poc_vectors")
    await pg_conn.execute("""
        CREATE TABLE svr_poc_vectors (
            id SERIAL PRIMARY KEY,
            partition_name TEXT NOT NULL,
            content TEXT,
            embedding vector(3)
        )
    """)

    # Insert test vectors
    await pg_conn.execute("""
        INSERT INTO svr_poc_vectors (partition_name, content, embedding) VALUES
        ('electronics', 'laptop', '[1,0,0]'),
        ('electronics', 'phone', '[0.9,0.1,0]'),
        ('clothing', 'shirt', '[0,1,0]'),
        ('clothing', 'pants', '[0,0.9,0.1]')
    """)

    # Query: find closest to [1,0,0] (should be laptop, then phone)
    cur = await pg_conn.execute("""
        SELECT content, embedding <=> '[1,0,0]'::vector AS distance
        FROM svr_poc_vectors
        ORDER BY distance
        LIMIT 2
    """)
    rows = await cur.fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "laptop"
    assert rows[0][1] == 0.0  # exact match
    assert rows[1][0] == "phone"

    # Query with partition filter (like SVR would do)
    cur = await pg_conn.execute("""
        SELECT content, embedding <=> '[0,1,0]'::vector AS distance
        FROM svr_poc_vectors
        WHERE partition_name = 'clothing'
        ORDER BY distance
        LIMIT 2
    """)
    rows = await cur.fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "shirt"

    # Cleanup
    await pg_conn.execute("DROP TABLE svr_poc_vectors")


async def test_ivfflat_index(pg_conn):
    """Create an IVFFlat index (pgvector's ANN index)."""
    await pg_conn.execute("DROP TABLE IF EXISTS svr_poc_ann")
    await pg_conn.execute("""
        CREATE TABLE svr_poc_ann (
            id SERIAL PRIMARY KEY,
            embedding vector(4)
        )
    """)

    # Insert enough rows for IVFFlat (needs at least sqrt(N) lists)
    for i in range(100):
        v = [float(i % 10), float(i % 5), float(i % 3), float(i % 2)]
        await pg_conn.execute(
            "INSERT INTO svr_poc_ann (embedding) VALUES (%s::vector)",
            (f"[{v[0]},{v[1]},{v[2]},{v[3]}]",),
        )

    # Create IVFFlat index
    await pg_conn.execute("""
        CREATE INDEX svr_poc_ann_idx ON svr_poc_ann
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 5)
    """)

    # Use the index
    cur = await pg_conn.execute("""
        SET ivfflat.probes = 3;
    """)
    cur = await pg_conn.execute("""
        SELECT id, embedding <=> '[1,1,1,1]'::vector AS distance
        FROM svr_poc_ann
        ORDER BY distance
        LIMIT 5
    """)
    rows = await cur.fetchall()
    assert len(rows) == 5

    # Cleanup
    await pg_conn.execute("DROP TABLE svr_poc_ann")


async def test_hnsw_index(pg_conn):
    """Create an HNSW index (pgvector's graph-based ANN index)."""
    await pg_conn.execute("DROP TABLE IF EXISTS svr_poc_hnsw")
    await pg_conn.execute("""
        CREATE TABLE svr_poc_hnsw (
            id SERIAL PRIMARY KEY,
            embedding vector(3)
        )
    """)

    # Insert test data
    for i in range(50):
        v = [float(i % 10) / 10, float(i % 7) / 7, float(i % 3) / 3]
        await pg_conn.execute(
            "INSERT INTO svr_poc_hnsw (embedding) VALUES (%s::vector)",
            (f"[{v[0]},{v[1]},{v[2]}]",),
        )

    # Create HNSW index (preferred for Phase 13)
    await pg_conn.execute("""
        CREATE INDEX svr_poc_hnsw_idx ON svr_poc_hnsw
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 8, ef_construction = 32)
    """)

    # Query using the HNSW index
    cur = await pg_conn.execute("""
        SELECT id, embedding <=> '[0.5,0.5,0.5]'::vector AS distance
        FROM svr_poc_hnsw
        ORDER BY distance
        LIMIT 3
    """)
    rows = await cur.fetchall()
    assert len(rows) == 3

    # Cleanup
    await pg_conn.execute("DROP TABLE svr_poc_hnsw")


async def test_abstraction_layer_compatibility(pg_conn):
    """Verify the SVR abstraction models work with Postgres concepts."""
    from semantic_vector_router.models.backend import (
        IndexStatus,
        PartitionStorageResult,
    )

    # Simulate what a PostgresBackend.create_partition_storage would return
    result = PartitionStorageResult(
        storage_name="svr_partition_electronics",
        storage_type="table",
        metadata={
            "schema": "public",
            "index_type": "hnsw",
            "vector_dimensions": 1536,
        },
    )
    assert result.storage_type == "table"
    assert result.metadata["index_type"] == "hnsw"

    # Simulate index status mapping
    # In Postgres, indexes build synchronously, so status is always READY or ERROR
    assert IndexStatus.READY == "ready"
    assert IndexStatus.NOT_FOUND == "not_found"
