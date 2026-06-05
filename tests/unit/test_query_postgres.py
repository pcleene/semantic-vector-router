"""Unit tests for PostgreSQL query builder: $not filters, pre_native, post_native, exact mode.

Tests cover:
1. translate_filters $not operator support (top-level, field-level, nested)
2. execute_search pre_native WHERE injection
3. execute_search post_native CTE wrapping
4. execute_search exact mode (disable index scan)

All tests mock the psycopg connection pool — no real PostgreSQL connection.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_vector_router.backends.postgres.backend import PostgresBackend
from semantic_vector_router.backends.postgres.config import (
    PgDistanceMetric,
    PgIndexType,
    PostgresBackendConfig,
)
from semantic_vector_router.backends.postgres.filters import translate_filters
from semantic_vector_router.models.partition import PartitionInfo


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(**overrides):
    """Build a minimal SVRConfig for testing."""
    from semantic_vector_router.models.svr_config import SVRConfig

    base = {
        "database": {
            "backend": "postgres",
            "database": "testdb",
            "source_collection": "docs",
        },
        "partitioning": {"field": "category"},
        "vector_search": {"dimensions": 3, "similarity": "cosine"},
    }
    base.update(overrides)
    return SVRConfig(**base)


def _make_partition(name="electronics", filter_value=None, **kwargs):
    """Build a PartitionInfo for testing."""
    return PartitionInfo(
        name=name,
        filter_value=filter_value or name,
        **kwargs,
    )


def _mock_cursor(rows=None, fetchone_result=None):
    """Create a mock cursor with configurable results."""
    cursor = AsyncMock()
    if rows is not None:
        cursor.fetchall = AsyncMock(return_value=rows)
    else:
        cursor.fetchall = AsyncMock(return_value=[])
    if fetchone_result is not None:
        cursor.fetchone = AsyncMock(return_value=fetchone_result)
    else:
        cursor.fetchone = AsyncMock(return_value=None)
    return cursor


def _mock_conn(cursor=None):
    """Create a mock async connection."""
    conn = AsyncMock()
    if cursor:
        conn.execute = AsyncMock(return_value=cursor)
    else:
        conn.execute = AsyncMock(return_value=_mock_cursor())
    return conn


class MockConnectionCtx:
    """Mock async context manager for pool.connection()."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


def _make_backend_with_mock_pool(rows=None, pg_config=None):
    """Create a PostgresBackend with a mocked connection pool.

    Returns (backend, mock_conn) so tests can inspect call args.
    """
    config = _make_config(
        **({"postgres": pg_config.model_dump()} if pg_config else {})
    )
    backend = PostgresBackend(config)

    cursor = _mock_cursor(rows=rows or [])
    mock_conn = _mock_conn(cursor)
    mock_pool = MagicMock()
    mock_pool.connection.return_value = MockConnectionCtx(mock_conn)
    backend._pool = mock_pool

    return backend, mock_conn


def _get_sql_string(call_args):
    """Extract a renderable SQL string from a mock execute call.

    psycopg sql.Composed objects support .as_string(None) for rendering
    without a real connection. Plain strings are returned as-is.
    """
    query = call_args[0][0]
    if hasattr(query, "as_string"):
        return query.as_string(None)
    return str(query)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1: $not operator in translate_filters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNotOperatorInTranslateFilters:
    """Test $not operator support in translate_filters."""

    def test_top_level_not_simple_equality(self):
        """Top-level $not: {"$not": {"name": "test"}} negates the inner clause."""
        sql_str, params = translate_filters({"$not": {"name": "test"}})
        assert sql_str == "NOT (content->>'name' = %s)"
        assert params == ["test"]

    def test_top_level_not_multiple_fields(self):
        """Top-level $not with multiple fields inside."""
        sql_str, params = translate_filters(
            {"$not": {"name": "test", "status": "active"}}
        )
        assert "NOT (" in sql_str
        assert "content->>'name' = %s" in sql_str
        assert "content->>'status' = %s" in sql_str
        assert params == ["test", "active"]

    def test_field_level_not_with_eq_operator(self):
        """Field-level $not wrapping $eq: {"name": {"$not": {"$eq": "test"}}}."""
        sql_str, params = translate_filters({"name": {"$not": {"$eq": "test"}}})
        assert sql_str == "NOT (content->>'name' = %s)"
        assert params == ["test"]

    def test_field_level_not_with_scalar(self):
        """Field-level $not with scalar: {"name": {"$not": "test"}} becomes !=."""
        sql_str, params = translate_filters({"name": {"$not": "test"}})
        assert sql_str == "content->>'name' != %s"
        assert params == ["test"]

    def test_field_level_not_with_gt_comparison(self):
        """$not with numeric comparison: {"price": {"$not": {"$gt": 100}}}."""
        sql_str, params = translate_filters({"price": {"$not": {"$gt": 100}}})
        assert sql_str == "NOT (content->>'price'::numeric > %s)"
        assert params == [100]

    def test_field_level_not_with_lte_comparison(self):
        """$not with $lte: {"score": {"$not": {"$lte": 50}}}."""
        sql_str, params = translate_filters({"score": {"$not": {"$lte": 50}}})
        assert sql_str == "NOT (content->>'score'::numeric <= %s)"
        assert params == [50]

    def test_field_level_not_with_ne_comparison(self):
        """$not with $ne: double negation still produces correct SQL."""
        sql_str, params = translate_filters(
            {"status": {"$not": {"$ne": "active"}}}
        )
        assert sql_str == "NOT (content->>'status' != %s)"
        assert params == ["active"]

    def test_not_nested_inside_and(self):
        """$not inside $and: {"$and": [{"$not": {"name": "bad"}}, {"status": "ok"}]}."""
        sql_str, params = translate_filters(
            {"$and": [{"$not": {"name": "bad"}}, {"status": "ok"}]}
        )
        assert "NOT (content->>'name' = %s)" in sql_str
        assert "content->>'status' = %s" in sql_str
        # Both clauses are AND-joined inside parens
        assert sql_str.startswith("(")
        assert " AND " in sql_str
        assert params == ["bad", "ok"]

    def test_not_nested_inside_or(self):
        """$not inside $or: one negated clause, one normal."""
        sql_str, params = translate_filters(
            {"$or": [{"$not": {"active": True}}, {"role": "admin"}]}
        )
        assert "NOT (content->>'active' = %s)" in sql_str
        assert "content->>'role' = %s" in sql_str
        assert " OR " in sql_str
        assert params == [True, "admin"]

    def test_not_combined_with_other_operators_on_same_field(self):
        """$not alongside other operators: {"price": {"$not": {"$lt": 10}, "$gt": 5}}."""
        sql_str, params = translate_filters(
            {"price": {"$not": {"$lt": 10}, "$gt": 5}}
        )
        # Both conditions should appear, AND-joined
        assert "NOT (content->>'price'::numeric < %s)" in sql_str
        assert "content->>'price'::numeric > %s" in sql_str
        assert 10 in params
        assert 5 in params

    def test_top_level_not_with_empty_dict(self):
        """$not with empty dict produces no output (inner translate returns "")."""
        sql_str, params = translate_filters({"$not": {}})
        assert sql_str == ""
        assert params == []

    def test_not_on_column_field(self):
        """$not on a top-level column field like partition_name."""
        sql_str, params = translate_filters(
            {"$not": {"partition_name": "archived"}}
        )
        assert sql_str == "NOT (partition_name = %s)"
        assert params == ["archived"]

    def test_field_level_not_with_in_operator(self):
        """$not wrapping $in: {"status": {"$not": {"$in": ["a", "b"]}}}."""
        sql_str, params = translate_filters(
            {"status": {"$not": {"$in": ["a", "b"]}}}
        )
        assert sql_str == "NOT (content->>'status' IN (%s, %s))"
        assert params == ["a", "b"]

    def test_field_level_not_scalar_numeric(self):
        """$not with scalar numeric: {"count": {"$not": 0}}."""
        sql_str, params = translate_filters({"count": {"$not": 0}})
        assert sql_str == "content->>'count' != %s"
        assert params == [0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2: pre_native in execute_search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPreNativeInExecuteSearch:
    """Test pre_native raw SQL injection into WHERE clause."""

    async def test_pre_native_adds_to_where_clause(self):
        """pre_native content appears in the WHERE clause."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            pre_native="content->>'color' = 'red'",
        )

        # The search query is the last execute call (first is SET hnsw...)
        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])
        assert "content->>'color' = 'red'" in search_sql

    async def test_pre_native_is_parenthesized(self):
        """pre_native is wrapped in parentheses to prevent precedence issues."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            pre_native="x = 1 OR y = 2",
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])
        # Must be parenthesized: AND (x = 1 OR y = 2)
        assert "(x = 1 OR y = 2)" in search_sql

    async def test_pre_native_combined_with_svr_filters(self):
        """pre_native and SVR filters both appear in WHERE clause."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            filters={"brand": "apple"},
            pre_native="content->>'in_stock' = 'true'",
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])

        # All three WHERE parts should be present
        assert "partition_name = %s" in search_sql
        assert "content->>'brand' = %s" in search_sql
        assert "(content->>'in_stock' = 'true')" in search_sql

    async def test_pre_native_without_svr_filters(self):
        """pre_native without SVR filters still AND-joins with partition_name."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            pre_native="created_at > '2024-01-01'",
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])

        # partition_name filter is always first
        assert "partition_name = %s" in search_sql
        # pre_native is AND-joined
        assert "AND (created_at > '2024-01-01')" in search_sql

    async def test_pre_native_with_complex_sql(self):
        """pre_native with subquery-style SQL."""
        pre = (
            "content->>'category_id' IN "
            "(SELECT id FROM categories WHERE active = true)"
        )
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            pre_native=pre,
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])
        assert f"({pre})" in search_sql


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3: post_native in execute_search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPostNativeInExecuteSearch:
    """Test post_native CTE wrapping behavior."""

    async def test_post_native_wraps_in_cte(self):
        """post_native wraps the core query in WITH svr_results AS (...)."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        post_sql = "SELECT * FROM svr_results WHERE score > 0.5"

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            post_native=post_sql,
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])

        assert "WITH svr_results AS (" in search_sql
        assert post_sql in search_sql

    async def test_no_post_native_no_cte_wrapper(self):
        """Without post_native, no CTE wrapper is added."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])

        assert "WITH svr_results AS" not in search_sql
        # Should still have the core SELECT
        assert "SELECT" in search_sql
        assert "ORDER BY" in search_sql

    async def test_post_native_with_join(self):
        """post_native with a JOIN (real-world enrichment use case)."""
        post_sql = (
            "SELECT r.*, p.display_name "
            "FROM svr_results r "
            "JOIN products p ON r.id = p.id "
            "ORDER BY r.score DESC"
        )
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            post_native=post_sql,
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])

        assert "WITH svr_results AS (" in search_sql
        assert "JOIN products p ON r.id = p.id" in search_sql

    async def test_post_native_with_aggregation(self):
        """post_native with GROUP BY aggregation."""
        post_sql = (
            "SELECT partition_name, COUNT(*) as cnt, AVG(score) as avg_score "
            "FROM svr_results GROUP BY partition_name"
        )
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            post_native=post_sql,
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])

        assert "WITH svr_results AS (" in search_sql
        assert "GROUP BY partition_name" in search_sql

    async def test_post_native_combined_with_pre_native(self):
        """Both pre_native and post_native used together."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            pre_native="content->>'status' = 'published'",
            post_native="SELECT * FROM svr_results LIMIT 5",
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])

        # CTE wrapper present
        assert "WITH svr_results AS (" in search_sql
        # pre_native in the inner WHERE
        assert "(content->>'status' = 'published')" in search_sql
        # post_native as outer query
        assert "SELECT * FROM svr_results LIMIT 5" in search_sql


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 4: exact mode in execute_search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExactModeInExecuteSearch:
    """Test exact=True disables index scan, exact=False sets index config."""

    async def test_exact_true_disables_index_scan(self):
        """exact=True executes SET LOCAL enable_indexscan = off."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            exact=True,
        )

        calls = mock_conn.execute.call_args_list
        # First call should be the SET LOCAL
        first_sql = _get_sql_string(calls[0])
        assert "SET LOCAL enable_indexscan = off" in first_sql

    async def test_exact_false_sets_normal_search_config(self):
        """exact=False (default) sets HNSW ef_search parameter."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            exact=False,
        )

        calls = mock_conn.execute.call_args_list
        first_sql = _get_sql_string(calls[0])
        assert "hnsw.ef_search" in first_sql
        assert "enable_indexscan" not in first_sql

    async def test_exact_true_does_not_set_hnsw_config(self):
        """When exact=True, no HNSW/IVFFlat config SET is issued."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            exact=True,
        )

        calls = mock_conn.execute.call_args_list
        all_sqls = [_get_sql_string(c) for c in calls]
        # No hnsw.ef_search should appear in any call
        for s in all_sqls:
            assert "hnsw.ef_search" not in s

    async def test_exact_true_with_filters(self):
        """exact=True combined with SVR filters — both work."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            filters={"brand": "sony"},
            exact=True,
        )

        calls = mock_conn.execute.call_args_list
        # First call: SET LOCAL
        first_sql = _get_sql_string(calls[0])
        assert "SET LOCAL enable_indexscan = off" in first_sql

        # Second call: search query with filters
        search_sql = _get_sql_string(calls[-1])
        assert "content->>'brand' = %s" in search_sql
        assert "partition_name = %s" in search_sql

    async def test_exact_true_with_pre_native(self):
        """exact=True combined with pre_native."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            exact=True,
            pre_native="content->>'verified' = 'true'",
        )

        calls = mock_conn.execute.call_args_list
        first_sql = _get_sql_string(calls[0])
        assert "SET LOCAL enable_indexscan = off" in first_sql

        search_sql = _get_sql_string(calls[-1])
        assert "(content->>'verified' = 'true')" in search_sql

    async def test_exact_false_with_ivfflat_sets_probes(self):
        """exact=False with IVFFlat index sets ivfflat.probes."""
        pg_config = PostgresBackendConfig(index_type=PgIndexType.IVFFLAT)
        backend, mock_conn = _make_backend_with_mock_pool(
            pg_config=pg_config
        )
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
            exact=False,
        )

        calls = mock_conn.execute.call_args_list
        first_sql = _get_sql_string(calls[0])
        assert "ivfflat.probes" in first_sql


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 5: Parameter correctness in execute_search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExecuteSearchParameters:
    """Verify that the SQL parameters tuple is correctly assembled."""

    async def test_basic_search_params(self):
        """Basic search: params = (vector_literal, partition_name, vector_literal, limit)."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition("electronics")

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        calls = mock_conn.execute.call_args_list
        # The search query call (last one)
        search_call = calls[-1]
        params = search_call[0][1]

        # Params should be: (vector_literal, partition_name, vector_literal, limit)
        vector_literal = "[1.0,0.0,0.0]"
        assert params[0] == vector_literal  # score expression vector
        assert "electronics" in params  # partition_name
        assert params[-1] == 10  # limit

    async def test_search_with_filter_params(self):
        """Filter params are injected between partition_name and vector + limit."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition("electronics")

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=5,
            num_candidates=50,
            filters={"brand": "apple"},
        )

        calls = mock_conn.execute.call_args_list
        search_call = calls[-1]
        params = search_call[0][1]

        # "apple" should be in the params (for the brand filter)
        assert "apple" in params
        # Limit should be last
        assert params[-1] == 5

    async def test_search_result_mapping(self):
        """Verify result rows are correctly mapped to dicts."""
        rows = [
            ("doc1", {"title": "Laptop", "brand": "Dell"}, "electronics", 0.92),
            ("doc2", {"title": "Phone"}, "electronics", 0.85),
        ]
        backend, mock_conn = _make_backend_with_mock_pool(rows=rows)
        partition = _make_partition("electronics")

        results = await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        assert len(results) == 2
        assert results[0]["_id"] == "doc1"
        assert results[0]["title"] == "Laptop"
        assert results[0]["brand"] == "Dell"
        assert results[0]["_svr_score"] == 0.92
        assert results[0]["_svr_partition"] == "electronics"
        assert results[1]["_id"] == "doc2"
        assert results[1]["_svr_score"] == 0.85

    async def test_search_with_non_dict_content(self):
        """If content column is not a dict, result has just _id."""
        rows = [
            ("doc1", None, "electronics", 0.8),
        ]
        backend, mock_conn = _make_backend_with_mock_pool(rows=rows)
        partition = _make_partition("electronics")

        results = await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        assert len(results) == 1
        assert results[0]["_id"] == "doc1"
        assert results[0]["_svr_score"] == 0.8

    async def test_partition_filter_value_used(self):
        """Partition's filter_value is used in WHERE, not name if different."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition("elec", filter_value="electronics_v2")

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        calls = mock_conn.execute.call_args_list
        search_call = calls[-1]
        params = search_call[0][1]

        # Should use filter_value, not name
        assert "electronics_v2" in params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 6: Additional translate_filters edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTranslateFiltersEdgeCases:
    """Additional translate_filters tests for completeness."""

    def test_empty_filters(self):
        """Empty dict returns empty string and empty params."""
        sql_str, params = translate_filters({})
        assert sql_str == ""
        assert params == []

    def test_simple_equality(self):
        """Basic equality: {"name": "test"}."""
        sql_str, params = translate_filters({"name": "test"})
        assert sql_str == "content->>'name' = %s"
        assert params == ["test"]

    def test_column_field_equality(self):
        """Column field partition_name is referenced directly."""
        sql_str, params = translate_filters({"partition_name": "electronics"})
        assert sql_str == "partition_name = %s"
        assert params == ["electronics"]

    def test_gt_numeric(self):
        """Numeric $gt adds ::numeric cast."""
        sql_str, params = translate_filters({"price": {"$gt": 100}})
        assert sql_str == "content->>'price'::numeric > %s"
        assert params == [100]

    def test_in_operator(self):
        """$in produces IN clause with placeholders."""
        sql_str, params = translate_filters(
            {"status": {"$in": ["active", "pending"]}}
        )
        assert sql_str == "content->>'status' IN (%s, %s)"
        assert params == ["active", "pending"]

    def test_in_empty_list(self):
        """$in with empty list produces FALSE."""
        sql_str, params = translate_filters({"status": {"$in": []}})
        assert sql_str == "FALSE"
        assert params == []

    def test_nin_operator(self):
        """$nin produces NOT IN clause."""
        sql_str, params = translate_filters(
            {"status": {"$nin": ["deleted", "archived"]}}
        )
        assert sql_str == "content->>'status' NOT IN (%s, %s)"
        assert params == ["deleted", "archived"]

    def test_nin_empty_list(self):
        """$nin with empty list produces no filter (matches everything)."""
        sql_str, params = translate_filters({"status": {"$nin": []}})
        assert sql_str == ""
        assert params == []

    def test_exists_true(self):
        """$exists: true uses JSONB ? operator."""
        sql_str, params = translate_filters({"email": {"$exists": True}})
        assert sql_str == "content ? %s"
        assert params == ["email"]

    def test_exists_false(self):
        """$exists: false negates JSONB ? operator."""
        sql_str, params = translate_filters({"email": {"$exists": False}})
        assert sql_str == "NOT (content ? %s)"
        assert params == ["email"]

    def test_and_operator(self):
        """$and produces AND-joined parenthesized clause."""
        sql_str, params = translate_filters(
            {"$and": [{"a": 1}, {"b": 2}]}
        )
        assert sql_str == "(content->>'a' = %s AND content->>'b' = %s)"
        assert params == [1, 2]

    def test_or_operator(self):
        """$or produces OR-joined parenthesized clause."""
        sql_str, params = translate_filters(
            {"$or": [{"a": 1}, {"b": 2}]}
        )
        assert sql_str == "(content->>'a' = %s OR content->>'b' = %s)"
        assert params == [1, 2]

    def test_unsupported_operator_raises(self):
        """Unsupported operators raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            translate_filters({"field": {"$regex": ".*"}})

    def test_multiple_fields_and_joined(self):
        """Multiple top-level fields are AND-joined."""
        sql_str, params = translate_filters({"a": 1, "b": "x"})
        assert "content->>'a' = %s" in sql_str
        assert "content->>'b' = %s" in sql_str
        assert " AND " in sql_str
        assert params == [1, "x"]

    def test_string_comparison_no_cast(self):
        """String values don't get ::numeric cast."""
        sql_str, params = translate_filters({"name": {"$gt": "abc"}})
        assert "::numeric" not in sql_str
        assert sql_str == "content->>'name' > %s"

    def test_float_comparison_gets_cast(self):
        """Float values get ::numeric cast."""
        sql_str, params = translate_filters({"score": {"$lte": 3.14}})
        assert sql_str == "content->>'score'::numeric <= %s"
        assert params == [3.14]

    def test_invalid_field_name_raises(self):
        """Invalid field names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid field name"):
            translate_filters({"field; DROP TABLE": "val"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 7: Distance metric variations in execute_search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDistanceMetricVariations:
    """Test that different distance metrics produce correct SQL operators."""

    async def test_cosine_distance_operator(self):
        """Cosine metric uses <=> operator."""
        backend, mock_conn = _make_backend_with_mock_pool()
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])
        assert "<=>" in search_sql

    async def test_l2_distance_operator(self):
        """L2 metric uses <-> operator."""
        pg_config = PostgresBackendConfig(distance_metric=PgDistanceMetric.L2)
        backend, mock_conn = _make_backend_with_mock_pool(pg_config=pg_config)
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])
        assert "<->" in search_sql

    async def test_inner_product_distance_operator(self):
        """Inner product metric uses <#> operator."""
        pg_config = PostgresBackendConfig(
            distance_metric=PgDistanceMetric.INNER_PRODUCT
        )
        backend, mock_conn = _make_backend_with_mock_pool(pg_config=pg_config)
        partition = _make_partition()

        await backend.execute_search(
            partition=partition,
            query_vector=[1.0, 0.0, 0.0],
            limit=10,
            num_candidates=100,
        )

        calls = mock_conn.execute.call_args_list
        search_sql = _get_sql_string(calls[-1])
        assert "<#>" in search_sql
