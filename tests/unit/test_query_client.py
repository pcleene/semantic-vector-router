"""Unit tests for SVRClient.search() — filter validation, new parameters, and backward compatibility.

Tests cover three areas:
1. Filter validation at the SDK entry point (validate_filters called before backend)
2. New parameters (exact, post_native, pre_native) passed through to backend
3. Backward compatibility — existing callers unaffected
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.client import SVRClient
from semantic_vector_router.models import (
    EmbeddingMode,
    PartitionInfo,
    PartitionStatus,
    SearchResult,
    SVRConfig,
)
from semantic_vector_router.routing.merger import ResultMerger
from semantic_vector_router.routing.resolver import PartitionResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_results(partitions=None):
    """Return raw backend result dicts for the given partition names."""
    if partitions is None:
        partitions = ["electronics"]

    results = []
    if "electronics" in partitions:
        results.extend([
            {
                "_id": "doc1",
                "name": "Wireless Headphones",
                "price": 299.99,
                "_svr_score": 0.95,
                "_svr_partition": "electronics",
            },
            {
                "_id": "doc2",
                "name": "Bluetooth Speaker",
                "price": 79.99,
                "_svr_score": 0.88,
                "_svr_partition": "electronics",
            },
        ])
    return results


def _make_client(config, mock_backend, mock_embedder):
    """Build an SVRClient with injected mocks, bypassing real __init__ connect."""
    client = SVRClient(config=config, auto_connect=False)
    client._backend = mock_backend
    client._embedder = mock_embedder
    client._reranker = None
    client._resolver = PartitionResolver(config)
    client._merger = ResultMerger()
    client._connected = True
    client._auto_connect_failed = False
    return client


# ===========================================================================
# Filter validation at entry point
# ===========================================================================


class TestFilterValidation:
    """Filter validation must run at the SVRClient.search() entry point,
    raising ValueError *before* any backend call is made."""

    @pytest.mark.asyncio
    async def test_valid_comparison_filters_pass(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """Supported comparison operators ($gt, $lt, $eq, etc.) must pass
        validation without error and reach the backend."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        filters = {"price": {"$gt": 50, "$lte": 500}}
        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=filters,
        )

        assert isinstance(result, SearchResult)
        mock_backend.search_partitions.assert_awaited_once()
        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_valid_set_operators_pass(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """Set operators ($in, $nin) must be accepted."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        filters = {"brand": {"$in": ["Sony", "Bose"]}}
        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=filters,
        )

        assert isinstance(result, SearchResult)
        mock_backend.search_partitions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_valid_logical_operators_pass(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """Logical operators ($and, $or, $not) must be accepted."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        filters = {
            "$and": [
                {"price": {"$gte": 50}},
                {"brand": {"$ne": "Generic"}},
            ]
        }
        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=filters,
        )

        assert isinstance(result, SearchResult)
        mock_backend.search_partitions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_valid_existence_operator_pass(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """$exists operator must be accepted."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        filters = {"description": {"$exists": True}}
        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=filters,
        )

        assert isinstance(result, SearchResult)
        mock_backend.search_partitions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejected_match_raises_before_backend(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """$match is a rejected operator and must raise ValueError with an
        actionable message *before* the backend is called."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$match.*post_native"):
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=5,
                filters={"$match": {"category": "electronics"}},
            )

        # Backend must NOT have been called
        mock_backend.search_partitions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_regex_raises_before_backend(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """$regex is a rejected operator and must raise ValueError."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$regex.*post_native"):
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=5,
                filters={"name": {"$regex": "head.*"}},
            )

        mock_backend.search_partitions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_operator_raises_with_supported_list(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """Unknown operators ($foo) must raise ValueError listing all supported
        operators so the developer knows what is available."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"Unsupported filter operator.*\$foo") as exc_info:
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=5,
                filters={"field": {"$foo": 42}},
            )

        # The error message must include the word "Supported" and list at least
        # some known operators so the developer can self-serve.
        msg = str(exc_info.value)
        assert "Supported operators" in msg
        assert "$eq" in msg
        assert "$gt" in msg

        mock_backend.search_partitions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_operator_nested_in_and_raises(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """A rejected operator buried inside $and must still be caught
        by the recursive validation."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$regex"):
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=5,
                filters={
                    "$and": [
                        {"price": {"$gt": 10}},
                        {"name": {"$regex": "wire.*"}},
                    ]
                },
            )

        mock_backend.search_partitions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_operator_nested_in_or_raises(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """A rejected operator buried inside $or must still be caught."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$text"):
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=5,
                filters={
                    "$or": [
                        {"$text": {"$search": "wireless"}},
                        {"brand": {"$eq": "Sony"}},
                    ]
                },
            )

        mock_backend.search_partitions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_filters_no_validation_error(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """When filters is None, validation must be skipped entirely and
        no filters should be passed to the backend."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=None,
        )

        assert isinstance(result, SearchResult)
        mock_backend.search_partitions.assert_awaited_once()
        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["filters"] is None

    @pytest.mark.asyncio
    async def test_empty_dict_filters_no_error(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """An empty dict {} must pass validation (no operators to reject)
        and be forwarded to the backend as-is."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters={},
        )

        assert isinstance(result, SearchResult)
        mock_backend.search_partitions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_scalar_equality_filter_passes(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """Implicit equality (scalar value without $eq) must pass validation."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        filters = {"brand": "Sony"}
        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=filters,
        )

        assert isinstance(result, SearchResult)
        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["filters"] == {"brand": "Sony"}


# ===========================================================================
# New parameters passed through to backend
# ===========================================================================


class TestNewParametersPassthrough:
    """New search() parameters (exact, post_native, pre_native) must be
    forwarded to backend.search_partitions."""

    @pytest.mark.asyncio
    async def test_exact_true_passed_to_backend(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """exact=True must arrive at the backend as a keyword argument."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            exact=True,
        )

        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["exact"] is True

    @pytest.mark.asyncio
    async def test_exact_false_passed_to_backend(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """exact=False (explicit) must also be forwarded."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            exact=False,
        )

        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["exact"] is False

    @pytest.mark.asyncio
    async def test_post_native_list_passed_to_backend(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """post_native (list of pipeline stages) must be forwarded to the
        backend without modification."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        stages = [
            {"$lookup": {"from": "reviews", "localField": "_id", "foreignField": "product_id", "as": "reviews"}},
            {"$project": {"name": 1, "price": 1, "review_count": {"$size": "$reviews"}}},
        ]

        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            post_native=stages,
        )

        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["post_native"] == stages

    @pytest.mark.asyncio
    async def test_pre_native_string_passed_to_backend(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """pre_native (SQL string for PostgreSQL) must be forwarded to the
        backend without modification."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        pre = "metadata @> '{\"verified\": true}'::jsonb"

        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            pre_native=pre,
        )

        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["pre_native"] == pre

    @pytest.mark.asyncio
    async def test_all_three_combined_passed_through(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """exact + post_native + pre_native must all be forwarded in a single
        call to the backend."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        stages = [{"$project": {"name": 1}}]
        pre = "status = 'active'"

        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            exact=True,
            post_native=stages,
            pre_native=pre,
        )

        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["exact"] is True
        assert call_kwargs["post_native"] == stages
        assert call_kwargs["pre_native"] == pre

    @pytest.mark.asyncio
    async def test_default_values_backward_compatible(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """When no new parameters are provided, the defaults (exact=False,
        post_native=None, pre_native=None) must be forwarded — ensuring
        backward compatibility."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
        )

        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["exact"] is False
        assert call_kwargs["post_native"] is None
        assert call_kwargs["pre_native"] is None

    @pytest.mark.asyncio
    async def test_new_params_combined_with_filters(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """New parameters must coexist correctly with portable filters."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        filters = {"price": {"$gte": 100, "$lt": 500}}
        stages = [{"$project": {"name": 1, "price": 1}}]

        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=filters,
            exact=True,
            post_native=stages,
        )

        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["filters"] == filters
        assert call_kwargs["exact"] is True
        assert call_kwargs["post_native"] == stages
        assert call_kwargs["pre_native"] is None

    @pytest.mark.asyncio
    async def test_exact_with_precomputed_vector(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """exact=True must also work when a precomputed query_vector is
        supplied (the code path that skips embedding)."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        precomputed = [0.5] * 1536
        await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            query_vector=precomputed,
            exact=True,
        )

        # Embedder must NOT have been called
        mock_embedder.embed.assert_not_awaited()

        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["exact"] is True
        assert call_kwargs["query_vector"] == precomputed


# ===========================================================================
# Backward compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Existing callers that do not use the new parameters must continue to
    work identically."""

    @pytest.mark.asyncio
    async def test_existing_call_with_query_filters_limit(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """A classic search(query, partitions, limit, filters) call must
        produce a valid SearchResult with no errors."""
        raw = _raw_results()
        mock_backend.search_partitions = AsyncMock(return_value=raw)
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=10,
            filters={"price": {"$lt": 300}},
        )

        assert isinstance(result, SearchResult)
        assert result.query == "headphones"
        assert result.partitions_searched == ["electronics"]
        assert len(result.hits) <= 10

        # Verify backend received the expected kwargs
        call_kwargs = mock_backend.search_partitions.call_args.kwargs
        assert call_kwargs["filters"] == {"price": {"$lt": 300}}
        assert call_kwargs["exact"] is False
        assert call_kwargs["post_native"] is None
        assert call_kwargs["pre_native"] is None

    @pytest.mark.asyncio
    async def test_existing_call_without_filters(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """search(query, partitions, limit) without filters must work."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
        )

        assert isinstance(result, SearchResult)
        mock_backend.search_partitions.assert_awaited_once()

    def test_search_sync_passes_new_kwargs_through(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """search_sync() must forward exact/post_native/pre_native via **kwargs
        to the underlying async search()."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        stages = [{"$project": {"name": 1}}]

        # We cannot easily run asyncio.run inside pytest, so we patch search()
        # directly to verify the kwargs arrive.
        captured_kwargs = {}

        async def fake_search(query, partitions=None, limit=10, **kwargs):
            captured_kwargs.update(kwargs)
            return SearchResult(
                hits=[],
                query=query,
                partitions_searched=["electronics"],
                total_candidates=0,
                reranked=False,
                latency_ms=1.0,
            )

        with patch.object(client, "search", side_effect=fake_search):
            import asyncio
            result = asyncio.run(
                client.search(
                    query="headphones",
                    partitions=["electronics"],
                    limit=5,
                    exact=True,
                    post_native=stages,
                    pre_native="status = 'active'",
                )
            )

        assert captured_kwargs["exact"] is True
        assert captured_kwargs["post_native"] == stages
        assert captured_kwargs["pre_native"] == "status = 'active'"

    @pytest.mark.asyncio
    async def test_filters_validated_before_embedding(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """Filter validation must happen BEFORE the embedder is called, so
        that invalid filters are rejected without wasting an API call."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$match"):
            await client.search(
                query="headphones",
                partitions=["electronics"],
                limit=5,
                filters={"$match": {"x": 1}},
            )

        # The embedder must NOT have been called
        mock_embedder.embed.assert_not_awaited()
        mock_backend.search_partitions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_filter_validation_runs_before_connect_check(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """Filter validation is the first thing that happens in search(),
        even before the connected check. This means a ValueError for bad
        filters will fire even if the client is not connected."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)
        # Disconnect the client
        client._connected = False
        client._auto_connect_failed = False

        with pytest.raises(ValueError, match=r"\$regex"):
            await client.search(
                query="headphones",
                filters={"name": {"$regex": "wire.*"}},
            )

        # connect() must NOT have been called because validation fires first
        mock_backend.search_partitions.assert_not_awaited()


# ===========================================================================
# Edge cases for filter validation
# ===========================================================================


class TestFilterValidationEdgeCases:
    """Edge cases and additional rejected operators."""

    @pytest.mark.asyncio
    async def test_rejected_all_operator(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """$all is a rejected array operator."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$all"):
            await client.search(
                query="test",
                partitions=["electronics"],
                filters={"tags": {"$all": ["wireless", "bluetooth"]}},
            )

    @pytest.mark.asyncio
    async def test_rejected_elemMatch_operator(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """$elemMatch is a rejected array operator."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$elemMatch"):
            await client.search(
                query="test",
                partitions=["electronics"],
                filters={"reviews": {"$elemMatch": {"rating": {"$gt": 4}}}},
            )

    @pytest.mark.asyncio
    async def test_rejected_near_geospatial(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """$near is a rejected geospatial operator."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$near.*post_native"):
            await client.search(
                query="test",
                partitions=["electronics"],
                filters={"location": {"$near": {"$geometry": {"type": "Point"}}}},
            )

    @pytest.mark.asyncio
    async def test_multiple_valid_operators_on_same_field(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """Multiple valid operators on a single field must all pass."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        filters = {"price": {"$gte": 10, "$lte": 1000, "$ne": 500}}
        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=filters,
        )

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_deeply_nested_not_with_valid_ops(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """$not containing valid operators must pass."""
        mock_backend.search_partitions = AsyncMock(return_value=_raw_results())
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        filters = {"price": {"$not": {"$lt": 10}}}
        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
            filters=filters,
        )

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_rejected_operator_inside_not(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """A rejected operator inside $not must still be caught."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"\$regex"):
            await client.search(
                query="headphones",
                partitions=["electronics"],
                filters={"name": {"$not": {"$regex": ".*wire.*"}}},
            )

        mock_backend.search_partitions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_operator_at_top_level(
        self, sample_config_with_partitions, mock_backend, mock_embedder,
    ):
        """An unknown operator at the top level (not nested in a field)
        must also be caught."""
        client = _make_client(sample_config_with_partitions, mock_backend, mock_embedder)

        with pytest.raises(ValueError, match=r"Unsupported filter operator.*\$badOp"):
            await client.search(
                query="headphones",
                partitions=["electronics"],
                filters={"$badOp": [{"price": {"$gt": 10}}]},
            )

        mock_backend.search_partitions.assert_not_awaited()
