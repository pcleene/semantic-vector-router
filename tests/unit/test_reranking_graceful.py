"""Unit tests for graceful reranking degradation in SVRClient.connect().

When reranking is enabled but the API key environment variable is not set,
connect() should gracefully degrParts Distributor by setting self._reranker = None and
logging an informational message, rather than raising an error. This allows
the client to fall back to score-based merging without reranking.
"""

import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.client import SVRClient
from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
    RerankingConfig,
    RerankerProvider,
    SearchResult,
    SVRConfig,
    VectorSearchConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    reranking_enabled: bool = True,
    api_key_env: str | None = "VOYAGE_API_KEY",
) -> SVRConfig:
    """Build an SVRConfig with the given reranking settings."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(field="category"),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key_env="OPENAI_API_KEY",
            dimensions=1536,
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=1536,
            similarity="cosine",
        ),
        reranking=RerankingConfig(
            enabled=reranking_enabled,
            provider=RerankerProvider.VOYAGE,
            model="rerank-2",
            api_key_env=api_key_env,
        ),
    )


def _mock_backend():
    """Create a mock backend with the minimal interface needed by connect()."""
    backend = AsyncMock()
    backend.connect = AsyncMock()
    backend._db = MagicMock()
    return backend


def _mock_metadata():
    """Create a mock MetadataStore with the minimal interface needed by connect()."""
    meta = AsyncMock()
    meta.connect = AsyncMock()
    meta.migrate_from_config = AsyncMock(return_value=0)
    meta._set_shared_db = MagicMock()
    return meta


async def _run_connect(config: SVRConfig, env: dict[str, str] | None = None):
    """Create an SVRClient and run connect() with all externals mocked.

    Returns the client instance after connect() completes.
    """
    if env is None:
        env = {}

    mock_backend = _mock_backend()
    mock_meta = _mock_metadata()
    mock_embedder = MagicMock()

    with patch("semantic_vector_router.client.create_backend", return_value=mock_backend):
        with patch(
            "semantic_vector_router.backends.metadata.MetadataStore",
            return_value=mock_meta,
        ):
            with patch.dict(os.environ, env, clear=True):
                client = SVRClient(config=config, auto_connect=False)
                with patch.object(
                    client, "_create_embedder", return_value=mock_embedder
                ):
                    await client.connect()

    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRerankingGracefulDegradation:
    """Tests for the reranking graceful degradation path in connect()."""

    @pytest.mark.asyncio
    async def test_reranking_enabled_key_present_creates_reranker(self):
        """When reranking is enabled and the API key is set, _create_reranker()
        is called and self._reranker is assigned."""
        config = _make_config(reranking_enabled=True, api_key_env="VOYAGE_API_KEY")
        mock_backend = _mock_backend()
        mock_meta = _mock_metadata()
        sentinel_reranker = MagicMock()

        with patch("semantic_vector_router.client.create_backend", return_value=mock_backend):
            with patch(
                "semantic_vector_router.backends.metadata.MetadataStore",
                return_value=mock_meta,
            ):
                with patch.dict(os.environ, {"VOYAGE_API_KEY": "vk-test-key-123"}, clear=True):
                    client = SVRClient(config=config, auto_connect=False)
                    with patch.object(
                        client, "_create_embedder", return_value=MagicMock()
                    ):
                        with patch.object(
                            client, "_create_reranker", return_value=sentinel_reranker
                        ) as mock_create:
                            await client.connect()

        mock_create.assert_called_once()
        assert client._reranker is sentinel_reranker

    @pytest.mark.asyncio
    async def test_reranking_enabled_key_missing_reranker_is_none(self):
        """When reranking is enabled but the API key env var is not set,
        self._reranker should be None (graceful degradation)."""
        config = _make_config(reranking_enabled=True, api_key_env="VOYAGE_API_KEY")
        # Do not set VOYAGE_API_KEY in the environment
        client = await _run_connect(config, env={})

        assert client._reranker is None

    @pytest.mark.asyncio
    async def test_reranking_enabled_api_key_env_is_none(self):
        """When reranking is enabled but api_key_env is None,
        self._reranker should be None."""
        config = _make_config(reranking_enabled=True, api_key_env=None)
        client = await _run_connect(config, env={})

        assert client._reranker is None

    @pytest.mark.asyncio
    async def test_reranking_enabled_api_key_env_is_empty_string(self):
        """When reranking is enabled but api_key_env is an empty string,
        self._reranker should be None."""
        config = _make_config(reranking_enabled=True, api_key_env="")
        client = await _run_connect(config, env={})

        assert client._reranker is None

    @pytest.mark.asyncio
    async def test_reranking_disabled_does_not_create_reranker(self):
        """When reranking is disabled, _create_reranker() must not be called,
        regardless of whether the API key is available."""
        config = _make_config(reranking_enabled=False, api_key_env="VOYAGE_API_KEY")
        mock_backend = _mock_backend()
        mock_meta = _mock_metadata()

        with patch("semantic_vector_router.client.create_backend", return_value=mock_backend):
            with patch(
                "semantic_vector_router.backends.metadata.MetadataStore",
                return_value=mock_meta,
            ):
                with patch.dict(os.environ, {}, clear=True):
                    client = SVRClient(config=config, auto_connect=False)
                    with patch.object(client, "_create_embedder", return_value=MagicMock()):
                        with patch.object(client, "_create_reranker") as mock_create:
                            await client.connect()

        mock_create.assert_not_called()
        # _reranker stays at its __init__ default of None
        assert client._reranker is None

    @pytest.mark.asyncio
    async def test_reranking_disabled_with_key_present_does_not_create_reranker(self):
        """Even when the API key IS present, reranking.enabled=False means
        _create_reranker() must not be called."""
        config = _make_config(reranking_enabled=False, api_key_env="VOYAGE_API_KEY")
        mock_backend = _mock_backend()
        mock_meta = _mock_metadata()

        with patch("semantic_vector_router.client.create_backend", return_value=mock_backend):
            with patch(
                "semantic_vector_router.backends.metadata.MetadataStore",
                return_value=mock_meta,
            ):
                with patch.dict(
                    os.environ, {"VOYAGE_API_KEY": "vk-real-key"}, clear=True
                ):
                    client = SVRClient(config=config, auto_connect=False)
                    with patch.object(client, "_create_embedder", return_value=MagicMock()):
                        with patch.object(client, "_create_reranker") as mock_create:
                            await client.connect()

        mock_create.assert_not_called()
        assert client._reranker is None

    @pytest.mark.asyncio
    async def test_connect_succeeds_with_reranking_disabled(self):
        """connect() should complete successfully when reranking is disabled."""
        config = _make_config(reranking_enabled=False)
        client = await _run_connect(config, env={})

        assert client._connected is True
        assert client._reranker is None
        assert client._resolver is not None
        assert client._merger is not None

    @pytest.mark.asyncio
    async def test_connect_succeeds_with_graceful_degradation(self):
        """connect() should complete successfully when reranking is enabled
        but the key is missing (graceful degradation)."""
        config = _make_config(reranking_enabled=True, api_key_env="VOYAGE_API_KEY")
        client = await _run_connect(config, env={})

        assert client._connected is True
        assert client._reranker is None
        # Routing components should still be initialized
        assert client._resolver is not None
        assert client._merger is not None

    @pytest.mark.asyncio
    async def test_logger_info_message_contains_env_var_name(self):
        """The info log message should mention the specific env var name
        so users know which variable to set."""
        config = _make_config(reranking_enabled=True, api_key_env="VOYAGE_API_KEY")

        with patch("semantic_vector_router.client.logger") as mock_logger:
            client = await _run_connect(config, env={})

        # Find the info call that contains the degradation message
        info_calls = [
            call for call in mock_logger.info.call_args_list
            if "Reranking disabled" in str(call)
        ]
        assert len(info_calls) == 1, (
            f"Expected exactly one degradation log call, got {len(info_calls)}"
        )
        # The log uses % formatting: logger.info("... %s ...", env_var)
        call_str = str(info_calls[0])
        assert "VOYAGE_API_KEY" in call_str
        assert "score-based merging" in call_str

    @pytest.mark.asyncio
    async def test_logger_message_contains_custom_env_var_name(self):
        """When a custom api_key_env name is configured, it should appear
        in the degradation log message."""
        config = _make_config(
            reranking_enabled=True, api_key_env="MY_CUSTOM_RERANK_KEY"
        )

        with patch("semantic_vector_router.client.logger") as mock_logger:
            client = await _run_connect(config, env={})

        info_calls = [
            call for call in mock_logger.info.call_args_list
            if "Reranking disabled" in str(call)
        ]
        assert len(info_calls) == 1
        assert "MY_CUSTOM_RERANK_KEY" in str(info_calls[0])

    @pytest.mark.asyncio
    async def test_search_works_after_graceful_degradation(self):
        """After graceful degradation (reranker=None), search should still
        work correctly — it skips the reranking step and uses score-based
        merging only."""
        config = _make_config(reranking_enabled=True, api_key_env="VOYAGE_API_KEY")

        electronics_partition = PartitionInfo(
            name="electronics",
            view_name="svr_partition_electronics",
            index_name="svr_idx_electronics",
            filter_value="electronics",
            document_count=100,
            status=PartitionStatus.ACTIVE,
        )

        raw_results = [
            {
                "_id": "doc1",
                "name": "Headphones",
                "_svr_score": 0.95,
                "_svr_partition": "electronics",
            },
            {
                "_id": "doc2",
                "name": "Speaker",
                "_svr_score": 0.88,
                "_svr_partition": "electronics",
            },
        ]

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        client = await _run_connect(config, env={})

        # Verify degradation happened
        assert client._reranker is None
        assert client._connected is True

        # Replace internals for search
        client._embedder = mock_embedder
        client._backend.search_partitions = AsyncMock(return_value=raw_results)
        client._resolver = AsyncMock()
        client._resolver.resolve = AsyncMock(return_value=[electronics_partition])

        result = await client.search(
            query="headphones",
            partitions=["electronics"],
            limit=5,
        )

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 2
        # Should not be marked as reranked (single partition, no reranker)
        assert result.reranked is False

    @pytest.mark.asyncio
    async def test_no_log_message_when_reranking_disabled(self):
        """When reranking is entirely disabled (enabled=False), there should
        be no 'Reranking disabled' degradation log message — the code path
        that logs it is never entered."""
        config = _make_config(reranking_enabled=False)

        with patch("semantic_vector_router.client.logger") as mock_logger:
            client = await _run_connect(config, env={})

        degradation_calls = [
            call for call in mock_logger.info.call_args_list
            if "Reranking disabled" in str(call)
        ]
        assert len(degradation_calls) == 0

    @pytest.mark.asyncio
    async def test_reranker_created_when_key_in_env(self):
        """Verify _create_reranker is called exactly once when the key exists,
        and the return value is assigned to self._reranker."""
        config = _make_config(reranking_enabled=True, api_key_env="VOYAGE_API_KEY")
        mock_backend = _mock_backend()
        mock_meta = _mock_metadata()
        fake_reranker = MagicMock(spec=["rerank", "rerank_hits"])

        with patch("semantic_vector_router.client.create_backend", return_value=mock_backend):
            with patch(
                "semantic_vector_router.backends.metadata.MetadataStore",
                return_value=mock_meta,
            ):
                with patch.dict(
                    os.environ, {"VOYAGE_API_KEY": "vk-abc123"}, clear=True
                ):
                    client = SVRClient(config=config, auto_connect=False)
                    with patch.object(client, "_create_embedder", return_value=MagicMock()):
                        with patch.object(
                            client, "_create_reranker", return_value=fake_reranker
                        ) as mock_create:
                            await client.connect()

        mock_create.assert_called_once()
        assert client._reranker is fake_reranker

    @pytest.mark.asyncio
    async def test_connect_idempotent_with_degraded_reranker(self):
        """Calling connect() twice should be idempotent — the second call
        returns immediately because self._connected is True."""
        config = _make_config(reranking_enabled=True, api_key_env="VOYAGE_API_KEY")
        client = await _run_connect(config, env={})

        # First connect completed; reranker is None due to missing key
        assert client._reranker is None
        assert client._connected is True

        # Second call should be a no-op (early return)
        await client.connect()
        assert client._connected is True
        assert client._reranker is None

    @pytest.mark.asyncio
    async def test_key_present_but_empty_value_degrades_gracefully(self):
        """When the env var exists but has an empty string value, os.environ.get()
        returns '' which is falsy, so reranking should degrParts Distributor gracefully."""
        config = _make_config(reranking_enabled=True, api_key_env="VOYAGE_API_KEY")
        client = await _run_connect(config, env={"VOYAGE_API_KEY": ""})

        assert client._reranker is None
