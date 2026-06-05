"""Unit tests for embedding providers.

Tests VoyageEmbedder, OpenAIEmbedder, and CohereEmbedder with all HTTP calls mocked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.embedders.voyage import VoyageEmbedder
from semantic_vector_router.embedders.openai import OpenAIEmbedder
from semantic_vector_router.embedders.cohere import CohereEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_httpx_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = ""
    return response


def _mock_async_client(response: MagicMock) -> AsyncMock:
    """Create a mock httpx.AsyncClient that returns *response* on POST."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=response)
    return client


# ===========================================================================
# VoyageEmbedder
# ===========================================================================


class TestVoyageEmbedderInit:
    """Tests for VoyageEmbedder constructor and properties."""

    def test_voyage_embedder_init(self):
        """Constructor sets model_name, dimensions, and is_voyage_4 correctly."""
        embedder = VoyageEmbedder(
            model="voyage-4-large",
            api_key="<redacted>",
            input_type="query",
        )
        assert embedder.model_name == "voyage-4-large"
        assert embedder.dimensions == 1024
        assert embedder.is_voyage_4 is True

    def test_voyage_embedder_init_legacy_model(self):
        """Legacy model is not Voyage 4."""
        embedder = VoyageEmbedder(
            model="voyage-3-large",
            api_key="<redacted>",
        )
        assert embedder.model_name == "voyage-3-large"
        assert embedder.dimensions == 1024
        assert embedder.is_voyage_4 is False

    def test_voyage_embedder_custom_dimensions(self):
        """output_dimension overrides default dimensions for Voyage 4."""
        embedder = VoyageEmbedder(
            model="voyage-4",
            api_key="<redacted>",
            output_dimension=512,
        )
        assert embedder.dimensions == 512

    def test_voyage_embedder_all_v4_models_recognised(self):
        """All four Voyage 4 model names are recognised."""
        for model in ("voyage-4-large", "voyage-4", "voyage-4-lite", "voyage-4-nano"):
            embedder = VoyageEmbedder(model=model, api_key="k")
            assert embedder.is_voyage_4 is True


class TestVoyageAsymmetricConfig:
    """Tests for for_documents() / for_queries() asymmetric helpers."""

    def test_for_documents_returns_document_input_type(self):
        """for_documents() returns a new embedder with input_type='document'."""
        query_embedder = VoyageEmbedder(
            model="voyage-4-lite",
            api_key="<redacted>",
            input_type="query",
        )
        doc_embedder = query_embedder.for_documents()

        assert doc_embedder.input_type == "document"
        assert doc_embedder.model == "voyage-4-lite"
        assert doc_embedder.api_key == "test-key"

    def test_for_queries_returns_query_input_type(self):
        """for_queries() returns a new embedder with input_type='query'."""
        doc_embedder = VoyageEmbedder(
            model="voyage-4-large",
            api_key="<redacted>",
            input_type="document",
        )
        query_embedder = doc_embedder.for_queries()

        assert query_embedder.input_type == "query"
        assert query_embedder.model == "voyage-4-large"

    def test_for_documents_with_different_model(self):
        """for_documents() accepts a different Voyage 4 model."""
        query_embedder = VoyageEmbedder(
            model="voyage-4-lite",
            api_key="<redacted>",
            input_type="query",
        )
        doc_embedder = query_embedder.for_documents(model="voyage-4-large")

        assert doc_embedder.model == "voyage-4-large"
        assert doc_embedder.input_type == "document"


class TestVoyageDimensionQuantizationParams:
    """Tests for output_dimension and output_dtype in Voyage 4 payload."""

    @pytest.mark.asyncio
    async def test_dimension_and_dtype_in_payload(self):
        """output_dimension and output_dtype appear in the Voyage 4 POST payload."""
        embedder = VoyageEmbedder(
            model="voyage-4",
            api_key="<redacted>",
            output_dimension=512,
            output_dtype="int8",
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 512, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test text")

            # Inspect the payload sent to the API
            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]

            assert payload["output_dimension"] == 512
            assert payload["output_dtype"] == "int8"
            assert payload["model"] == "voyage-4"
            assert payload["input_type"] == "query"

    @pytest.mark.asyncio
    async def test_default_dtype_not_in_payload(self):
        """When output_dtype is 'float' (default), it should NOT appear in payload."""
        embedder = VoyageEmbedder(
            model="voyage-4",
            api_key="<redacted>",
            output_dtype="float",
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 1024, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]

            assert "output_dtype" not in payload


class TestVoyageEmbedBatch:
    """Tests for embed and embed_batch with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_embed_batch_extracts_embeddings(self):
        """embed_batch returns the correct vectors from the API response."""
        embedder = VoyageEmbedder(model="voyage-4", api_key="<redacted>")

        vec1 = [0.1] * 1024
        vec2 = [0.2] * 1024
        mock_response = _mock_httpx_response({
            "data": [
                {"embedding": vec1, "index": 0},
                {"embedding": vec2, "index": 1},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            results = await embedder.embed_batch(["text one", "text two"])

        assert len(results) == 2
        assert results[0] == vec1
        assert results[1] == vec2

    @pytest.mark.asyncio
    async def test_embed_single_delegates_to_batch(self):
        """embed() calls embed_batch with a single-element list."""
        embedder = VoyageEmbedder(model="voyage-4-lite", api_key="<redacted>")

        vec = [0.5] * 1024
        mock_response = _mock_httpx_response({
            "data": [{"embedding": vec, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            result = await embedder.embed("hello world")

        assert result == vec

    @pytest.mark.asyncio
    async def test_embed_batch_calls_correct_url(self):
        """embed_batch posts to {base_url}/embeddings."""
        embedder = VoyageEmbedder(
            model="voyage-4",
            api_key="<redacted>",
            base_url="https://custom.api.com/v1",
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 1024, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed_batch(["test"])

            call_args = mock_client.post.call_args
            url = call_args.args[0] if call_args.args else call_args[0][0]
            assert url == "https://custom.api.com/v1/embeddings"

    @pytest.mark.asyncio
    async def test_embed_batch_empty_returns_empty(self):
        """embed_batch with empty list returns empty without calling API."""
        embedder = VoyageEmbedder(model="voyage-4", api_key="<redacted>")
        result = await embedder.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_auth_header(self):
        """embed_batch sends the correct Authorization header."""
        embedder = VoyageEmbedder(model="voyage-4", api_key="<redacted>")

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 1024, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers["Authorization"] == "Bearer secret-key-123"


class TestVoyageNonV4IgnoresDimension:
    """Non-Voyage-4 models should silently ignore output_dimension and output_dtype."""

    def test_non_v4_ignores_output_dimension(self):
        """output_dimension is reset to None for non-Voyage-4 models."""
        embedder = VoyageEmbedder(
            model="voyage-3-large",
            api_key="<redacted>",
            output_dimension=512,
        )
        assert embedder.output_dimension is None
        assert embedder.dimensions == 1024  # default for voyage-3-large

    def test_non_v4_ignores_output_dtype(self):
        """output_dtype is reset to 'float' for non-Voyage-4 models."""
        embedder = VoyageEmbedder(
            model="voyage-3",
            api_key="<redacted>",
            output_dtype="int8",
        )
        assert embedder.output_dtype == "float"

    @pytest.mark.asyncio
    async def test_non_v4_payload_omits_dimension_and_dtype(self):
        """Non-V4 models should not include output_dimension or output_dtype in payload."""
        embedder = VoyageEmbedder(
            model="voyage-3-large",
            api_key="<redacted>",
            output_dimension=512,
            output_dtype="int8",
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 1024, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]

            assert "output_dimension" not in payload
            assert "output_dtype" not in payload


# ===========================================================================
# OpenAIEmbedder
# ===========================================================================


class TestOpenAIEmbedderInit:
    """Tests for OpenAIEmbedder constructor and properties."""

    def test_openai_embedder_init(self):
        """Constructor sets model_name and default dimensions."""
        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key="<redacted>",
        )
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.dimensions == 1536

    def test_openai_embedder_large_model(self):
        """text-embedding-3-large defaults to 3072 dimensions."""
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            api_key="<redacted>",
        )
        assert embedder.dimensions == 3072

    def test_openai_embedder_custom_dimensions(self):
        """Custom dimensions override the default."""
        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key="<redacted>",
            dimensions=256,
        )
        assert embedder.dimensions == 256

    def test_openai_embedder_unknown_model_defaults(self):
        """Unknown model falls back to 1536 dimensions."""
        embedder = OpenAIEmbedder(model="custom-model", api_key="<redacted>")
        assert embedder.dimensions == 1536


class TestOpenAIEmbedCall:
    """Tests for OpenAIEmbedder embed with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_embed_call_returns_vector(self):
        """embed() returns the embedding vector from the API."""
        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key="<redacted>",
        )

        vec = [0.1] * 1536
        mock_response = _mock_httpx_response({
            "data": [{"embedding": vec, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            result = await embedder.embed("test text")

        assert result == vec

    @pytest.mark.asyncio
    async def test_embed_call_payload_structure(self):
        """Verify the POST payload has the correct structure."""
        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key="<redacted>",
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 1536, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed_batch(["hello", "world"])

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]

            assert payload["model"] == "text-embedding-3-small"
            assert payload["input"] == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_embed_calls_correct_url(self):
        """embed_batch posts to {base_url}/embeddings."""
        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key="<redacted>",
            base_url="https://custom.openai.com/v1",
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 1536, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_args = mock_client.post.call_args
            url = call_args.args[0] if call_args.args else call_args[0][0]
            assert url == "https://custom.openai.com/v1/embeddings"

    @pytest.mark.asyncio
    async def test_embed_batch_empty_returns_empty(self):
        """embed_batch with empty list returns empty without calling API."""
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="k")
        result = await embedder.embed_batch([])
        assert result == []


class TestOpenAIDimensionPassing:
    """Tests for dimensions parameter in payload for text-embedding-3-* models."""

    @pytest.mark.asyncio
    async def test_dimensions_included_for_embedding_3_small(self):
        """dimensions parameter is included in payload for text-embedding-3-small."""
        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            api_key="<redacted>",
            dimensions=512,
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 512, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
            assert payload["dimensions"] == 512

    @pytest.mark.asyncio
    async def test_dimensions_included_for_embedding_3_large(self):
        """dimensions parameter is included in payload for text-embedding-3-large."""
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            api_key="<redacted>",
            dimensions=1024,
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 1024, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
            assert payload["dimensions"] == 1024

    @pytest.mark.asyncio
    async def test_dimensions_not_included_for_ada(self):
        """text-embedding-ada-002 does not support custom dimensions, so omit from payload."""
        embedder = OpenAIEmbedder(
            model="text-embedding-ada-002",
            api_key="<redacted>",
        )

        mock_response = _mock_httpx_response({
            "data": [{"embedding": [0.1] * 1536, "index": 0}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
            assert "dimensions" not in payload


# ===========================================================================
# CohereEmbedder
# ===========================================================================


class TestCohereEmbedderInit:
    """Tests for CohereEmbedder constructor and properties."""

    def test_cohere_embedder_init(self):
        """Constructor sets model_name and default dimensions."""
        embedder = CohereEmbedder(
            model="embed-english-v3.0",
            api_key="<redacted>",
        )
        assert embedder.model_name == "embed-english-v3.0"
        assert embedder.dimensions == 1024

    def test_cohere_embedder_light_model(self):
        """Light model has 384 dimensions."""
        embedder = CohereEmbedder(
            model="embed-english-light-v3.0",
            api_key="<redacted>",
        )
        assert embedder.dimensions == 384

    def test_cohere_embedder_default_input_type(self):
        """Default input_type is 'search_query'."""
        embedder = CohereEmbedder(model="embed-english-v3.0", api_key="k")
        assert embedder.input_type == "search_query"


class TestCohereInputTypeHandling:
    """Tests for input_type being passed correctly in the API payload."""

    @pytest.mark.asyncio
    async def test_input_type_in_payload(self):
        """input_type is included in the Cohere API payload."""
        embedder = CohereEmbedder(
            model="embed-english-v3.0",
            api_key="<redacted>",
            input_type="search_query",
        )

        mock_response = _mock_httpx_response({
            "embeddings": [[0.1] * 1024]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test text")

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]

            assert payload["input_type"] == "search_query"
            assert payload["model"] == "embed-english-v3.0"
            assert payload["texts"] == ["test text"]

    @pytest.mark.asyncio
    async def test_classification_input_type(self):
        """Classification input_type is passed through correctly."""
        embedder = CohereEmbedder(
            model="embed-english-v3.0",
            api_key="<redacted>",
            input_type="classification",
        )

        mock_response = _mock_httpx_response({
            "embeddings": [[0.1] * 1024]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
            assert payload["input_type"] == "classification"

    @pytest.mark.asyncio
    async def test_cohere_calls_correct_url(self):
        """Cohere embed_batch posts to {base_url}/embed (not /embeddings)."""
        embedder = CohereEmbedder(
            model="embed-english-v3.0",
            api_key="<redacted>",
        )

        mock_response = _mock_httpx_response({
            "embeddings": [[0.1] * 1024]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await embedder.embed("test")

            call_args = mock_client.post.call_args
            url = call_args.args[0] if call_args.args else call_args[0][0]
            assert url == "https://api.cohere.ai/v1/embed"


class TestCohereForDocuments:
    """Tests for for_documents() and for_queries() helpers."""

    def test_for_documents_returns_search_document(self):
        """for_documents() returns a new embedder with input_type='search_document'."""
        embedder = CohereEmbedder(
            model="embed-english-v3.0",
            api_key="<redacted>",
            input_type="search_query",
        )
        doc_embedder = embedder.for_documents()

        assert doc_embedder.input_type == "search_document"
        assert doc_embedder.model == "embed-english-v3.0"
        assert doc_embedder.api_key == "test-key"

    def test_for_queries_returns_search_query(self):
        """for_queries() returns a new embedder with input_type='search_query'."""
        embedder = CohereEmbedder(
            model="embed-english-v3.0",
            api_key="<redacted>",
            input_type="search_document",
        )
        query_embedder = embedder.for_queries()

        assert query_embedder.input_type == "search_query"

    def test_for_documents_preserves_truncate_setting(self):
        """for_documents() preserves the truncate setting."""
        embedder = CohereEmbedder(
            model="embed-english-v3.0",
            api_key="<redacted>",
            truncate="START",
        )
        doc_embedder = embedder.for_documents()

        assert doc_embedder.truncate == "START"

    @pytest.mark.asyncio
    async def test_embed_batch_empty_returns_empty(self):
        """embed_batch with empty list returns empty without calling API."""
        embedder = CohereEmbedder(model="embed-english-v3.0", api_key="k")
        result = await embedder.embed_batch([])
        assert result == []
