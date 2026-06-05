"""Unit tests for reranking providers.

Tests VoyageReranker and CohereReranker with all HTTP calls mocked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.models import SearchHit
from semantic_vector_router.rerankers.voyage import VoyageReranker
from semantic_vector_router.rerankers.cohere import CohereReranker


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


def _make_hits() -> list[SearchHit]:
    """Create a standard set of test SearchHit objects."""
    return [
        SearchHit(id="1", score=0.9, partition="test", document={"text": "doc 1"}),
        SearchHit(id="2", score=0.8, partition="test", document={"text": "doc 2"}),
        SearchHit(id="3", score=0.7, partition="test", document={"text": "doc 3"}),
    ]


# ===========================================================================
# VoyageReranker
# ===========================================================================


class TestVoyageRerankerScores:
    """Tests for VoyageReranker.rerank() returning scores in original order."""

    @pytest.mark.asyncio
    async def test_rerank_scores_in_original_order(self):
        """Scores are returned in original document order, not relevance order."""
        reranker = VoyageReranker(model="rerank-2", api_key="<redacted>")

        # Voyage returns results sorted by relevance (highest first),
        # but rerank() should map them back to original order.
        mock_response = _mock_httpx_response({
            "data": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
                {"index": 1, "relevance_score": 0.50},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            scores = await reranker.rerank(
                query="test query",
                documents=["doc A", "doc B", "doc C"],
            )

        assert len(scores) == 3
        assert scores[0] == pytest.approx(0.85)  # index 0
        assert scores[1] == pytest.approx(0.50)  # index 1
        assert scores[2] == pytest.approx(0.95)  # index 2

    @pytest.mark.asyncio
    async def test_rerank_calls_correct_url(self):
        """rerank() posts to {base_url}/rerank."""
        reranker = VoyageReranker(
            model="rerank-2",
            api_key="<redacted>",
            base_url="https://api.voyageai.com/v1",
        )

        mock_response = _mock_httpx_response({
            "data": [{"index": 0, "relevance_score": 0.9}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await reranker.rerank("query", ["doc"])

            call_args = mock_client.post.call_args
            url = call_args.args[0] if call_args.args else call_args[0][0]
            assert url == "https://api.voyageai.com/v1/rerank"

    @pytest.mark.asyncio
    async def test_rerank_payload_structure(self):
        """Verify the POST payload includes model, query, documents, truncation."""
        reranker = VoyageReranker(model="rerank-2", api_key="<redacted>")

        mock_response = _mock_httpx_response({
            "data": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await reranker.rerank("my query", ["doc A", "doc B"])

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]

            assert payload["model"] == "rerank-2"
            assert payload["query"] == "my query"
            assert payload["documents"] == ["doc A", "doc B"]
            assert payload["truncation"] is True


class TestVoyageRerankerHits:
    """Tests for VoyageReranker.rerank_hits() with SearchHit objects."""

    @pytest.mark.asyncio
    async def test_rerank_hits_transforms_scores(self):
        """rerank_hits sets rerank_score on each SearchHit."""
        reranker = VoyageReranker(model="rerank-2", api_key="<redacted>")
        hits = _make_hits()

        mock_response = _mock_httpx_response({
            "data": [
                {"index": 0, "relevance_score": 0.6},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.3},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            result = await reranker.rerank_hits("query", hits)

        # All hits should have rerank_score set
        for hit in result:
            assert hit.rerank_score is not None

        # Should be sorted by rerank_score descending
        assert result[0].id == "2"  # highest rerank_score (0.9)
        assert result[0].rerank_score == pytest.approx(0.9)
        assert result[1].id == "1"  # 0.6
        assert result[2].id == "3"  # 0.3

    @pytest.mark.asyncio
    async def test_rerank_hits_respects_top_k(self):
        """rerank_hits limits results when top_k is provided."""
        reranker = VoyageReranker(model="rerank-2", api_key="<redacted>")
        hits = _make_hits()

        mock_response = _mock_httpx_response({
            "data": [
                {"index": 0, "relevance_score": 0.6},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.3},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            result = await reranker.rerank_hits("query", hits, top_k=2)

        assert len(result) == 2
        # Top 2 by rerank_score
        assert result[0].rerank_score == pytest.approx(0.9)
        assert result[1].rerank_score == pytest.approx(0.6)


class TestVoyageRerankerEmpty:
    """Tests for edge cases with empty inputs."""

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """rerank() with empty documents returns empty list without API call."""
        reranker = VoyageReranker(model="rerank-2", api_key="<redacted>")
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_hits_empty_hits(self):
        """rerank_hits() with empty hits returns empty list without API call."""
        reranker = VoyageReranker(model="rerank-2", api_key="<redacted>")
        result = await reranker.rerank_hits("query", [])
        assert result == []


# ===========================================================================
# CohereReranker
# ===========================================================================


class TestCohereRerankerScores:
    """Tests for CohereReranker.rerank() returning scores in original order."""

    @pytest.mark.asyncio
    async def test_rerank_scores_in_original_order(self):
        """Scores are returned in original document order."""
        reranker = CohereReranker(model="rerank-english-v3.0", api_key="<redacted>")

        # Cohere returns "results" (not "data"), sorted by relevance
        mock_response = _mock_httpx_response({
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.70},
                {"index": 2, "relevance_score": 0.40},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            scores = await reranker.rerank(
                query="test query",
                documents=["doc A", "doc B", "doc C"],
            )

        assert len(scores) == 3
        assert scores[0] == pytest.approx(0.70)  # index 0
        assert scores[1] == pytest.approx(0.95)  # index 1
        assert scores[2] == pytest.approx(0.40)  # index 2

    @pytest.mark.asyncio
    async def test_rerank_calls_correct_url(self):
        """rerank() posts to {base_url}/rerank."""
        reranker = CohereReranker(
            model="rerank-english-v3.0",
            api_key="<redacted>",
            base_url="https://api.cohere.ai/v1",
        )

        mock_response = _mock_httpx_response({
            "results": [{"index": 0, "relevance_score": 0.9}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await reranker.rerank("query", ["doc"])

            call_args = mock_client.post.call_args
            url = call_args.args[0] if call_args.args else call_args[0][0]
            assert url == "https://api.cohere.ai/v1/rerank"

    @pytest.mark.asyncio
    async def test_rerank_payload_structure(self):
        """Verify the POST payload includes model, query, documents, return_documents."""
        reranker = CohereReranker(model="rerank-english-v3.0", api_key="<redacted>")

        mock_response = _mock_httpx_response({
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await reranker.rerank("my query", ["doc A", "doc B"])

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]

            assert payload["model"] == "rerank-english-v3.0"
            assert payload["query"] == "my query"
            assert payload["documents"] == ["doc A", "doc B"]
            assert payload["return_documents"] is False

    @pytest.mark.asyncio
    async def test_rerank_top_k_uses_top_n(self):
        """Cohere uses 'top_n' in the payload when top_k is specified."""
        reranker = CohereReranker(model="rerank-english-v3.0", api_key="<redacted>")

        mock_response = _mock_httpx_response({
            "results": [{"index": 0, "relevance_score": 0.9}]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            await reranker.rerank("query", ["doc"], top_k=5)

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
            assert payload["top_n"] == 5


class TestCohereRerankerHits:
    """Tests for CohereReranker.rerank_hits() with SearchHit objects."""

    @pytest.mark.asyncio
    async def test_rerank_hits_transforms_scores(self):
        """rerank_hits sets rerank_score on each SearchHit and sorts by it."""
        reranker = CohereReranker(model="rerank-english-v3.0", api_key="<redacted>")
        hits = _make_hits()

        mock_response = _mock_httpx_response({
            "results": [
                {"index": 0, "relevance_score": 0.4},
                {"index": 1, "relevance_score": 0.8},
                {"index": 2, "relevance_score": 0.95},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            result = await reranker.rerank_hits("query", hits)

        # All hits should have rerank_score set
        for hit in result:
            assert hit.rerank_score is not None

        # Sorted by rerank_score descending
        assert result[0].id == "3"  # highest rerank_score (0.95)
        assert result[0].rerank_score == pytest.approx(0.95)
        assert result[1].id == "2"  # 0.8
        assert result[2].id == "1"  # 0.4

    @pytest.mark.asyncio
    async def test_rerank_hits_respects_top_k(self):
        """rerank_hits limits results when top_k is provided."""
        reranker = CohereReranker(model="rerank-english-v3.0", api_key="<redacted>")
        hits = _make_hits()

        mock_response = _mock_httpx_response({
            "results": [
                {"index": 0, "relevance_score": 0.4},
                {"index": 1, "relevance_score": 0.8},
                {"index": 2, "relevance_score": 0.95},
            ]
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = _mock_async_client(mock_response)
            mock_client_class.return_value = mock_client

            result = await reranker.rerank_hits("query", hits, top_k=1)

        assert len(result) == 1
        assert result[0].id == "3"
        assert result[0].rerank_score == pytest.approx(0.95)


class TestCohereRerankerEmpty:
    """Tests for edge cases with empty inputs."""

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """rerank() with empty documents returns empty list without API call."""
        reranker = CohereReranker(model="rerank-english-v3.0", api_key="<redacted>")
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_hits_empty_hits(self):
        """rerank_hits() with empty hits returns empty list without API call."""
        reranker = CohereReranker(model="rerank-english-v3.0", api_key="<redacted>")
        result = await reranker.rerank_hits("query", [])
        assert result == []
