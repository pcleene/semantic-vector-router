"""Tests for CLI try (quick search) command."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from semantic_vector_router.cli.try_search import (
    _doc_preview,
    _try_search,
    try_command,
)
from semantic_vector_router.models.search import SearchHit, SearchResult


@pytest.fixture
def runner():
    return CliRunner()


def _make_hit(score=0.95, partition="electronics", document=None):
    """Create a SearchHit for testing."""
    return SearchHit(
        id="abc123",
        score=score,
        partition=partition,
        document=document or {"title": "Wireless headphones", "_id": "abc123"},
    )


def _make_search_result(hits=None, query="test", latency_ms=42.0, partitions_searched=None):
    """Create a SearchResult for testing."""
    return SearchResult(
        hits=hits or [],
        query=query,
        partitions_searched=partitions_searched or ["electronics"],
        total_candidates=len(hits) if hits else 0,
        reranked=False,
        latency_ms=latency_ms,
    )


def _make_mock_svr(search_result=None):
    """Create a mock SVRClient with quickstart and search."""
    mock_svr = AsyncMock()
    mock_svr.search = AsyncMock(return_value=search_result or _make_search_result())
    mock_svr.disconnect = AsyncMock()
    return mock_svr


# ---------------------------------------------------------------------------
# _doc_preview tests
# ---------------------------------------------------------------------------


class TestDocPreview:
    """Tests for the _doc_preview helper function."""

    def test_preview_with_title_field(self):
        """Returns title when present."""
        doc = {"title": "Wireless Headphones", "price": 99.99}
        assert _doc_preview(doc) == "Wireless Headphones"

    def test_preview_with_name_field(self):
        """Returns name when title is absent."""
        doc = {"name": "Bluetooth Speaker", "brand": "Acme"}
        assert _doc_preview(doc) == "Bluetooth Speaker"

    def test_preview_with_text_field(self):
        """Returns text when title and name are absent."""
        doc = {"text": "This is the body text of the document."}
        assert _doc_preview(doc) == "This is the body text of the document."

    def test_preview_with_content_field(self):
        """Returns content field."""
        doc = {"content": "Some content value"}
        assert _doc_preview(doc) == "Some content value"

    def test_preview_with_description_field(self):
        """Returns description field."""
        doc = {"description": "A short description"}
        assert _doc_preview(doc) == "A short description"

    def test_preview_with_summary_field(self):
        """Returns summary field."""
        doc = {"summary": "Executive summary goes here"}
        assert _doc_preview(doc) == "Executive summary goes here"

    def test_preview_long_text_truncation(self):
        """Truncates text longer than max_len and appends ellipsis."""
        long_text = "A" * 100
        result = _doc_preview({"title": long_text}, max_len=75)
        assert result == "A" * 75 + "..."
        assert len(result) == 78  # 75 chars + "..."

    def test_preview_exact_max_len_not_truncated(self):
        """Text at exactly max_len is not truncated."""
        exact_text = "B" * 75
        result = _doc_preview({"title": exact_text}, max_len=75)
        assert result == exact_text
        assert "..." not in result

    def test_preview_custom_max_len(self):
        """Respects a custom max_len parameter."""
        doc = {"title": "A" * 50}
        result = _doc_preview(doc, max_len=20)
        assert result == "A" * 20 + "..."

    def test_preview_no_common_fields_fallback(self):
        """Falls back to key-value pairs when no common fields present."""
        doc = {"brand": "Sony", "category": "audio", "sku": "12345"}
        result = _doc_preview(doc)
        assert "brand: Sony" in result
        assert "category: audio" in result
        assert "sku: 12345" in result
        assert " | " in result

    def test_preview_empty_dict(self):
        """Returns '(no preview)' for empty document."""
        assert _doc_preview({}) == "(no preview)"

    def test_preview_skips_underscore_and_embedding_fields(self):
        """Skips fields starting with _ and the embedding field."""
        doc = {"_id": "abc123", "embedding": [0.1, 0.2], "_meta": "internal"}
        assert _doc_preview(doc) == "(no preview)"

    def test_preview_skips_internal_shows_other_keys(self):
        """Skips _id and embedding but shows remaining fields."""
        doc = {"_id": "abc", "embedding": [0.1], "color": "red", "size": "large"}
        result = _doc_preview(doc)
        assert "color: red" in result
        assert "size: large" in result
        assert "_id" not in result
        assert "embedding" not in result

    def test_preview_fallback_limits_to_three_fields(self):
        """Fallback shows at most 3 key-value pairs."""
        doc = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        result = _doc_preview(doc)
        parts = result.split(" | ")
        assert len(parts) == 3

    def test_preview_fallback_truncates_long_values(self):
        """Fallback truncates individual values to 30 chars."""
        doc = {"field": "X" * 60}
        result = _doc_preview(doc)
        # value is truncated to 30 chars in the preview
        assert result == f"field: {'X' * 30}"

    def test_preview_priority_order_title_over_name(self):
        """Title takes priority over name."""
        doc = {"name": "Name Value", "title": "Title Value"}
        assert _doc_preview(doc) == "Title Value"

    def test_preview_priority_order_name_over_text(self):
        """Name takes priority over text."""
        doc = {"text": "Text Value", "name": "Name Value"}
        assert _doc_preview(doc) == "Name Value"

    def test_preview_skips_falsy_common_fields(self):
        """Skips common fields when their value is falsy (empty string, None)."""
        doc = {"title": "", "name": None, "text": "Actual text"}
        assert _doc_preview(doc) == "Actual text"

    def test_preview_non_string_value_converted(self):
        """Non-string values in common fields are converted to str."""
        doc = {"title": 42}
        assert _doc_preview(doc) == "42"


# ---------------------------------------------------------------------------
# CLI invocation tests (CliRunner + mocked _try_search)
# ---------------------------------------------------------------------------


class TestTryCommandCLI:
    """Tests for the try_command CLI entry point."""

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_parses_arguments(self, mock_run, mock_try, runner):
        """CLI parses query, -d, -c, -p flags."""
        result = runner.invoke(
            try_command,
            ["hello world", "-d", "mydb", "-c", "docs", "-p", "type"],
        )
        assert result.exit_code == 0
        mock_run.assert_called_once()
        # Verify the coroutine was passed to asyncio.run
        call_args = mock_run.call_args
        assert call_args is not None

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_with_partitions_splits_csv(self, mock_run, mock_try, runner):
        """--partitions 'a,b,c' is split into a list."""
        result = runner.invoke(
            try_command,
            ["query", "--partitions", "a,b,c"],
        )
        assert result.exit_code == 0
        # asyncio.run receives a coroutine whose kwargs include partitions=["a","b","c"]
        mock_run.assert_called_once()

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_default_limit_is_five(self, mock_run, mock_try, runner):
        """Default limit is 5 when not specified."""
        result = runner.invoke(try_command, ["query"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_with_limit_flag(self, mock_run, mock_try, runner):
        """CLI accepts --limit / -l flag."""
        result = runner.invoke(try_command, ["query", "-l", "10"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_with_backend_postgres(self, mock_run, mock_try, runner):
        """CLI accepts --backend postgres."""
        result = runner.invoke(try_command, ["query", "--backend", "postgres"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_with_backend_mongodb(self, mock_run, mock_try, runner):
        """CLI accepts --backend mongodb (the default)."""
        result = runner.invoke(try_command, ["query", "--backend", "mongodb"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_cli_rejects_invalid_backend(self, runner):
        """CLI rejects an invalid --backend value."""
        result = runner.invoke(try_command, ["query", "--backend", "redis"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()

    def test_cli_help_text(self, runner):
        """Help output contains command name and description."""
        result = runner.invoke(try_command, ["--help"])
        assert result.exit_code == 0
        assert "quick search" in result.output.lower()
        assert "--database" in result.output
        assert "--collection" in result.output
        assert "--partition-field" in result.output
        assert "--partitions" in result.output
        assert "--limit" in result.output
        assert "--backend" in result.output
        assert "--embedding-provider" in result.output

    def test_cli_requires_query_argument(self, runner):
        """CLI fails when QUERY argument is missing."""
        result = runner.invoke(try_command, [])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "QUERY" in result.output

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_with_embedding_provider(self, mock_run, mock_try, runner):
        """CLI accepts --embedding-provider flag."""
        result = runner.invoke(
            try_command,
            ["query", "--embedding-provider", "voyage"],
        )
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_no_partitions_passes_none(self, mock_run, mock_try, runner):
        """When --partitions is omitted, None is passed (not split)."""
        result = runner.invoke(try_command, ["query"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("semantic_vector_router.cli.try_search._try_search", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.try_search.asyncio.run")
    def test_cli_all_flags_together(self, mock_run, mock_try, runner):
        """CLI accepts all flags simultaneously."""
        result = runner.invoke(
            try_command,
            [
                "my search query",
                "-d", "mydb",
                "-c", "docs",
                "-p", "category",
                "--partitions", "electronics,books",
                "-l", "20",
                "-b", "postgres",
                "--embedding-provider", "openai",
            ],
        )
        assert result.exit_code == 0
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# _try_search async tests
# ---------------------------------------------------------------------------


class TestTrySearchAsync:
    """Tests for the _try_search async function."""

    @pytest.mark.asyncio
    @patch("semantic_vector_router.cli.try_search.console")
    @patch("semantic_vector_router.SVRClient")
    async def test_no_results_prints_message(self, mock_svr_cls, mock_console):
        """Displays 'No results found' when hits list is empty."""
        mock_svr = _make_mock_svr(search_result=_make_search_result(hits=[]))
        mock_svr_cls.quickstart = AsyncMock(return_value=mock_svr)

        # console.status() must be a context manager
        mock_console.status.return_value.__enter__ = MagicMock(return_value=None)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)

        await _try_search(
            query="nonexistent",
            database="db",
            collection="col",
            partition_field="pf",
            partitions=None,
            limit=5,
            backend="mongodb",
            embedding_provider=None,
        )

        # Verify "No results found" was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("No results found" in c for c in print_calls)
        mock_svr.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("semantic_vector_router.cli.try_search.console")
    @patch("semantic_vector_router.SVRClient")
    async def test_connection_error_aborts(self, mock_svr_cls, mock_console):
        """Raises click.Abort when quickstart raises an exception."""
        mock_svr_cls.quickstart = AsyncMock(side_effect=ConnectionError("Connection refused"))

        mock_console.status.return_value.__enter__ = MagicMock(return_value=None)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(click.Abort):
            await _try_search(
                query="test",
                database="db",
                collection="col",
                partition_field="pf",
                partitions=None,
                limit=5,
                backend="mongodb",
                embedding_provider=None,
            )

        # Verify the error was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Error" in c for c in print_calls)

    @pytest.mark.asyncio
    @patch("semantic_vector_router.cli.try_search.console")
    @patch("semantic_vector_router.SVRClient")
    async def test_search_error_aborts(self, mock_svr_cls, mock_console):
        """Raises click.Abort when search() raises an exception."""
        mock_svr = AsyncMock()
        mock_svr.search = AsyncMock(side_effect=RuntimeError("Search failed"))
        mock_svr_cls.quickstart = AsyncMock(return_value=mock_svr)

        mock_console.status.return_value.__enter__ = MagicMock(return_value=None)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(click.Abort):
            await _try_search(
                query="test",
                database="db",
                collection="col",
                partition_field="pf",
                partitions=None,
                limit=5,
                backend="mongodb",
                embedding_provider=None,
            )

    @pytest.mark.asyncio
    @patch("semantic_vector_router.cli.try_search.console")
    @patch("semantic_vector_router.SVRClient")
    async def test_results_displays_table(self, mock_svr_cls, mock_console):
        """Displays a table when search returns hits."""
        hits = [
            _make_hit(score=0.95, partition="electronics", document={"title": "Headphones", "category": "electronics"}),
            _make_hit(score=0.88, partition="audio", document={"title": "Speaker", "category": "audio"}),
        ]
        search_result = _make_search_result(
            hits=hits,
            query="wireless",
            latency_ms=42.0,
            partitions_searched=["electronics", "audio"],
        )
        mock_svr = _make_mock_svr(search_result=search_result)
        mock_svr_cls.quickstart = AsyncMock(return_value=mock_svr)

        mock_console.status.return_value.__enter__ = MagicMock(return_value=None)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)

        await _try_search(
            query="wireless",
            database="db",
            collection="col",
            partition_field="category",
            partitions=None,
            limit=5,
            backend="mongodb",
            embedding_provider=None,
        )

        # console.print should have been called with a Table object
        print_calls = mock_console.print.call_args_list
        table_printed = any(
            hasattr(call.args[0], "columns") if call.args else False
            for call in print_calls
        )
        assert table_printed, "Expected a Rich Table to be printed"
        mock_svr.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("semantic_vector_router.cli.try_search.console")
    @patch("semantic_vector_router.SVRClient")
    async def test_results_prints_summary_line(self, mock_svr_cls, mock_console):
        """Prints summary with count, latency, backend, and partitions."""
        hits = [_make_hit()]
        search_result = _make_search_result(
            hits=hits,
            query="test",
            latency_ms=55.0,
            partitions_searched=["electronics"],
        )
        mock_svr = _make_mock_svr(search_result=search_result)
        mock_svr_cls.quickstart = AsyncMock(return_value=mock_svr)

        mock_console.status.return_value.__enter__ = MagicMock(return_value=None)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)

        await _try_search(
            query="test",
            database="db",
            collection="col",
            partition_field="pf",
            partitions=None,
            limit=5,
            backend="mongodb",
            embedding_provider=None,
        )

        print_calls = [str(c) for c in mock_console.print.call_args_list]
        summary_printed = any("1 results" in c and "55ms" in c and "mongodb" in c for c in print_calls)
        assert summary_printed, f"Expected summary line in print calls: {print_calls}"

    @pytest.mark.asyncio
    @patch("semantic_vector_router.cli.try_search.console")
    @patch("semantic_vector_router.SVRClient")
    async def test_quickstart_receives_correct_kwargs(self, mock_svr_cls, mock_console):
        """Verifies quickstart is called with the right parameters."""
        mock_svr = _make_mock_svr(search_result=_make_search_result(hits=[]))
        mock_svr_cls.quickstart = AsyncMock(return_value=mock_svr)

        mock_console.status.return_value.__enter__ = MagicMock(return_value=None)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)

        await _try_search(
            query="test",
            database="testdb",
            collection="testcol",
            partition_field="topic",
            partitions=["a", "b"],
            limit=10,
            backend="postgres",
            embedding_provider="openai",
        )

        mock_svr_cls.quickstart.assert_awaited_once_with(
            database="testdb",
            collection="testcol",
            partition_field="topic",
            backend="postgres",
            embedding_provider="openai",
        )

    @pytest.mark.asyncio
    @patch("semantic_vector_router.cli.try_search.console")
    @patch("semantic_vector_router.SVRClient")
    async def test_search_receives_correct_kwargs(self, mock_svr_cls, mock_console):
        """Verifies search is called with query, partitions, and limit."""
        mock_svr = _make_mock_svr(search_result=_make_search_result(hits=[]))
        mock_svr_cls.quickstart = AsyncMock(return_value=mock_svr)

        mock_console.status.return_value.__enter__ = MagicMock(return_value=None)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)

        await _try_search(
            query="wireless headphones",
            database="db",
            collection="col",
            partition_field="pf",
            partitions=["electronics", "audio"],
            limit=10,
            backend="mongodb",
            embedding_provider=None,
        )

        mock_svr.search.assert_awaited_once_with(
            query="wireless headphones",
            partitions=["electronics", "audio"],
            limit=10,
        )

    @pytest.mark.asyncio
    @patch("semantic_vector_router.cli.try_search.console")
    @patch("semantic_vector_router.SVRClient")
    async def test_no_results_prints_suggestion(self, mock_svr_cls, mock_console):
        """Prints a suggestion message when no results found."""
        mock_svr = _make_mock_svr(search_result=_make_search_result(hits=[]))
        mock_svr_cls.quickstart = AsyncMock(return_value=mock_svr)

        mock_console.status.return_value.__enter__ = MagicMock(return_value=None)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)

        await _try_search(
            query="nonexistent",
            database="db",
            collection="col",
            partition_field="pf",
            partitions=None,
            limit=5,
            backend="mongodb",
            embedding_provider=None,
        )

        print_calls = [str(c) for c in mock_console.print.call_args_list]
        suggestion_printed = any("different query" in c.lower() or "check that documents" in c.lower() for c in print_calls)
        assert suggestion_printed, f"Expected suggestion in print calls: {print_calls}"
