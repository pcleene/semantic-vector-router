"""Tests for CLI ingest command."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from semantic_vector_router.cli.ingest import ingest_command, _load_documents
from semantic_vector_router.models import IngestMode


@pytest.fixture
def runner():
    return CliRunner()


def _make_mock_config():
    """Create a mock SVRConfig with ingestion settings."""
    mock_config = MagicMock()
    mock_config.ingestion.batch_size = 100
    mock_config.ingestion.mode = IngestMode.INSERT
    return mock_config


def _make_mock_result(inserted=2, failed=0, errors=None, elapsed_ms=100.0,
                      embed_ms=50.0, write_ms=50.0):
    """Create a mock IngestResult."""
    mock_result = MagicMock()
    mock_result.inserted = inserted
    mock_result.failed = failed
    mock_result.errors = errors or []
    mock_result.elapsed_ms = elapsed_ms
    mock_result.embed_ms = embed_ms
    mock_result.write_ms = write_ms
    return mock_result


def _make_mock_client(result=None):
    """Create a mock SVRClient with async methods."""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.ingest = AsyncMock(return_value=result or _make_mock_result())
    return mock_client


class TestLoadDocuments:
    """Tests for the _load_documents helper function."""

    def test_load_json_array(self, tmp_path):
        """Reads a .json file containing an array of objects."""
        docs = [{"title": "Doc 1"}, {"title": "Doc 2"}, {"title": "Doc 3"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        result = _load_documents(str(json_file))

        assert result == docs
        assert len(result) == 3

    def test_load_json_single(self, tmp_path):
        """Reads a single JSON object and wraps it in a list."""
        doc = {"title": "Single Doc", "body": "Content here"}
        json_file = tmp_path / "single.json"
        json_file.write_text(json.dumps(doc))

        result = _load_documents(str(json_file))

        assert result == [doc]
        assert len(result) == 1

    def test_load_jsonl(self, tmp_path):
        """Reads a .jsonl file with one JSON object per line."""
        docs = [
            {"title": "Line 1", "id": 1},
            {"title": "Line 2", "id": 2},
            {"title": "Line 3", "id": 3},
        ]
        jsonl_content = "\n".join(json.dumps(d) for d in docs)
        jsonl_file = tmp_path / "docs.jsonl"
        jsonl_file.write_text(jsonl_content)

        result = _load_documents(str(jsonl_file))

        assert result == docs
        assert len(result) == 3

    def test_load_stdin(self):
        """Reads documents from stdin when file_path is '-'."""
        docs = [{"title": "Stdin Doc 1"}, {"title": "Stdin Doc 2"}]
        stdin_content = json.dumps(docs)

        with patch("semantic_vector_router.cli.ingest.sys") as mock_sys:
            mock_sys.stdin.read.return_value = stdin_content
            result = _load_documents("-")

        assert result == docs
        assert len(result) == 2

    def test_load_file_not_found(self):
        """Non-existent file raises ClickException."""
        with pytest.raises(Exception) as exc_info:
            _load_documents("/nonexistent/path/docs.json")

        assert "File not found" in str(exc_info.value)

    def test_load_invalid_json(self, tmp_path):
        """Malformed JSON raises ClickException."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json at all")

        with pytest.raises(Exception) as exc_info:
            _load_documents(str(bad_file))

        assert "Invalid JSON" in str(exc_info.value)

    def test_load_empty_file(self, tmp_path):
        """Empty file raises ClickException."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")

        with pytest.raises(Exception) as exc_info:
            _load_documents(str(empty_file))

        assert "Empty input" in str(exc_info.value)

    def test_load_empty_whitespace_file(self, tmp_path):
        """File with only whitespace raises ClickException."""
        ws_file = tmp_path / "whitespace.json"
        ws_file.write_text("   \n  \n  ")

        with pytest.raises(Exception) as exc_info:
            _load_documents(str(ws_file))

        assert "Empty input" in str(exc_info.value)

    def test_load_jsonl_with_blank_lines(self, tmp_path):
        """JSONL with blank lines between objects should skip blanks."""
        docs = [{"a": 1}, {"a": 2}]
        content = json.dumps(docs[0]) + "\n\n" + json.dumps(docs[1]) + "\n"
        # This will parse as JSONL since the whole thing isn't valid JSON
        jsonl_file = tmp_path / "blanks.jsonl"
        jsonl_file.write_text(content)

        result = _load_documents(str(jsonl_file))
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"a": 2}

    def test_load_jsonl_non_dict_line(self, tmp_path):
        """JSONL line containing a non-dict (e.g. array) raises ClickException."""
        content = '{"a": 1}\n[1, 2, 3]\n'
        jsonl_file = tmp_path / "mixed.jsonl"
        jsonl_file.write_text(content)

        with pytest.raises(Exception) as exc_info:
            _load_documents(str(jsonl_file))

        assert "Expected JSON object" in str(exc_info.value)
        assert "Line 2" in str(exc_info.value)


class TestIngestCommand:
    """Tests for the ingest CLI command."""

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_json_file(self, mock_load_config, MockSVRClient, tmp_path, runner):
        """Reads .json file and calls client.ingest()."""
        docs = [{"title": "Doc 1"}, {"title": "Doc 2"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=2)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        assert "Ingestion complete" in result.output
        assert "2" in result.output  # inserted count
        mock_client.ingest.assert_called_once()
        mock_client.connect.assert_awaited_once()
        mock_client.disconnect.assert_awaited_once()

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_jsonl_file(self, mock_load_config, MockSVRClient, tmp_path, runner):
        """Reads .jsonl file and calls client.ingest()."""
        docs = [{"id": 1, "text": "Hello"}, {"id": 2, "text": "World"}]
        jsonl_file = tmp_path / "docs.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(d) for d in docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=2)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(jsonl_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        assert "Ingestion complete" in result.output
        mock_client.ingest.assert_called_once()
        # Verify the documents passed to ingest
        call_kwargs = mock_client.ingest.call_args
        assert len(call_kwargs.kwargs["documents"]) == 2

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_with_partition(self, mock_load_config, MockSVRClient, tmp_path, runner):
        """--partition flag is passed to client.ingest()."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(
            ingest_command,
            [str(json_file), "-c", "config.json", "--partition", "electronics"],
        )

        assert result.exit_code == 0, f"Output: {result.output}"
        mock_client.ingest.assert_called_once()
        call_kwargs = mock_client.ingest.call_args
        assert call_kwargs.kwargs["partition"] == "electronics"

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_with_mode(self, mock_load_config, MockSVRClient, tmp_path, runner):
        """--mode upsert is correctly parsed and passed to client.ingest()."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(
            ingest_command,
            [str(json_file), "-c", "config.json", "--mode", "upsert"],
        )

        assert result.exit_code == 0, f"Output: {result.output}"
        mock_client.ingest.assert_called_once()
        call_kwargs = mock_client.ingest.call_args
        assert call_kwargs.kwargs["mode"] == IngestMode.UPSERT

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_with_batch_size(self, mock_load_config, MockSVRClient, tmp_path, runner):
        """--batch-size overrides config.ingestion.batch_size."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(
            ingest_command,
            [str(json_file), "-c", "config.json", "--batch-size", "50"],
        )

        assert result.exit_code == 0, f"Output: {result.output}"
        # Verify the config's batch_size was overridden
        assert mock_config.ingestion.batch_size == 50

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_result_summary(self, mock_load_config, MockSVRClient, tmp_path, runner):
        """Output shows inserted/failed counts and timing info."""
        docs = [{"title": f"Doc {i}"} for i in range(5)]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(
            inserted=3,
            failed=2,
            errors=[(2, "Duplicate key"), (4, "Validation error")],
            elapsed_ms=250.0,
            embed_ms=150.0,
            write_ms=100.0,
        )
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        assert "Ingestion complete" in result.output
        assert "3" in result.output  # Inserted count
        assert "2" in result.output  # Failed count
        assert "Duplicate key" in result.output
        assert "Validation error" in result.output
        assert "250ms" in result.output  # Total time
        assert "150ms" in result.output  # Embedding time
        assert "100ms" in result.output  # Writing time

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_result_summary_truncates_errors(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """When more than 10 errors, output shows first 10 and a count of remaining."""
        docs = [{"title": f"Doc {i}"} for i in range(15)]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        errors = [(i, f"Error for doc {i}") for i in range(12)]
        mock_result = _make_mock_result(
            inserted=3,
            failed=12,
            errors=errors,
            elapsed_ms=500.0,
            embed_ms=300.0,
            write_ms=200.0,
        )
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        # First 10 errors should be displayed
        assert "Error for doc 0" in result.output
        assert "Error for doc 9" in result.output
        # The truncation message
        assert "2 more errors" in result.output

    def test_ingest_file_not_found(self, runner):
        """Non-existent file produces an error message."""
        result = runner.invoke(
            ingest_command,
            ["/nonexistent/file.json", "-c", "config.json"],
        )

        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_ingest_invalid_json(self, tmp_path, runner):
        """Malformed JSON file produces an error message."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json!!")

        result = runner.invoke(
            ingest_command,
            [str(bad_file), "-c", "config.json"],
        )

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_ingest_empty_file(self, tmp_path, runner):
        """Empty file produces an error message."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")

        result = runner.invoke(
            ingest_command,
            [str(empty_file), "-c", "config.json"],
        )

        assert result.exit_code != 0
        assert "Empty input" in result.output

    def test_ingest_missing_config_option(self, tmp_path, runner):
        """Missing required -c/--config option produces an error."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        result = runner.invoke(ingest_command, [str(json_file)])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_default_mode_is_none(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """When --mode is not specified, mode=None is passed to client.ingest()."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        call_kwargs = mock_client.ingest.call_args
        assert call_kwargs.kwargs["mode"] is None

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_mode_insert(self, mock_load_config, MockSVRClient, tmp_path, runner):
        """--mode insert is correctly parsed as IngestMode.INSERT."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(
            ingest_command,
            [str(json_file), "-c", "config.json", "--mode", "insert"],
        )

        assert result.exit_code == 0, f"Output: {result.output}"
        call_kwargs = mock_client.ingest.call_args
        assert call_kwargs.kwargs["mode"] == IngestMode.INSERT

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_client_created_with_auto_connect_false(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """SVRClient is created with auto_connect=False."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        MockSVRClient.assert_called_once_with(config=mock_config, auto_connect=False)

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_disconnect_called_on_success(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """client.disconnect() is called even on successful completion."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        mock_client.disconnect.assert_awaited_once()

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_disconnect_called_on_error(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """client.disconnect() is called even when ingest raises an exception."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.ingest = AsyncMock(side_effect=RuntimeError("Ingest failed"))
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        # Command fails but disconnect should still be called
        assert result.exit_code != 0
        mock_client.disconnect.assert_awaited_once()

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_loads_document_count(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """Output shows the number of loaded documents."""
        docs = [{"title": f"Doc {i}"} for i in range(7)]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=7)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        assert "7" in result.output  # "Loaded 7 documents"

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_no_failed_section_when_zero_failures(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """When failed=0, the 'Failed:' line should not appear."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1, failed=0)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        assert "Failed:" not in result.output

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_single_json_object(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """A single JSON object (not array) is treated as one document."""
        doc = {"title": "Solo Doc", "body": "Content"}
        json_file = tmp_path / "single.json"
        json_file.write_text(json.dumps(doc))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        call_kwargs = mock_client.ingest.call_args
        assert len(call_kwargs.kwargs["documents"]) == 1
        assert call_kwargs.kwargs["documents"][0]["title"] == "Solo Doc"

    @patch("semantic_vector_router.cli.ingest.SVRClient")
    @patch("semantic_vector_router.cli.ingest.load_config")
    def test_ingest_progress_callback_passed(
        self, mock_load_config, MockSVRClient, tmp_path, runner
    ):
        """A progress_callback is passed to client.ingest()."""
        docs = [{"title": "Doc 1"}]
        json_file = tmp_path / "docs.json"
        json_file.write_text(json.dumps(docs))

        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config

        mock_result = _make_mock_result(inserted=1)
        mock_client = _make_mock_client(result=mock_result)
        MockSVRClient.return_value = mock_client

        result = runner.invoke(ingest_command, [str(json_file), "-c", "config.json"])

        assert result.exit_code == 0, f"Output: {result.output}"
        call_kwargs = mock_client.ingest.call_args
        assert "progress_callback" in call_kwargs.kwargs
        assert callable(call_kwargs.kwargs["progress_callback"])
