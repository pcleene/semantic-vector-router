"""Integration tests: CLI commands against real Atlas.

Tests CLI commands using Click CliRunner with real MongoDB Atlas backend.
Commands that need real data set up state programmatically first.

Requires: MONGODB_URI and VOYAGE_API_KEY environment variables.
Run with: .venv/bin/pytest tests/integration/test_cli_flow.py -v -s --timeout=300
"""

import asyncio
import json
import logging
import os
import tempfile

import pytest
from click.testing import CliRunner
from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from semantic_vector_router.cli import main as cli_main
from semantic_vector_router.client import SVRClient
from semantic_vector_router.config import save_config
from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    IngestConfig,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
    RerankingConfig,
    ResilienceConfig,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
    VectorStorageFormat,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_DB = "svr_integration_test"
CLI_COLLECTION = "products_cli"
DIMENSIONS = 512
CATEGORIES = ["electronics", "food", "outdoor"]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _make_cli_config(
    collection_name: str = CLI_COLLECTION,
    index_on: IndexLocation = IndexLocation.SOURCE,
) -> SVRConfig:
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database=TEST_DB,
            source_collection=collection_name,
        ),
        partitioning=PartitioningConfig(
            field="category",
            view_prefix="svr_int_cli_",
            index_name_prefix="svr_int_cli_idx_",
        ),
        vector_storage=VectorStorageConfig(
            index_on=index_on,
            storage_format=VectorStorageFormat.ARRAY,
        ),
        vector_search=VectorSearchConfig(
            embedding_field="embedding",
            dimensions=DIMENSIONS,
            similarity="cosine",
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.VOYAGE,
            model="voyage-3-lite",
            dimensions=DIMENSIONS,
            api_key_env="VOYAGE_API_KEY",
        ),
        reranking=RerankingConfig(enabled=False),
        ingestion=IngestConfig(
            text_fields=["description"],
            batch_size=50,
        ),
        resilience=ResilienceConfig(embedding_timeout_ms=30000),
    )


def _save_config_to_tmpfile(config: SVRConfig) -> str:
    """Save config to a temporary JSON file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="svr_cli_test_")
    os.close(fd)
    save_config(config, path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mongodb_uri() -> str:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set")
    return uri


@pytest.fixture(scope="module")
def voyage_api_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        pytest.skip("VOYAGE_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def cli_runner():
    return CliRunner(catch_exceptions=False)


@pytest.fixture(scope="module")
async def cli_env(mongodb_uri, voyage_api_key):
    """Set up CLI test environment: clean collection, seed data, create config files.

    Yields a dict with config paths and config objects.
    """
    # Clean slate
    raw_client = AsyncMongoClient(mongodb_uri)
    db = raw_client[TEST_DB]
    await db.drop_collection(CLI_COLLECTION)
    await db.drop_collection("svr_metadata")

    # Seed some data for partition scanning
    coll = db[CLI_COLLECTION]
    seed_docs = []
    for cat in CATEGORIES:
        for i in range(5):
            seed_docs.append({
                "name": f"{cat} item {i}",
                "description": f"A {cat} product number {i} with features",
                "category": cat,
                "price": 10.0 + i * 5,
            })
    await coll.insert_many(seed_docs)
    await raw_client.close()

    # Create bare config (no partitions registered)
    bare_config = _make_cli_config()
    bare_config_path = _save_config_to_tmpfile(bare_config)

    # Create config with partitions registered
    partitioned_config = _make_cli_config()
    for cat in CATEGORIES:
        partitioned_config.partitions.registry[cat] = PartitionInfo(
            name=cat,
            index_name="svr_vector_idx_source",
            filter_value=cat,
            document_count=5,
            status=PartitionStatus.ACTIVE,
            index_location=IndexLocation.SOURCE,
            search_collection=CLI_COLLECTION,
        )
    partitioned_config_path = _save_config_to_tmpfile(partitioned_config)

    env = {
        "bare_config_path": bare_config_path,
        "partitioned_config_path": partitioned_config_path,
        "bare_config": bare_config,
        "partitioned_config": partitioned_config,
    }

    yield env

    # Teardown
    for path_key in ["bare_config_path", "partitioned_config_path"]:
        try:
            os.unlink(env[path_key])
        except Exception:
            pass
    try:
        raw_client2 = AsyncMongoClient(mongodb_uri)
        await raw_client2[TEST_DB].drop_collection(CLI_COLLECTION)
        await raw_client2[TEST_DB].drop_collection("svr_metadata")
        await raw_client2.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCLIFlow:
    """CLI commands against real Atlas."""

    def test_config_show(self, cli_runner, cli_env):
        """'svr config show' displays configuration with the test database."""
        config_path = cli_env["bare_config_path"]
        result = cli_runner.invoke(cli_main, ["config", "show", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert TEST_DB in result.output
        assert CLI_COLLECTION in result.output

    def test_config_validate(self, cli_runner, cli_env):
        """'svr config validate' runs validation."""
        config_path = cli_env["bare_config_path"]
        result = cli_runner.invoke(cli_main, ["config", "validate", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        # Either "valid" or warnings are acceptable
        assert "valid" in result.output.lower() or "Warning" in result.output

    def test_config_path(self, cli_runner, cli_env):
        """'svr config path' shows the config file path."""
        config_path = cli_env["bare_config_path"]
        result = cli_runner.invoke(cli_main, ["config", "path", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        # Output should contain the temp file path or at least the prefix
        assert "svr_cli_test_" in result.output or config_path in result.output

    def test_cache_stats(self, cli_runner, cli_env):
        """'svr cache stats' shows cache statistics."""
        config_path = cli_env["bare_config_path"]
        result = cli_runner.invoke(cli_main, ["cache", "stats", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "Embedding Cache" in result.output
        assert "Entries" in result.output
        assert "Hit rate" in result.output

    def test_partitions_list_empty(self, cli_runner, cli_env):
        """'svr partitions list' with no partitions shows helpful message."""
        config_path = cli_env["bare_config_path"]
        result = cli_runner.invoke(cli_main, ["partitions", "list", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "No partitions" in result.output

    def test_partitions_list_with_data(self, cli_runner, cli_env):
        """'svr partitions list' with registered partitions shows table."""
        config_path = cli_env["partitioned_config_path"]
        result = cli_runner.invoke(cli_main, ["partitions", "list", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        for cat in CATEGORIES:
            assert cat in result.output
        # Should show document count
        assert "5" in result.output

    def test_partitions_status_single(self, cli_runner, cli_env):
        """'svr partitions status <name>' shows details for one partition."""
        config_path = cli_env["partitioned_config_path"]
        result = cli_runner.invoke(
            cli_main, ["partitions", "status", "electronics", "-c", config_path]
        )
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "electronics" in result.output
        assert "active" in result.output.lower() or "ACTIVE" in result.output

    def test_partitions_status_all(self, cli_runner, cli_env):
        """'svr partitions status' shows status for all partitions."""
        config_path = cli_env["partitioned_config_path"]
        result = cli_runner.invoke(cli_main, ["partitions", "status", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        for cat in CATEGORIES:
            assert cat in result.output

    def test_partitions_scan(self, cli_runner, cli_env):
        """'svr partitions scan' finds partition values from seeded data."""
        config_path = cli_env["bare_config_path"]
        result = cli_runner.invoke(cli_main, ["partitions", "scan", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        # Should find all categories since no partitions are registered
        for cat in CATEGORIES:
            assert cat in result.output

    def test_cli_help_main(self, cli_runner, cli_env):
        """Main help shows all top-level commands."""
        result = cli_runner.invoke(cli_main, ["--help"])
        assert result.exit_code == 0
        for cmd in [
            "partitions", "search", "analyze", "config",
            "index", "watch", "split", "init", "cache", "ingest",
        ]:
            assert cmd in result.output

    def test_cli_help_ingest(self, cli_runner, cli_env):
        """'svr ingest --help' shows ingest command options."""
        result = cli_runner.invoke(cli_main, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--partition" in result.output
        assert "--mode" in result.output
        assert "--config" in result.output or "-c" in result.output

    def test_cli_ingest_json_file(self, cli_runner, cli_env, voyage_api_key):
        """'svr ingest' reads a JSON file and ingests documents."""
        config_path = cli_env["partitioned_config_path"]

        # Create a temp JSON file with documents
        docs = [
            {
                "name": "CLI Test Widget",
                "description": "A widget specifically created for CLI ingest testing",
                "category": "electronics",
                "price": 15.99,
            },
            {
                "name": "CLI Test Gadget",
                "description": "A gadget specifically created for CLI ingest testing",
                "category": "electronics",
                "price": 25.99,
            },
        ]
        fd, json_path = tempfile.mkstemp(suffix=".json", prefix="svr_cli_ingest_")
        os.close(fd)
        with open(json_path, "w") as f:
            json.dump(docs, f)

        try:
            result = cli_runner.invoke(
                cli_main,
                ["ingest", json_path, "-c", config_path, "--partition", "electronics"],
            )
            assert result.exit_code == 0, (
                f"Exit code {result.exit_code}: {result.output}"
            )
            assert "Ingestion complete" in result.output
            assert "Inserted" in result.output
        finally:
            os.unlink(json_path)

    def test_cli_ingest_jsonl_file(self, cli_runner, cli_env, voyage_api_key):
        """'svr ingest' reads a JSONL file and ingests documents."""
        config_path = cli_env["partitioned_config_path"]

        # Create a temp JSONL file
        lines = [
            json.dumps({
                "name": "JSONL Item 1",
                "description": "First item from JSONL format for CLI ingest test",
                "category": "food",
                "price": 8.99,
            }),
            json.dumps({
                "name": "JSONL Item 2",
                "description": "Second item from JSONL format for CLI ingest test",
                "category": "food",
                "price": 12.99,
            }),
        ]
        fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl", prefix="svr_cli_ingest_")
        os.close(fd)
        with open(jsonl_path, "w") as f:
            f.write("\n".join(lines))

        try:
            result = cli_runner.invoke(
                cli_main,
                ["ingest", jsonl_path, "-c", config_path, "--partition", "food"],
            )
            assert result.exit_code == 0, (
                f"Exit code {result.exit_code}: {result.output}"
            )
            assert "Ingestion complete" in result.output
        finally:
            os.unlink(jsonl_path)

    def test_cli_analyze(self, cli_runner, cli_env):
        """'svr analyze' shows field analysis from real collection."""
        config_path = cli_env["bare_config_path"]
        result = cli_runner.invoke(cli_main, ["analyze", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        # Should find 'category' field among analyzed fields
        assert "category" in result.output

    def test_cli_index_status_no_partitions(self, cli_runner, cli_env):
        """'svr index status' with no partitions shows message."""
        config_path = cli_env["bare_config_path"]
        result = cli_runner.invoke(cli_main, ["index", "status", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "No partitions" in result.output

    def test_cli_index_status_with_partitions(self, cli_runner, cli_env):
        """'svr index status' with registered partitions shows index info."""
        config_path = cli_env["partitioned_config_path"]
        result = cli_runner.invoke(cli_main, ["index", "status", "-c", config_path])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        # Should reference partition names or show an error for missing indexes
        output_lower = result.output.lower()
        assert "electronics" in output_lower or "error" in output_lower

    def test_cli_version(self, cli_runner, cli_env):
        """'svr --version' shows version."""
        result = cli_runner.invoke(cli_main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output or "svr" in result.output.lower()

    def test_cli_each_command_help(self, cli_runner, cli_env):
        """Every command and subgroup has working --help."""
        commands = [
            ["partitions", "--help"],
            ["search", "--help"],
            ["analyze", "--help"],
            ["config", "--help"],
            ["index", "--help"],
            ["watch", "--help"],
            ["split", "--help"],
            ["init", "--help"],
            ["cache", "--help"],
            ["ingest", "--help"],
            ["monitor", "--help"],
            ["repartition", "--help"],
        ]
        for cmd in commands:
            result = cli_runner.invoke(cli_main, cmd)
            assert result.exit_code == 0, (
                f"Help failed for: {' '.join(cmd)} - {result.output}"
            )
