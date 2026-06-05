"""Unit tests for CLI commands using Click's CliRunner."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from semantic_vector_router.cli import main
from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    LifecycleConfig,
    PartitionInfo,
    PartitioningConfig,
    PartitionStatus,
    RerankingConfig,
    RerankerProvider,
    SearchHit,
    SearchResult,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
)


@pytest.fixture
def runner():
    return CliRunner()


def _make_mock_config(with_partitions=True, index_location=IndexLocation.VIEWS):
    """Create a mock config for CLI tests."""
    config = SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="products",
        ),
        partitioning=PartitioningConfig(field="category"),
        vector_storage=VectorStorageConfig(index_on=index_location),
        vector_search=VectorSearchConfig(dimensions=1536),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key_env="OPENAI_API_KEY",
            dimensions=1536,
        ),
        reranking=RerankingConfig(
            enabled=True,
            provider=RerankerProvider.VOYAGE,
            model="rerank-2",
            api_key_env="VOYAGE_API_KEY",
        ),
    )
    if with_partitions:
        config.partitions.registry = {
            "electronics": PartitionInfo(
                name="electronics",
                view_name="svr_partition_electronics",
                index_name="svr_vector_idx_electronics",
                filter_value="electronics",
                document_count=15000,
                status=PartitionStatus.ACTIVE,
                index_location=index_location,
                search_collection="svr_partition_electronics" if index_location == IndexLocation.VIEWS else "products",
            ),
            "furniture": PartitionInfo(
                name="furniture",
                view_name="svr_partition_furniture",
                index_name="svr_vector_idx_furniture",
                filter_value="furniture",
                document_count=8500,
                status=PartitionStatus.ACTIVE,
                index_location=index_location,
                search_collection="svr_partition_furniture" if index_location == IndexLocation.VIEWS else "products",
            ),
            "clothing": PartitionInfo(
                name="clothing",
                view_name="svr_partition_clothing",
                index_name="svr_vector_idx_clothing",
                filter_value="clothing",
                document_count=23400,
                status=PartitionStatus.ACTIVE,
                index_location=index_location,
                search_collection="svr_partition_clothing" if index_location == IndexLocation.VIEWS else "products",
            ),
        }
    return config


def _make_mock_backend():
    """Create a mock backend for CLI tests."""
    backend = AsyncMock()
    backend.connect = AsyncMock()
    backend.disconnect = AsyncMock()
    backend.is_connected = AsyncMock(return_value=True)
    backend.count_documents = AsyncMock(return_value=1000)
    backend.get_distinct_values = AsyncMock(return_value=["electronics", "furniture", "clothing"])
    backend.get_partition_document_counts = AsyncMock(
        return_value={"electronics": 15000, "furniture": 8500, "clothing": 23400}
    )
    backend.get_index_status = AsyncMock(return_value={
        "name": "svr_vector_idx_electronics",
        "status": "READY",
        "queryable": True,
    })
    backend.create_partition_view = AsyncMock(return_value="svr_partition_test")
    backend.create_vector_search_index = AsyncMock()
    backend.delete_vector_search_index = AsyncMock()
    backend.delete_partition_view = AsyncMock()
    return backend


# ===========================
# Partitions commands
# ===========================

class TestPartitionsList:
    @patch("semantic_vector_router.cli.partitions.load_config")
    def test_partitions_list_shows_table(self, mock_load_config, runner):
        mock_load_config.return_value = _make_mock_config()
        result = runner.invoke(main, ["partitions", "list"])
        assert result.exit_code == 0
        assert "electronics" in result.output
        assert "furniture" in result.output
        assert "clothing" in result.output
        assert "15,000" in result.output

    @patch("semantic_vector_router.cli.partitions.load_config")
    def test_partitions_list_empty(self, mock_load_config, runner):
        mock_load_config.return_value = _make_mock_config(with_partitions=False)
        result = runner.invoke(main, ["partitions", "list"])
        assert result.exit_code == 0
        assert "No partitions" in result.output


class TestPartitionsStatus:
    @patch("semantic_vector_router.cli.partitions.load_config")
    def test_partitions_status_single(self, mock_load_config, runner):
        mock_load_config.return_value = _make_mock_config()
        result = runner.invoke(main, ["partitions", "status", "electronics"])
        assert result.exit_code == 0
        assert "electronics" in result.output
        assert "15,000" in result.output

    @patch("semantic_vector_router.cli.partitions.load_config")
    def test_partitions_status_not_found(self, mock_load_config, runner):
        mock_load_config.return_value = _make_mock_config()
        result = runner.invoke(main, ["partitions", "status", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

    @patch("semantic_vector_router.cli.partitions.load_config")
    def test_partitions_status_all(self, mock_load_config, runner):
        mock_load_config.return_value = _make_mock_config()
        result = runner.invoke(main, ["partitions", "status"])
        assert result.exit_code == 0
        assert "electronics" in result.output
        assert "furniture" in result.output


class TestPartitionsCreate:
    @patch("semantic_vector_router.cli.partitions.PartitionProvisioner")
    @patch("semantic_vector_router.cli.partitions._get_backend")
    def test_partitions_create_success(self, mock_get_backend, mock_prov_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_prov = AsyncMock()
        mock_prov.create_partition = AsyncMock(return_value=PartitionInfo(
            name="toys",
            index_name="svr_vector_idx_toys",
            filter_value="toys",
            document_count=500,
            status=PartitionStatus.ACTIVE,
        ))
        mock_prov_cls.return_value = mock_prov

        result = runner.invoke(main, ["partitions", "create", "toys"])
        assert result.exit_code == 0
        assert "Created partition" in result.output


class TestPartitionsDelete:
    @patch("semantic_vector_router.cli.partitions.PartitionProvisioner")
    @patch("semantic_vector_router.cli.partitions._get_backend")
    @patch("semantic_vector_router.cli.partitions.load_config")
    def test_partitions_delete_with_confirmation(self, mock_load_config, mock_get_backend, mock_prov_cls, runner):
        mock_config = _make_mock_config()
        mock_load_config.return_value = mock_config
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_prov = AsyncMock()
        mock_prov.delete_partition = AsyncMock()
        mock_prov_cls.return_value = mock_prov

        result = runner.invoke(main, ["partitions", "delete", "electronics", "-y"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    @patch("semantic_vector_router.cli.partitions.load_config")
    def test_partitions_delete_not_found(self, mock_load_config, runner):
        mock_load_config.return_value = _make_mock_config()
        result = runner.invoke(main, ["partitions", "delete", "nonexistent", "-y"])
        assert result.exit_code == 0
        assert "not found" in result.output


class TestPartitionsRefresh:
    @patch("semantic_vector_router.cli.partitions.PartitionProvisioner")
    @patch("semantic_vector_router.cli.partitions._get_backend")
    def test_partitions_refresh_updates_counts(self, mock_get_backend, mock_prov_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_prov = AsyncMock()
        mock_prov.update_all_partition_counts = AsyncMock(
            return_value={"electronics": 16000, "furniture": 9000, "clothing": 24000}
        )
        mock_prov_cls.return_value = mock_prov

        result = runner.invoke(main, ["partitions", "refresh"])
        assert result.exit_code == 0
        assert "16,000" in result.output


class TestPartitionsScan:
    @patch("semantic_vector_router.cli.partitions.PartitionScanner")
    @patch("semantic_vector_router.cli.partitions._get_backend")
    def test_partitions_scan_shows_new_values(self, mock_get_backend, mock_scanner_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_scanner = AsyncMock()
        mock_scanner.get_new_partition_values = AsyncMock(return_value=["toys", "books"])
        mock_scanner_cls.return_value = mock_scanner

        result = runner.invoke(main, ["partitions", "scan"])
        assert result.exit_code == 0
        assert "toys" in result.output
        assert "books" in result.output

    @patch("semantic_vector_router.cli.partitions.PartitionScanner")
    @patch("semantic_vector_router.cli.partitions._get_backend")
    def test_partitions_scan_no_new(self, mock_get_backend, mock_scanner_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_scanner = AsyncMock()
        mock_scanner.get_new_partition_values = AsyncMock(return_value=[])
        mock_scanner_cls.return_value = mock_scanner

        result = runner.invoke(main, ["partitions", "scan"])
        assert result.exit_code == 0
        assert "No new" in result.output


class TestPartitionsProvision:
    @patch("semantic_vector_router.cli.partitions.PartitionProvisioner")
    @patch("semantic_vector_router.cli.partitions.PartitionScanner")
    @patch("semantic_vector_router.cli.partitions._get_backend")
    def test_partitions_provision_creates_new(self, mock_get_backend, mock_scanner_cls, mock_prov_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_scanner = AsyncMock()
        mock_scanner.get_new_partition_values = AsyncMock(return_value=["toys"])
        mock_scanner_cls.return_value = mock_scanner

        mock_prov = AsyncMock()
        mock_prov.create_partitions_batch = AsyncMock(return_value={
            "toys": PartitionInfo(
                name="toys",
                index_name="svr_vector_idx_toys",
                filter_value="toys",
                document_count=300,
            ),
        })
        mock_prov_cls.return_value = mock_prov

        result = runner.invoke(main, ["partitions", "provision"])
        assert result.exit_code == 0
        assert "Provisioned" in result.output


# ===========================
# Search command
# ===========================

class TestSearch:
    @patch("semantic_vector_router.cli.search.SVRClient")
    @patch("semantic_vector_router.cli.search._get_backend")
    def test_search_table_format(self, mock_get_backend, mock_client_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_client = MagicMock()
        mock_client._backend = mock_backend
        mock_client._connected = True
        mock_client._create_embedder = MagicMock()
        mock_client._create_reranker = MagicMock()

        search_result = SearchResult(
            hits=[
                SearchHit(id="doc1", score=0.95, partition="electronics",
                          document={"name": "Headphones", "price": 99.99}),
                SearchHit(id="doc2", score=0.88, partition="electronics",
                          document={"name": "Speaker", "price": 49.99}),
            ],
            query="wireless audio",
            partitions_searched=["electronics"],
            total_candidates=20,
            reranked=False,
            latency_ms=150.5,
        )
        mock_client.search = AsyncMock(return_value=search_result)
        mock_client._resolver = MagicMock()
        mock_client._merger = MagicMock()
        mock_client_cls.return_value = mock_client

        result = runner.invoke(main, ["search", "wireless audio"])
        assert result.exit_code == 0
        assert "electronics" in result.output

    @patch("semantic_vector_router.cli.search.SVRClient")
    @patch("semantic_vector_router.cli.search._get_backend")
    def test_search_json_format(self, mock_get_backend, mock_client_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_client = MagicMock()
        mock_client._backend = mock_backend
        mock_client._connected = True
        mock_client._create_embedder = MagicMock()
        mock_client._create_reranker = MagicMock()

        search_result = SearchResult(
            hits=[
                SearchHit(id="doc1", score=0.95, partition="electronics",
                          document={"name": "Headphones"}),
            ],
            query="wireless",
            partitions_searched=["electronics"],
            total_candidates=10,
            reranked=False,
            latency_ms=100.0,
        )
        mock_client.search = AsyncMock(return_value=search_result)
        mock_client._resolver = MagicMock()
        mock_client._merger = MagicMock()
        mock_client_cls.return_value = mock_client

        result = runner.invoke(main, ["search", "wireless", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output.strip())
        assert parsed["query"] == "wireless"
        assert len(parsed["hits"]) == 1

    @patch("semantic_vector_router.cli.search.SVRClient")
    @patch("semantic_vector_router.cli.search._get_backend")
    def test_search_with_partitions_filter(self, mock_get_backend, mock_client_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_client = MagicMock()
        mock_client._backend = mock_backend
        mock_client._connected = True
        mock_client._create_embedder = MagicMock()
        mock_client._create_reranker = MagicMock()

        search_result = SearchResult(
            hits=[], query="test", partitions_searched=["electronics"],
            total_candidates=0, reranked=False, latency_ms=50.0,
        )
        mock_client.search = AsyncMock(return_value=search_result)
        mock_client._resolver = MagicMock()
        mock_client._merger = MagicMock()
        mock_client_cls.return_value = mock_client

        result = runner.invoke(main, ["search", "test", "-p", "electronics,furniture"])
        assert result.exit_code == 0

    @patch("semantic_vector_router.cli.search.SVRClient")
    @patch("semantic_vector_router.cli.search._get_backend")
    def test_search_with_rerank_flag(self, mock_get_backend, mock_client_cls, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        mock_client = MagicMock()
        mock_client._backend = mock_backend
        mock_client._connected = True
        mock_client._create_embedder = MagicMock()
        mock_client._create_reranker = MagicMock()

        search_result = SearchResult(
            hits=[], query="test", partitions_searched=["electronics"],
            total_candidates=0, reranked=True, latency_ms=50.0,
        )
        mock_client.search = AsyncMock(return_value=search_result)
        mock_client._resolver = MagicMock()
        mock_client._merger = MagicMock()
        mock_client_cls.return_value = mock_client

        result = runner.invoke(main, ["search", "test", "--rerank"])
        assert result.exit_code == 0

    def test_search_no_config_error(self, runner):
        """Graceful error when config file doesn't exist."""
        result = runner.invoke(main, ["search", "test", "-c", "/nonexistent/config.json"])
        assert result.exit_code != 0 or "error" in result.output.lower() or "Aborted" in result.output


# ===========================
# Config commands
# ===========================

class TestConfigShow:
    @patch("semantic_vector_router.cli.config_cmd.load_config")
    def test_config_show_redacts_keys(self, mock_load_config, runner):
        mock_load_config.return_value = _make_mock_config()
        result = runner.invoke(main, ["config", "show"])
        assert result.exit_code == 0
        assert "****" in result.output


class TestConfigValidate:
    @patch("semantic_vector_router.cli.config_cmd.validate_config")
    @patch("semantic_vector_router.cli.config_cmd.load_config")
    def test_config_validate_shows_warnings(self, mock_load_config, mock_validate, runner):
        mock_load_config.return_value = _make_mock_config()
        mock_validate.return_value = ["Dimensions mismatch warning"]
        result = runner.invoke(main, ["config", "validate"])
        assert result.exit_code == 0
        assert "Dimensions mismatch warning" in result.output

    @patch("semantic_vector_router.cli.config_cmd.validate_config")
    @patch("semantic_vector_router.cli.config_cmd.load_config")
    def test_config_validate_no_warnings(self, mock_load_config, mock_validate, runner):
        mock_load_config.return_value = _make_mock_config()
        mock_validate.return_value = []
        result = runner.invoke(main, ["config", "validate"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()


class TestConfigSet:
    @patch("semantic_vector_router.cli.config_cmd.save_config")
    @patch("semantic_vector_router.cli.config_cmd.find_config_file")
    @patch("semantic_vector_router.cli.config_cmd.load_config")
    def test_config_set_updates_value(self, mock_load_config, mock_find, mock_save, runner):
        mock_load_config.return_value = _make_mock_config()
        mock_find.return_value = "/tmp/test_config.json"

        result = runner.invoke(main, ["config", "set", "resilience.search_timeout_ms", "60000"])
        assert result.exit_code == 0
        assert "Set" in result.output
        assert "60000" in result.output


class TestConfigPath:
    @patch("semantic_vector_router.cli.config_cmd.find_config_file")
    def test_config_path_shows_location(self, mock_find, runner):
        mock_find.return_value = "/home/user/.svr/config.json"
        result = runner.invoke(main, ["config", "path"])
        assert result.exit_code == 0
        assert "/home/user/.svr/config.json" in result.output

    @patch("semantic_vector_router.cli.config_cmd.find_config_file")
    def test_config_path_not_found(self, mock_find, runner):
        mock_find.return_value = None
        result = runner.invoke(main, ["config", "path"])
        assert result.exit_code == 0
        assert "No config" in result.output


# ===========================
# Index commands
# ===========================

class TestIndexStatus:
    @patch("semantic_vector_router.cli.index._get_backend")
    def test_index_status_table(self, mock_get_backend, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        result = runner.invoke(main, ["index", "status"])
        assert result.exit_code == 0
        assert "electronics" in result.output
        assert "READY" in result.output


class TestIndexWait:
    @patch("semantic_vector_router.cli.index._get_backend")
    def test_index_wait_already_queryable(self, mock_get_backend, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        result = runner.invoke(main, ["index", "wait", "electronics"])
        assert result.exit_code == 0
        assert "queryable" in result.output

    @patch("semantic_vector_router.cli.index._get_backend")
    def test_index_wait_timeout(self, mock_get_backend, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_backend.get_index_status = AsyncMock(return_value={
            "name": "svr_vector_idx_electronics",
            "status": "PENDING",
            "queryable": False,
        })
        mock_get_backend.return_value = (mock_config, mock_backend)

        result = runner.invoke(main, ["index", "wait", "electronics", "--timeout", "6"])
        assert result.exit_code == 0
        assert "Timeout" in result.output

    @patch("semantic_vector_router.cli.index._get_backend")
    def test_index_wait_not_found(self, mock_get_backend, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        result = runner.invoke(main, ["index", "wait", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output


# ===========================
# Analyze command
# ===========================

class TestAnalyze:
    @patch("semantic_vector_router.cli.analyze.analyze_fields")
    @patch("semantic_vector_router.cli.analyze._get_backend")
    def test_analyze_full(self, mock_get_backend, mock_analyze, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        from semantic_vector_router.utils.field_analyzer import FieldAnalysis
        mock_analyze.return_value = [
            FieldAnalysis(
                name="category",
                distinct_count=5,
                total_documents=10000,
                coverage=1.0,
                cardinality_ratio=0.0005,
                is_suitable=True,
                reason="Good filter candidate",
            ),
            FieldAnalysis(
                name="brand",
                distinct_count=200,
                total_documents=10000,
                coverage=0.95,
                cardinality_ratio=0.02,
                is_suitable=True,
                reason="Good filter candidate",
            ),
        ]

        result = runner.invoke(main, ["analyze"])
        assert result.exit_code == 0
        assert "category" in result.output
        assert "brand" in result.output

    @patch("semantic_vector_router.cli.analyze.get_recommended_filter_fields")
    @patch("semantic_vector_router.cli.analyze.analyze_fields")
    @patch("semantic_vector_router.cli.analyze._get_backend")
    def test_analyze_filters_only(self, mock_get_backend, mock_analyze, mock_rec, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        from semantic_vector_router.utils.field_analyzer import FieldAnalysis
        mock_analyze.return_value = [
            FieldAnalysis(
                name="category",
                distinct_count=5,
                total_documents=10000,
                coverage=1.0,
                cardinality_ratio=0.0005,
                is_suitable=True,
                reason="Good filter candidate",
            ),
        ]
        mock_rec.return_value = ["category"]

        result = runner.invoke(main, ["analyze", "--filters"])
        assert result.exit_code == 0
        assert "category" in result.output
        assert "Recommended" in result.output

    @patch("semantic_vector_router.cli.analyze.analyze_fields")
    @patch("semantic_vector_router.cli.analyze._get_backend")
    def test_analyze_specific_field(self, mock_get_backend, mock_analyze, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        from semantic_vector_router.utils.field_analyzer import FieldAnalysis
        mock_analyze.return_value = [
            FieldAnalysis(
                name="category",
                distinct_count=5,
                total_documents=10000,
                coverage=1.0,
                cardinality_ratio=0.0005,
                is_suitable=True,
                reason="Good filter candidate",
                sample_values=["electronics", "furniture", "clothing"],
            ),
        ]

        result = runner.invoke(main, ["analyze", "--field", "category"])
        assert result.exit_code == 0
        assert "category" in result.output
        assert "Distinct values" in result.output


# ===========================
# Watch commands
# ===========================

class TestWatch:
    @patch("semantic_vector_router.cli.watch.load_config")
    def test_watch_status(self, mock_load_config, runner):
        config = _make_mock_config()
        config.lifecycle.pending_partitions = ["toys", "books"]
        mock_load_config.return_value = config

        result = runner.invoke(main, ["watch", "status"])
        assert result.exit_code == 0
        assert "toys" in result.output
        assert "books" in result.output

    @patch("semantic_vector_router.cli.watch.PartitionWatcher")
    @patch("semantic_vector_router.cli.watch.load_config")
    def test_watch_reject(self, mock_load_config, mock_watcher_cls, runner):
        config = _make_mock_config()
        config.lifecycle.pending_partitions = ["toys"]
        mock_load_config.return_value = config

        mock_watcher = MagicMock()
        mock_watcher.reject_partition = MagicMock(return_value=True)
        mock_watcher_cls.return_value = mock_watcher

        result = runner.invoke(main, ["watch", "reject", "toys"])
        assert result.exit_code == 0
        assert "Rejected" in result.output


# ===========================
# Split commands
# ===========================

class TestSplit:
    @patch("semantic_vector_router.cli.split._get_backend")
    def test_split_check_no_auto_split(self, mock_get_backend, runner):
        mock_config = _make_mock_config()
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = (mock_config, mock_backend)

        result = runner.invoke(main, ["split", "check"])
        assert result.exit_code == 0
        assert "not enabled" in result.output

    def test_split_execute_no_args(self, runner):
        result = runner.invoke(main, ["split", "execute"])
        assert result.exit_code == 0
        assert "Provide a partition name" in result.output


# ===========================
# Non-interactive init
# ===========================

class TestInitNonInteractive:
    @patch("semantic_vector_router.cli.init.save_config")
    @patch("semantic_vector_router.cli.init.validate_config")
    def test_init_non_interactive_creates_config(self, mock_validate, mock_save, runner):
        mock_validate.return_value = []
        mock_save.return_value = "/tmp/config.json"

        result = runner.invoke(main, [
            "init", "--non-interactive",
            "--database", "mydb",
            "--collection", "products",
            "--partition-field", "category",
            "--embedding-provider", "voyage",
            "--embedding-model", "voyage-4",
            "--dimensions", "1024",
            "--index-location", "views",
        ])
        assert result.exit_code == 0
        assert "saved" in result.output.lower() or "Configuration" in result.output

    def test_init_non_interactive_missing_required(self, runner):
        result = runner.invoke(main, [
            "init", "--non-interactive",
            "--database", "mydb",
            # Missing --collection and --partition-field
        ])
        assert result.exit_code != 0 or "Missing" in result.output


# ===========================
# CLI help
# ===========================

class TestHelp:
    def test_main_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "partitions" in result.output
        assert "search" in result.output
        assert "analyze" in result.output
        assert "config" in result.output
        assert "index" in result.output
        assert "watch" in result.output
        assert "split" in result.output
        assert "init" in result.output

    def test_partitions_help(self, runner):
        result = runner.invoke(main, ["partitions", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "create" in result.output
        assert "delete" in result.output
        assert "scan" in result.output
        assert "provision" in result.output

    def test_search_help(self, runner):
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--partitions" in result.output
        assert "--limit" in result.output
        assert "--format" in result.output

    def test_config_help(self, runner):
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "validate" in result.output
        assert "set" in result.output
        assert "path" in result.output

    def test_index_help(self, runner):
        result = runner.invoke(main, ["index", "--help"])
        assert result.exit_code == 0
        assert "status" in result.output
        assert "rebuild" in result.output
        assert "wait" in result.output
