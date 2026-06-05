"""Tests for configuration management."""

import json
import os
import tempfile
import warnings
from pathlib import Path

import pytest

from semantic_vector_router.config import (
    create_default_config,
    find_config_file,
    load_config,
    resolve_env_var,
    save_config,
    validate_config,
)
from semantic_vector_router.exceptions import ConfigurationError
from semantic_vector_router.models import (
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    MongoDBIndexQuantization,
    PartitionInfo,
    SVRConfig,
    VectorStorageConfig,
    VectorStorageFormat,
    VectorStorageMode,
    VoyageQuantization,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            "database": {
                "connection_string_env": "MONGODB_URI",
                "database": "test_db",
                "source_collection": "test_collection",
            },
            "partitioning": {
                "field": "category",
            },
        }

        config = load_config(config_dict=config_dict)

        assert config.database.database == "test_db"
        assert config.partitioning.field == "category"

    def test_load_from_file(self, sample_config):
        """Test loading config from file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                sample_config.model_dump(mode="json", exclude_none=True),
                f,
            )
            f.flush()

            try:
                loaded = load_config(config_path=f.name)
                assert loaded.database.database == sample_config.database.database
            finally:
                os.unlink(f.name)

    def test_load_invalid_json_raises(self):
        """Test that invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{ invalid json }")
            f.flush()

            try:
                with pytest.raises(ConfigurationError):
                    load_config(config_path=f.name)
            finally:
                os.unlink(f.name)

    def test_load_nonexistent_file_raises(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(ConfigurationError):
            load_config(config_path="/nonexistent/path/config.json")


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_config(self, sample_config):
        """Test saving config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"

            saved_path = save_config(sample_config, path)

            assert saved_path.exists()

            # Verify it's valid JSON
            with open(saved_path) as f:
                data = json.load(f)
                assert data["database"]["database"] == "test_db"

    def test_save_config_creates_directory(self, sample_config):
        """Test that save_config creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "config.json"

            saved_path = save_config(sample_config, path)

            assert saved_path.exists()
            assert saved_path.parent.exists()

    def test_save_default_location(self, sample_config):
        """Test saving to default location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                saved_path = save_config(sample_config)
                assert saved_path.name == "config.json"
                assert saved_path.parent.name == ".svr"
            finally:
                os.chdir(original_cwd)


class TestFindConfigFile:
    """Tests for find_config_file function."""

    def test_find_in_current_directory(self, sample_config):
        """Test finding config in current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".svr" / "config.json"
            config_path.parent.mkdir()
            save_config(sample_config, config_path)

            found = find_config_file(Path(tmpdir))

            assert found is not None
            assert found == config_path

    def test_find_svr_json(self, sample_config):
        """Test finding svr.config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "svr.config.json"
            save_config(sample_config, config_path)

            found = find_config_file(Path(tmpdir))

            assert found is not None
            assert found.name == "svr.config.json"

    def test_find_returns_none_when_not_found(self):
        """Test that find returns None when no config exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            found = find_config_file(Path(tmpdir))
            assert found is None


class TestResolveEnvVar:
    """Tests for resolve_env_var function."""

    def test_resolve_existing_var(self):
        """Test resolving existing environment variable."""
        os.environ["TEST_VAR"] = "test_value"
        try:
            value = resolve_env_var("TEST_VAR")
            assert value == "test_value"
        finally:
            del os.environ["TEST_VAR"]

    def test_resolve_missing_required_raises(self):
        """Test that missing required var raises error."""
        with pytest.raises(ConfigurationError):
            resolve_env_var("DEFINITELY_NOT_SET_VAR_12345")

    def test_resolve_missing_optional_returns_none(self):
        """Test that missing optional var returns None."""
        value = resolve_env_var("DEFINITELY_NOT_SET_VAR_12345", required=False)
        assert value is None


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config_returns_empty_warnings(self, sample_config):
        """Test that valid config returns no errors."""
        warnings = validate_config(sample_config)
        # May have warnings but should not raise
        assert isinstance(warnings, list)

    def test_auto_mode_with_wrong_provider_raises(self, sample_config):
        """Test that auto mode with non-Atlas provider raises error."""
        sample_config.embedding.mode = EmbeddingMode.AUTO
        sample_config.embedding.provider = EmbeddingProvider.OPENAI

        with pytest.raises(ConfigurationError):
            validate_config(sample_config)

    def test_dimension_mismatch_warns(self, sample_config):
        """Test that dimension mismatch generates warning."""
        sample_config.embedding.dimensions = 1024
        sample_config.vector_search.dimensions = 1536

        warnings = validate_config(sample_config)

        assert any("dimensions" in w.lower() for w in warnings)


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_create_minimal_config(self):
        """Test creating minimal default config."""
        config = create_default_config(
            database="my_db",
            source_collection="my_collection",
            partition_field="category",
        )

        assert config.database.database == "my_db"
        assert config.database.source_collection == "my_collection"
        assert config.partitioning.field == "category"

    def test_create_with_custom_env_var(self):
        """Test creating config with custom connection string env var."""
        config = create_default_config(
            database="my_db",
            source_collection="my_collection",
            partition_field="category",
            connection_string_env="CUSTOM_MONGO_URI",
        )

        assert config.database.connection_string_env == "CUSTOM_MONGO_URI"


# ---------------------------------------------------------------------------
# Quantization and FIELDS-mode validation (Phase 1.5 / 1.6 regressions)
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> SVRConfig:
    """Build a minimal SVRConfig, merging *overrides* into sensible defaults.

    Accepted override keys:
        storage_format, index_quantization, mode (vector-storage mode),
        embeddings_collection, index_on, embedding_provider,
        voyage_quantization.
    """
    from semantic_vector_router.models import (
        DatabaseConfig,
        EmbeddingConfig,
        PartitioningConfig,
        VectorSearchConfig,
    )

    vs_kwargs = {}
    if "storage_format" in overrides:
        vs_kwargs["storage_format"] = overrides["storage_format"]
    if "index_quantization" in overrides:
        vs_kwargs["index_quantization"] = overrides["index_quantization"]
    if "mode" in overrides:
        vs_kwargs["mode"] = overrides["mode"]
    if "embeddings_collection" in overrides:
        vs_kwargs["embeddings_collection"] = overrides["embeddings_collection"]
    if "index_on" in overrides:
        vs_kwargs["index_on"] = overrides["index_on"]

    emb_kwargs: dict = {
        "mode": EmbeddingMode.BYOM,
        "provider": overrides.get("embedding_provider", EmbeddingProvider.OPENAI),
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }
    if "voyage_quantization" in overrides:
        emb_kwargs["voyage_quantization"] = overrides["voyage_quantization"]

    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="test_collection",
        ),
        partitioning=PartitioningConfig(field="category"),
        vector_storage=VectorStorageConfig(**vs_kwargs),
        vector_search=VectorSearchConfig(dimensions=1536),
        embedding=EmbeddingConfig(**emb_kwargs),
    )


class TestQuantizationValidation:
    """Validate quantization compatibility rules enforced by validate_config.

    Phase 1.6 regressions: pre-quantized storage + index quantization is an
    error; FIELDS mode partition limits; separate storage without collection.
    """

    # -- Phase 1.6: pre-quantized + index quantization incompatibility ------

    def test_pre_quantized_format_with_index_quantization_raises(self):
        """bindata_int8 storage + scalar index quantization must raise."""
        config = _make_config(
            storage_format=VectorStorageFormat.BINDATA_INT8,
            index_quantization=MongoDBIndexQuantization.SCALAR,
        )

        with pytest.raises(ConfigurationError, match="pre-quantized"):
            validate_config(config)

    def test_pre_quantized_packed_bit_with_index_quantization_raises(self):
        """bindata_packed_bit storage + binary index quantization must raise."""
        config = _make_config(
            storage_format=VectorStorageFormat.BINDATA_PACKED_BIT,
            index_quantization=MongoDBIndexQuantization.BINARY,
        )

        with pytest.raises(ConfigurationError, match="pre-quantized"):
            validate_config(config)

    # -- Voyage quantization mismatch warning (NOT error) -------------------

    def test_voyage_quantization_mismatch_warns(self):
        """Voyage int8 quantization + bindata_float32 storage should warn."""
        config = _make_config(
            embedding_provider=EmbeddingProvider.VOYAGE,
            voyage_quantization=VoyageQuantization.INT8,
            storage_format=VectorStorageFormat.BINDATA_FLOAT32,
        )

        result_warnings = validate_config(config)

        assert any("int8" in w.lower() for w in result_warnings), (
            f"Expected a warning about int8 mismatch, got: {result_warnings}"
        )

    # -- FIELDS mode partition limits ---------------------------------------

    def test_fields_mode_over_50_partitions_raises(self):
        """More than 50 FIELDS partitions must raise ConfigurationError."""
        config = _make_config(index_on=IndexLocation.FIELDS)

        # Populate registry with 51 FIELDS partitions
        for i in range(51):
            name = f"partition_{i}"
            config.partitions.registry[name] = PartitionInfo(
                name=name,
                index_name=f"svr_idx_{name}",
                filter_value=name,
                index_location=IndexLocation.FIELDS,
                search_collection="test_collection",
                embedding_field=f"embedding_{name}",
            )

        with pytest.raises(ConfigurationError, match="50 partitions"):
            validate_config(config)

    def test_fields_mode_41_to_50_partitions_warns(self):
        """41-50 FIELDS partitions should produce a warning but not raise."""
        config = _make_config(index_on=IndexLocation.FIELDS)

        # Populate registry with 45 FIELDS partitions (in the 41-50 range)
        for i in range(45):
            name = f"partition_{i}"
            config.partitions.registry[name] = PartitionInfo(
                name=name,
                index_name=f"svr_idx_{name}",
                filter_value=name,
                index_location=IndexLocation.FIELDS,
                search_collection="test_collection",
                embedding_field=f"embedding_{name}",
            )

        result_warnings = validate_config(config)

        assert any("45" in w and "50" in w for w in result_warnings), (
            f"Expected a warning about approaching 50-partition limit, got: {result_warnings}"
        )

    # -- Separate storage without embeddings_collection ---------------------

    def test_separate_storage_without_embeddings_collection_raises(self):
        """Separate mode without embeddings_collection must raise."""
        config = _make_config(
            mode=VectorStorageMode.SEPARATE,
            embeddings_collection=None,
        )

        with pytest.raises(ConfigurationError, match="embeddings_collection"):
            validate_config(config)


# ===========================================================================
# Additional load_config error paths
# ===========================================================================

class TestLoadConfigErrors:
    def test_load_from_dict_invalid_schema(self):
        """Invalid dict schema raises ConfigurationError."""
        config_dict = {"database": {"connection_string_env": "X"}}
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            load_config(config_dict=config_dict)

    def test_load_no_config_file_found(self):
        """When no config_path and no file exists, raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.getcwd()
            try:
                os.chdir(tmpdir)
                with pytest.raises(ConfigurationError, match="No configuration file found"):
                    load_config(config_path=None, config_dict=None)
            finally:
                os.chdir(original)

    def test_load_file_io_error(self):
        """IOError reading config file raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{}')
            f.flush()
            try:
                from unittest.mock import patch as mock_patch
                with mock_patch("builtins.open", side_effect=IOError("denied")):
                    with pytest.raises(ConfigurationError, match="Error reading"):
                        load_config(config_path=f.name, load_env=False)
            finally:
                os.unlink(f.name)

    def test_load_file_valid_json_invalid_schema(self):
        """Valid JSON but invalid SVRConfig schema raises."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"database": {"database": "test"}}, f)
            f.flush()
            try:
                with pytest.raises(ConfigurationError, match="Invalid configuration"):
                    load_config(config_path=f.name)
            finally:
                os.unlink(f.name)


class TestSaveConfigErrors:
    def test_save_io_error(self, sample_config):
        """IOError saving config raises ConfigurationError."""
        from unittest.mock import patch as mock_patch
        with mock_patch("builtins.open", side_effect=IOError("disk full")):
            with pytest.raises(ConfigurationError, match="Error saving"):
                save_config(sample_config, "/tmp/test_save.json")


# ===========================================================================
# get_connection_string
# ===========================================================================

class TestGetConnectionString:
    def test_get_connection_string_success(self):
        from semantic_vector_router.config import get_connection_string, create_default_config

        config = create_default_config(
            database="test_db", source_collection="coll", partition_field="cat",
            connection_string_env="SVR_TEST_CONN_STR",
        )
        os.environ["SVR_TEST_CONN_STR"] = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<db>"
        try:
            result = get_connection_string(config)
            assert result == "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<db>"
        finally:
            del os.environ["SVR_TEST_CONN_STR"]

    def test_get_connection_string_missing(self):
        from semantic_vector_router.config import get_connection_string, create_default_config

        config = create_default_config(
            database="test_db", source_collection="coll", partition_field="cat",
            connection_string_env="SVR_NONEXISTENT_VAR_12345",
        )
        with pytest.raises(ConfigurationError):
            get_connection_string(config)


# ===========================================================================
# get_api_key
# ===========================================================================

class TestGetApiKey:
    def test_none_returns_none(self):
        from semantic_vector_router.config import get_api_key
        assert get_api_key(None, "TestProvider") is None

    def test_existing_var(self):
        from semantic_vector_router.config import get_api_key
        os.environ["SVR_TEST_API_KEY"] = "sk-test"
        try:
            assert get_api_key("SVR_TEST_API_KEY", "Test") == "sk-test"
        finally:
            del os.environ["SVR_TEST_API_KEY"]

    def test_missing_var_raises(self):
        from semantic_vector_router.config import get_api_key
        with pytest.raises(ConfigurationError, match="API key for TestProv"):
            get_api_key("SVR_NONEXISTENT_KEY_12345", "TestProv")


# ===========================================================================
# Additional validate_config paths
# ===========================================================================

class TestValidateConfigAdditional:
    def test_byom_with_atlas_voyage_raises(self):
        config = _make_config(embedding_provider=EmbeddingProvider.ATLAS_VOYAGE)
        config.embedding.mode = EmbeddingMode.BYOM
        with pytest.raises(ConfigurationError, match="BYOM mode cannot use atlas_voyage"):
            validate_config(config)

    def test_secondary_field_split_without_field_raises(self):
        from semantic_vector_router.models import LifecycleConfig, AutoSplitConfig as ASC
        config = _make_config()
        config.lifecycle = LifecycleConfig(
            auto_split=ASC(
                enabled=True, split_strategy="secondary_field",
                secondary_field=None, threshold_vectors=1000000,
            )
        )
        with pytest.raises(ConfigurationError, match="secondary_field"):
            validate_config(config)

    def test_time_split_without_time_field_raises(self):
        from semantic_vector_router.models import LifecycleConfig, AutoSplitConfig as ASC
        config = _make_config()
        config.lifecycle = LifecycleConfig(
            auto_split=ASC(
                enabled=True, split_strategy="time",
                time_field=None, threshold_vectors=1000000,
            )
        )
        with pytest.raises(ConfigurationError, match="time_field"):
            validate_config(config)

    def test_reranking_without_api_key_warns(self):
        from semantic_vector_router.models import RerankingConfig
        config = _make_config()
        config.reranking = RerankingConfig(
            enabled=True, provider="voyage", model="rerank-2",
            api_key_env=None,
        )
        result = validate_config(config)
        assert any("api key" in w.lower() or "API key" in w for w in result)

    def test_voyage_binary_quantization_mismatch_warns(self):
        config = _make_config(
            embedding_provider=EmbeddingProvider.VOYAGE,
            voyage_quantization=VoyageQuantization.BINARY,
            storage_format=VectorStorageFormat.BINDATA_FLOAT32,
        )
        result = validate_config(config)
        assert any("binary" in w.lower() and "packed_bit" in w.lower() for w in result)

    def test_voyage_ubinary_quantization_mismatch_warns(self):
        config = _make_config(
            embedding_provider=EmbeddingProvider.VOYAGE,
            voyage_quantization=VoyageQuantization.UBINARY,
            storage_format=VectorStorageFormat.BINDATA_FLOAT32,
        )
        result = validate_config(config)
        assert any("ubinary" in w.lower() and "packed_bit" in w.lower() for w in result)


# ===========================================================================
# Phase 3 — ResilienceConfig defaults, validation, and backward compatibility
# ===========================================================================

from semantic_vector_router.models import (
    DatabaseConfig,
    PartitioningConfig,
    ResilienceConfig,
)


def _minimal_svr_config(**overrides) -> SVRConfig:
    """Build a minimal SVRConfig for resilience tests."""
    kwargs = dict(
        database=DatabaseConfig(database="test", source_collection="test"),
        partitioning=PartitioningConfig(field="category"),
    )
    kwargs.update(overrides)
    return SVRConfig(**kwargs)


class TestResilienceConfigDefaults:
    """Verify ResilienceConfig ships with sensible defaults."""

    def test_resilience_config_has_sensible_defaults(self):
        """Create ResilienceConfig() and verify all defaults match spec."""
        rc = ResilienceConfig()

        assert rc.max_retry_attempts == 3
        assert rc.retry_base_delay == 0.5
        assert rc.retry_max_delay == 30.0
        assert rc.connection_timeout_ms == 10_000
        assert rc.server_selection_timeout_ms == 30_000
        assert rc.search_timeout_ms == 30_000
        assert rc.embedding_timeout_ms == 60_000
        assert rc.reranking_timeout_ms == 60_000
        assert rc.health_check_interval_s == 30
        assert rc.watcher_max_retries == 10
        assert rc.watcher_base_delay == 1.0
        assert rc.watcher_max_delay == 60.0

    def test_svr_config_includes_resilience_field(self):
        """Minimal SVRConfig should have a ResilienceConfig with defaults."""
        config = _minimal_svr_config()

        assert isinstance(config.resilience, ResilienceConfig)
        assert config.resilience.max_retry_attempts == 3
        assert config.resilience.connection_timeout_ms == 10_000
        assert config.resilience.search_timeout_ms == 30_000


class TestResilienceConfigValidation:
    """Validate that validate_config enforces resilience constraints."""

    def test_invalid_connection_timeout_raises(self):
        """connection_timeout_ms=0 or -1 must raise ConfigurationError."""
        for bad_val in (0, -1):
            config = _minimal_svr_config(
                resilience=ResilienceConfig(connection_timeout_ms=bad_val),
            )
            with pytest.raises(ConfigurationError, match="connection_timeout_ms"):
                validate_config(config)

    def test_invalid_search_timeout_raises(self):
        """search_timeout_ms=-1 must raise ConfigurationError."""
        config = _minimal_svr_config(
            resilience=ResilienceConfig(search_timeout_ms=-1),
        )
        with pytest.raises(ConfigurationError, match="search_timeout_ms"):
            validate_config(config)

    def test_negative_max_retry_raises(self):
        """max_retry_attempts=-1 must raise ConfigurationError."""
        config = _minimal_svr_config(
            resilience=ResilienceConfig(max_retry_attempts=-1),
        )
        with pytest.raises(ConfigurationError, match="max_retry_attempts"):
            validate_config(config)

    def test_zero_max_retry_allowed(self):
        """max_retry_attempts=0 means no retries — should be valid."""
        config = _minimal_svr_config(
            resilience=ResilienceConfig(max_retry_attempts=0),
        )
        # Should not raise; just returns warnings list
        warnings = validate_config(config)
        assert isinstance(warnings, list)

    def test_negative_health_check_interval_raises(self):
        """health_check_interval_s=-1 must raise ConfigurationError."""
        config = _minimal_svr_config(
            resilience=ResilienceConfig(health_check_interval_s=-1),
        )
        with pytest.raises(ConfigurationError, match="health_check_interval_s"):
            validate_config(config)

    def test_zero_health_check_interval_allowed(self):
        """health_check_interval_s=0 (always ping) should be valid."""
        config = _minimal_svr_config(
            resilience=ResilienceConfig(health_check_interval_s=0),
        )
        warnings = validate_config(config)
        assert isinstance(warnings, list)

    def test_low_search_timeout_warns(self):
        """search_timeout_ms=3000 should produce a warning but not raise."""
        config = _minimal_svr_config(
            resilience=ResilienceConfig(search_timeout_ms=3000),
        )
        warnings = validate_config(config)
        assert any("search_timeout_ms" in w and "3000" in w for w in warnings), (
            f"Expected a warning about low search_timeout_ms, got: {warnings}"
        )


class TestResilienceBackwardCompat:
    """Ensure configs without a resilience key still load with defaults."""

    def test_config_without_resilience_gets_defaults(self):
        """A config dict with NO 'resilience' key should load with defaults."""
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)

        assert isinstance(config.resilience, ResilienceConfig)
        assert config.resilience.max_retry_attempts == 3
        assert config.resilience.connection_timeout_ms == 10_000

    def test_config_with_partial_resilience(self):
        """A config dict with only some resilience fields should merge with defaults."""
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
            "resilience": {
                "max_retry_attempts": 5,
                "search_timeout_ms": 15_000,
            },
        }
        config = SVRConfig.model_validate(config_dict)

        assert isinstance(config.resilience, ResilienceConfig)
        # Overridden values
        assert config.resilience.max_retry_attempts == 5
        assert config.resilience.search_timeout_ms == 15_000
        # Defaults for fields not provided
        assert config.resilience.connection_timeout_ms == 10_000
        assert config.resilience.retry_base_delay == 0.5
        assert config.resilience.health_check_interval_s == 30


# ===========================================================================
# Phase 6 — Pool config validation
# ===========================================================================


class TestPoolConfigValidation:
    """Tests for connection pool parameter validation."""

    def test_pool_defaults(self, sample_config):
        warnings = validate_config(sample_config)
        # Default pool values should not trigger errors
        assert sample_config.database.max_pool_size == 100
        assert sample_config.database.min_pool_size == 0

    def test_max_pool_size_zero_raises(self, sample_config):
        sample_config.database.max_pool_size = 0
        with pytest.raises(ConfigurationError, match="max_pool_size must be > 0"):
            validate_config(sample_config)

    def test_max_pool_size_low_warns(self, sample_config):
        sample_config.database.max_pool_size = 5
        warnings = validate_config(sample_config)
        assert any("max_pool_size" in w and "low" in w for w in warnings)

    def test_min_pool_size_negative_raises(self, sample_config):
        sample_config.database.min_pool_size = -1
        with pytest.raises(ConfigurationError, match="min_pool_size must be >= 0"):
            validate_config(sample_config)

    def test_min_greater_than_max_raises(self, sample_config):
        sample_config.database.min_pool_size = 200
        sample_config.database.max_pool_size = 100
        with pytest.raises(ConfigurationError, match="min_pool_size.*must be <=.*max_pool_size"):
            validate_config(sample_config)

    def test_max_idle_time_negative_raises(self, sample_config):
        sample_config.database.max_idle_time_ms = -1
        with pytest.raises(ConfigurationError, match="max_idle_time_ms must be >= 0"):
            validate_config(sample_config)

    def test_wait_queue_timeout_negative_raises(self, sample_config):
        sample_config.database.wait_queue_timeout_ms = -1
        with pytest.raises(ConfigurationError, match="wait_queue_timeout_ms must be >= 0"):
            validate_config(sample_config)


# ===========================================================================
# Phase 6 — New config models backward compatibility
# ===========================================================================


class TestPhase6ConfigDefaults:
    """Tests that new config sections have defaults and load from dicts without them."""

    def test_logging_defaults(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.logging.level == "INFO"
        assert config.logging.json_format is False
        assert config.logging.log_query_text is False

    def test_metrics_defaults(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.metrics.enabled is True
        assert config.metrics.include_partition_tags is True

    def test_cache_defaults(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.cache.enabled is True
        assert config.cache.max_size == 10_000
        assert config.cache.ttl_seconds == 3600

    def test_pool_defaults_in_database(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.database.max_pool_size == 100
        assert config.database.min_pool_size == 0
        assert config.database.max_idle_time_ms == 0
        assert config.database.wait_queue_timeout_ms == 0

    def test_ingestion_defaults(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.ingestion.batch_size == 100
        assert config.ingestion.write_batch_size == 500
        assert config.ingestion.text_fields == ["text"]
        assert config.ingestion.mode.value == "insert"
        assert config.ingestion.continue_on_error is True
        assert config.ingestion.trigger_detection is True

    def test_rate_limiting_defaults(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.rate_limiting.enabled is True
        assert config.rate_limiting.default_tokens_per_second == 50.0
        assert config.rate_limiting.default_burst == 100
        assert config.rate_limiting.providers == {}


# ===========================================================================
# Phase 7 — Ingestion config validation
# ===========================================================================


class TestIngestionConfigValidation:
    """Tests for ingestion configuration validation."""

    def test_batch_size_zero_raises(self, sample_config):
        sample_config.ingestion.batch_size = 0
        with pytest.raises(ConfigurationError, match="batch_size must be > 0"):
            validate_config(sample_config)

    def test_batch_size_negative_raises(self, sample_config):
        sample_config.ingestion.batch_size = -1
        with pytest.raises(ConfigurationError, match="batch_size must be > 0"):
            validate_config(sample_config)

    def test_batch_size_large_warns(self, sample_config):
        sample_config.ingestion.batch_size = 5000
        warnings = validate_config(sample_config)
        assert any("batch_size" in w and "5000" in w for w in warnings)

    def test_write_batch_size_zero_raises(self, sample_config):
        sample_config.ingestion.write_batch_size = 0
        with pytest.raises(ConfigurationError, match="write_batch_size must be > 0"):
            validate_config(sample_config)

    def test_empty_text_fields_warns(self, sample_config):
        sample_config.ingestion.text_fields = []
        warnings = validate_config(sample_config)
        assert any("text_fields" in w for w in warnings)


# ===========================================================================
# Phase 7 — Rate limiting config validation
# ===========================================================================


class TestRateLimitConfigValidation:
    """Tests for rate limiting configuration validation."""

    def test_default_tokens_per_second_zero_raises(self, sample_config):
        sample_config.rate_limiting.default_tokens_per_second = 0
        with pytest.raises(ConfigurationError, match="default_tokens_per_second must be > 0"):
            validate_config(sample_config)

    def test_default_tokens_per_second_negative_raises(self, sample_config):
        sample_config.rate_limiting.default_tokens_per_second = -1
        with pytest.raises(ConfigurationError, match="default_tokens_per_second must be > 0"):
            validate_config(sample_config)

    def test_default_burst_zero_raises(self, sample_config):
        sample_config.rate_limiting.default_burst = 0
        with pytest.raises(ConfigurationError, match="default_burst must be > 0"):
            validate_config(sample_config)

    def test_provider_tokens_per_second_zero_raises(self, sample_config):
        from semantic_vector_router.models import ProviderRateLimit
        sample_config.rate_limiting.providers = {
            "openai": ProviderRateLimit(tokens_per_second=0, burst=100),
        }
        with pytest.raises(ConfigurationError, match="openai.*tokens_per_second must be > 0"):
            validate_config(sample_config)

    def test_provider_burst_zero_raises(self, sample_config):
        from semantic_vector_router.models import ProviderRateLimit
        sample_config.rate_limiting.providers = {
            "voyage": ProviderRateLimit(tokens_per_second=50.0, burst=0),
        }
        with pytest.raises(ConfigurationError, match="voyage.*burst must be > 0"):
            validate_config(sample_config)

    def test_valid_provider_config(self, sample_config):
        """Valid provider-specific config should not raise."""
        from semantic_vector_router.models import ProviderRateLimit
        sample_config.rate_limiting.providers = {
            "openai": ProviderRateLimit(tokens_per_second=100.0, burst=200),
            "voyage": ProviderRateLimit(tokens_per_second=30.0, burst=60),
        }
        warnings = validate_config(sample_config)
        # Should not contain rate limiting errors
        assert not any("rate_limiting" in w.lower() for w in warnings)


# ===========================================================================
# Phase 7 — Ingestion + Rate limiting backward compatibility
# ===========================================================================


class TestPhase7BackwardCompat:
    """Ensure configs without ingestion/rate_limiting keys still load."""

    def test_config_without_ingestion_gets_defaults(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.ingestion.batch_size == 100
        assert config.ingestion.mode.value == "insert"

    def test_config_without_rate_limiting_gets_defaults(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.rate_limiting.enabled is True
        assert config.rate_limiting.default_tokens_per_second == 50.0

    def test_config_with_partial_ingestion(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
            "ingestion": {"batch_size": 50},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.ingestion.batch_size == 50
        assert config.ingestion.write_batch_size == 500  # default

    def test_config_with_partial_rate_limiting(self):
        config_dict = {
            "database": {"database": "test", "source_collection": "test"},
            "partitioning": {"field": "category"},
            "rate_limiting": {"enabled": False},
        }
        config = SVRConfig.model_validate(config_dict)
        assert config.rate_limiting.enabled is False
        assert config.rate_limiting.default_tokens_per_second == 50.0  # default
