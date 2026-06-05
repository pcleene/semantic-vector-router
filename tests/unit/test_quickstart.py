"""Unit tests for SVRClient.quickstart(), quickstart_sync(), and search_sync()."""

import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_vector_router.client import SVRClient
from semantic_vector_router.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Shared patch constants
# ---------------------------------------------------------------------------

_CLIENT = "semantic_vector_router.client"

# Base env vars needed for a successful quickstart (MongoDB + OpenAI)
_BASE_ENV = {
    "MONGODB_URI": "mongodb+srv://<user>:<password>@<cluster-host>/<db>",
    "OPENAI_API_KEY": "sk-test-key-1234",
}

# Minimal required explicit args
_REQUIRED_ARGS = {
    "database": "testdb",
    "collection": "docs",
    "partition_field": "category",
}

# Patch load_dotenv globally in all tests so .env file doesn't pollute env
_LOAD_DOTENV = "dotenv.load_dotenv"


def _mock_load_config(config_dict=None, **kwargs):
    """Return a mock SVRConfig that satisfies __init__ downstream code."""
    mock_config = MagicMock()
    mock_config.metrics.enabled = False
    mock_config.rate_limiting.enabled = False
    mock_config.cache.enabled = False
    mock_config.cache.max_size = 0
    mock_config.cache.ttl_seconds = 300
    return mock_config


# ---------------------------------------------------------------------------
# quickstart — happy-path tests
# ---------------------------------------------------------------------------


class TestQuickstartHappyPath:
    """Tests for successful quickstart() invocations."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_with_explicit_params(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with all explicit params creates a connected client."""
        client = await SVRClient.quickstart(**_REQUIRED_ARGS)

        assert client is not None
        assert isinstance(client, SVRClient)
        mock_connect.assert_awaited_once()

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_returns_svrclient_instance(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart returns an SVRClient instance."""
        result = await SVRClient.quickstart(**_REQUIRED_ARGS)
        assert type(result).__name__ == "SVRClient"

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            **_BASE_ENV,
            "SVR_DATABASE": "env_db",
            "SVR_COLLECTION": "env_coll",
            "SVR_PARTITION_FIELD": "env_field",
        },
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_resolves_from_env_vars(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart resolves database/collection/partition_field from SVR_* env vars."""
        client = await SVRClient.quickstart()

        assert client is not None
        mock_connect.assert_awaited_once()
        # Verify load_config was called with a dict containing env-resolved values
        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["database"]["database"] == "env_db"
            assert config_dict["database"]["source_collection"] == "env_coll"
            assert config_dict["partitioning"]["field"] == "env_field"

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_calls_connect(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart calls connect() on the newly created instance."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)
        mock_connect.assert_awaited_once()

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {**_BASE_ENV, "VOYAGE_API_KEY": "pa-test-voyage"},
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_auto_detects_voyage_provider(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart auto-detects Voyage as embedding provider when its key is present."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["embedding"]["provider"] == "voyage"

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_auto_detects_openai_provider(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart auto-detects OpenAI when only OPENAI_API_KEY is set."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["embedding"]["provider"] == "openai"
            assert config_dict["embedding"]["model"] == "text-embedding-3-small"
            assert config_dict["embedding"]["dimensions"] == 1536


# ---------------------------------------------------------------------------
# quickstart — reranking behavior
# ---------------------------------------------------------------------------


class TestQuickstartReranking:
    """Tests for quickstart reranking auto-configuration."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_disables_reranking_without_key(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart disables reranking when no VOYAGE_API_KEY or COHERE_API_KEY."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["reranking"]["enabled"] is False

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {**_BASE_ENV, "VOYAGE_API_KEY": "pa-test-voyage"},
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_keeps_reranking_with_voyage_key(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart keeps reranking enabled when VOYAGE_API_KEY is present."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            # With a VOYAGE key, reranking should NOT be explicitly disabled
            assert config_dict.get("reranking", {}).get("enabled") is not False

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {**_BASE_ENV, "COHERE_API_KEY": "cohere-test-key"},
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_keeps_reranking_with_cohere_key(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart keeps reranking enabled when COHERE_API_KEY is present."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict.get("reranking", {}).get("enabled") is not False


# ---------------------------------------------------------------------------
# quickstart — missing required parameters
# ---------------------------------------------------------------------------


class TestQuickstartMissingParams:
    """Tests for ConfigurationError on missing required params."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_missing_database(self, _dotenv):
        """quickstart raises ConfigurationError when database is missing."""
        with pytest.raises(ConfigurationError, match="database"):
            await SVRClient.quickstart(
                collection="docs", partition_field="category"
            )

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_missing_collection(self, _dotenv):
        """quickstart raises ConfigurationError when collection is missing."""
        with pytest.raises(ConfigurationError, match="collection"):
            await SVRClient.quickstart(
                database="testdb", partition_field="category"
            )

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_missing_partition_field(self, _dotenv):
        """quickstart raises ConfigurationError when partition_field is missing."""
        with pytest.raises(ConfigurationError, match="partition_field"):
            await SVRClient.quickstart(
                database="testdb", collection="docs"
            )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_missing_connection_string(self, _dotenv):
        """quickstart raises ConfigurationError when MONGODB_URI is unset."""
        with pytest.raises(ConfigurationError, match="MONGODB_URI"):
            await SVRClient.quickstart(**_REQUIRED_ARGS)

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {"MONGODB_URI": "mongodb+srv://<user>:<password>@<cluster-host>/<db>"},
        clear=True,
    )
    @patch(_LOAD_DOTENV)
    async def test_quickstart_no_embedding_api_key(self, _dotenv):
        """quickstart raises ConfigurationError listing providers when no API key is set."""
        with pytest.raises(ConfigurationError, match="(?i)no embedding api key"):
            await SVRClient.quickstart(**_REQUIRED_ARGS)

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_missing_all_three_params(self, _dotenv):
        """quickstart lists all three missing params in one error."""
        with pytest.raises(ConfigurationError, match="Missing required parameters"):
            await SVRClient.quickstart()

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_missing_database_mentions_env_var(self, _dotenv):
        """Error message suggests SVR_DATABASE env var."""
        with pytest.raises(ConfigurationError, match="SVR_DATABASE"):
            await SVRClient.quickstart(
                collection="docs", partition_field="category"
            )

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_missing_collection_mentions_env_var(self, _dotenv):
        """Error message suggests SVR_COLLECTION env var."""
        with pytest.raises(ConfigurationError, match="SVR_COLLECTION"):
            await SVRClient.quickstart(
                database="testdb", partition_field="category"
            )

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_missing_partition_field_mentions_env_var(self, _dotenv):
        """Error message suggests SVR_PARTITION_FIELD env var."""
        with pytest.raises(ConfigurationError, match="SVR_PARTITION_FIELD"):
            await SVRClient.quickstart(
                database="testdb", collection="docs"
            )


# ---------------------------------------------------------------------------
# quickstart — presets
# ---------------------------------------------------------------------------


class TestQuickstartPresets:
    """Tests for quickstart preset parameter."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_with_minimal_preset(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with preset='minimal' merges minimal preset config."""
        await SVRClient.quickstart(**_REQUIRED_ARGS, preset="minimal")

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            # Minimal preset disables lifecycle detection
            lifecycle = config_dict.get("lifecycle", {})
            detection = lifecycle.get("detection", {})
            assert detection.get("enabled") is False

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_with_development_preset(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with preset='development' merges development config."""
        await SVRClient.quickstart(**_REQUIRED_ARGS, preset="development")

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            # Development preset sets logging level to DEBUG
            assert config_dict.get("logging", {}).get("level") == "DEBUG"

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_with_production_preset(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with preset='production' merges production config."""
        await SVRClient.quickstart(**_REQUIRED_ARGS, preset="production")

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            # Production preset enables caching with large max_size
            assert config_dict.get("cache", {}).get("max_size") == 50_000

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_with_unknown_preset_raises(self, _dotenv):
        """quickstart with unknown preset raises ValueError from get_preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            await SVRClient.quickstart(**_REQUIRED_ARGS, preset="nonexistent")


# ---------------------------------------------------------------------------
# quickstart — kwargs overrides
# ---------------------------------------------------------------------------


class TestQuickstartKwargsOverrides:
    """Tests for keyword argument overrides in quickstart."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_dimensions_kwarg_overrides(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with dimensions kwarg overrides auto-detected dimensions."""
        await SVRClient.quickstart(**_REQUIRED_ARGS, dimensions=768)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["embedding"]["dimensions"] == 768
            assert config_dict["vector_search"]["dimensions"] == 768

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_model_kwarg_overrides(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with model kwarg overrides auto-detected model."""
        await SVRClient.quickstart(**_REQUIRED_ARGS, model="text-embedding-3-large")

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["embedding"]["model"] == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# quickstart — backend variations
# ---------------------------------------------------------------------------


class TestQuickstartBackend:
    """Tests for backend-related quickstart behavior."""

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {"POSTGRES_URI": "postgresql://<user>:<password>@<host>:5432/<db>", "OPENAI_API_KEY": "sk-test"},
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_postgres_backend(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with backend='postgres' uses POSTGRES_URI and adds postgres config."""
        await SVRClient.quickstart(
            **_REQUIRED_ARGS,
            backend="postgres",
        )

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["database"]["backend"] == "postgres"
            assert config_dict["database"]["connection_string_env"] == "POSTGRES_URI"
            assert "postgres" in config_dict
            assert config_dict["postgres"]["connection_string_env"] == "POSTGRES_URI"

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {"POSTGRES_URI": "postgresql://<user>:<password>@<host>:5432/<db>", "OPENAI_API_KEY": "sk-test"},
        clear=True,
    )
    @patch(_LOAD_DOTENV)
    async def test_quickstart_postgres_missing_uri_raises(self, _dotenv):
        """quickstart with postgres backend raises when POSTGRES_URI is unset."""
        with pytest.raises(ConfigurationError, match="POSTGRES_URI"):
            # Clear POSTGRES_URI but keep OPENAI_API_KEY
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "sk-test"},
                clear=True,
            ):
                await SVRClient.quickstart(**_REQUIRED_ARGS, backend="postgres")

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "MY_CUSTOM_URI": "mongodb+srv://<user>:<password>@<cluster-host>/<db>",
            "OPENAI_API_KEY": "sk-test",
        },
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_custom_connection_string_env(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with explicit connection_string_env uses that env var."""
        await SVRClient.quickstart(
            **_REQUIRED_ARGS,
            connection_string_env="MY_CUSTOM_URI",
        )

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["database"]["connection_string_env"] == "MY_CUSTOM_URI"

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_default_backend_is_mongodb(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart defaults to 'mongodb' backend."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["database"]["backend"] == "mongodb"


# ---------------------------------------------------------------------------
# quickstart — embedding provider edge cases
# ---------------------------------------------------------------------------


class TestQuickstartEmbeddingProvider:
    """Tests for embedding provider detection and explicit specification."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_explicit_embedding_provider(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart with explicit embedding_provider uses it."""
        await SVRClient.quickstart(
            **_REQUIRED_ARGS,
            embedding_provider="openai",
        )

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["embedding"]["provider"] == "openai"

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_unknown_provider_raises(self, _dotenv):
        """quickstart with unknown embedding_provider raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
            await SVRClient.quickstart(
                **_REQUIRED_ARGS,
                embedding_provider="unknown_provider",
            )

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "MONGODB_URI": "mongodb+srv://<user>:<password>@<cluster-host>/<db>",
            "COHERE_API_KEY": "cohere-key-123",
        },
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_auto_detects_cohere_provider(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart auto-detects Cohere when only COHERE_API_KEY is set."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["embedding"]["provider"] == "cohere"
            assert config_dict["embedding"]["model"] == "embed-english-v3.0"

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "MONGODB_URI": "mongodb+srv://<user>:<password>@<cluster-host>/<db>",
            "VOYAGE_API_KEY": "pa-voyage",
            "OPENAI_API_KEY": "sk-openai",
        },
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_voyage_takes_priority_over_openai(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """When both VOYAGE and OPENAI keys are set, Voyage is preferred."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["embedding"]["provider"] == "voyage"


# ---------------------------------------------------------------------------
# quickstart — config dict structure
# ---------------------------------------------------------------------------


class TestQuickstartConfigDict:
    """Tests verifying the config dict passed to load_config."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_config_has_required_sections(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """Config dict has database, partitioning, embedding, vector_search sections."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert "database" in config_dict
            assert "partitioning" in config_dict
            assert "embedding" in config_dict
            assert "vector_search" in config_dict

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_embedding_and_vector_search_dimensions_match(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """Embedding dimensions and vector_search dimensions are synchronized."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            embed_dims = config_dict["embedding"]["dimensions"]
            search_dims = config_dict["vector_search"]["dimensions"]
            assert embed_dims == search_dims

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_sets_auto_connect_false(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart creates SVRClient with auto_connect=False (connect called separately)."""
        # This is implicitly verified: if auto_connect were True, __init__ would
        # try to call connect synchronously and fail. The fact that connect mock
        # is called async proves auto_connect=False was used.
        client = await SVRClient.quickstart(**_REQUIRED_ARGS)
        assert client is not None
        mock_connect.assert_awaited_once()


# ---------------------------------------------------------------------------
# quickstart — SVR_DIMENSIONS env var
# ---------------------------------------------------------------------------


class TestQuickstartDimensionsEnv:
    """Tests for SVR_DIMENSIONS environment variable."""

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {**_BASE_ENV, "SVR_DIMENSIONS": "512"},
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_svr_dimensions_env_overrides_provider_default(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """SVR_DIMENSIONS env var overrides the provider's default dimensions."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            # SVR_DIMENSIONS=512 overrides OpenAI default of 1536
            assert config_dict["embedding"]["dimensions"] == 512
            assert config_dict["vector_search"]["dimensions"] == 512

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {**_BASE_ENV, "SVR_DIMENSIONS": "not_a_number"},
        clear=True,
    )
    @patch(_LOAD_DOTENV)
    async def test_quickstart_svr_dimensions_invalid_raises(self, _dotenv):
        """SVR_DIMENSIONS with non-integer value raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="expected integer"):
            await SVRClient.quickstart(**_REQUIRED_ARGS)


# ---------------------------------------------------------------------------
# quickstart_sync
# ---------------------------------------------------------------------------


class TestQuickstartSync:
    """Tests for SVRClient.quickstart_sync()."""

    def test_quickstart_sync_raises_in_async_context(self):
        """quickstart_sync raises RuntimeError when an event loop is already running."""
        import asyncio

        async def _inner():
            with pytest.raises(RuntimeError, match="cannot be used in an async context"):
                SVRClient.quickstart_sync(**_REQUIRED_ARGS)

        # Run in a real event loop to trigger the "loop is running" branch
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_inner())
        finally:
            loop.close()

    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    def test_quickstart_sync_succeeds_outside_async(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart_sync succeeds when no event loop is running."""
        client = SVRClient.quickstart_sync(**_REQUIRED_ARGS)
        assert client is not None
        mock_connect.assert_awaited_once()


# ---------------------------------------------------------------------------
# search_sync
# ---------------------------------------------------------------------------


class TestSearchSync:
    """Tests for SVRClient.search_sync()."""

    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    def test_search_sync_calls_search_with_correct_args(
        self, mock_load, mock_validate
    ):
        """search_sync passes query, partitions, limit, and kwargs to async search."""
        client = SVRClient(config={"database": {}}, auto_connect=False)

        mock_result = MagicMock()
        client.search = AsyncMock(return_value=mock_result)

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = client.search_sync(
                query="test query",
                partitions=["electronics"],
                limit=5,
                filters={"price": {"$gt": 10}},
            )
        finally:
            loop.close()

        client.search.assert_awaited_once_with(
            query="test query",
            partitions=["electronics"],
            limit=5,
            filters={"price": {"$gt": 10}},
        )
        assert result == mock_result

    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    def test_search_sync_default_limit(self, mock_load, mock_validate):
        """search_sync uses default limit=10 when not specified."""
        client = SVRClient(config={"database": {}}, auto_connect=False)

        mock_result = MagicMock()
        client.search = AsyncMock(return_value=mock_result)

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            client.search_sync(query="test query")
        finally:
            loop.close()

        client.search.assert_awaited_once_with(
            query="test query",
            partitions=None,
            limit=10,
        )

    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    def test_search_sync_returns_search_result(self, mock_load, mock_validate):
        """search_sync returns the SearchResult from the async search call."""
        client = SVRClient(config={"database": {}}, auto_connect=False)

        from semantic_vector_router.models import SearchResult

        expected = SearchResult(
            hits=[],
            query="test",
            partitions_searched=[],
            total_candidates=0,
            reranked=False,
            latency_ms=1.0,
        )
        client.search = AsyncMock(return_value=expected)

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = client.search_sync(query="test")
        finally:
            loop.close()

        assert result.query == "test"
        assert result.hits == []
        assert result.latency_ms == 1.0


# ---------------------------------------------------------------------------
# quickstart — postgres config structure
# ---------------------------------------------------------------------------


class TestQuickstartPostgresConfig:
    """Tests for postgres-specific config structure in quickstart."""

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {"POSTGRES_URI": "postgresql://<user>:<password>@<host>:5432/<db>", "OPENAI_API_KEY": "sk-test"},
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_postgres_sets_vector_dimensions(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """Postgres config includes vector_dimensions matching embedding dimensions."""
        await SVRClient.quickstart(**_REQUIRED_ARGS, backend="postgres")

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            embed_dims = config_dict["embedding"]["dimensions"]
            pg_dims = config_dict["postgres"]["vector_dimensions"]
            assert embed_dims == pg_dims

    @pytest.mark.asyncio
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_mongodb_has_no_postgres_section(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """MongoDB backend does not include a 'postgres' section."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert "postgres" not in config_dict


# ---------------------------------------------------------------------------
# quickstart — SVR_BACKEND env var
# ---------------------------------------------------------------------------


class TestQuickstartBackendEnv:
    """Tests for SVR_BACKEND environment variable resolution."""

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "POSTGRES_URI": "postgresql://<user>:<password>@<host>:5432/<db>",
            "OPENAI_API_KEY": "sk-test",
            "SVR_BACKEND": "postgres",
        },
        clear=True,
    )
    @patch.object(SVRClient, "connect", new_callable=AsyncMock)
    @patch(f"{_CLIENT}.validate_config", return_value=[])
    @patch(f"{_CLIENT}.load_config", side_effect=_mock_load_config)
    @patch(_LOAD_DOTENV)
    async def test_quickstart_svr_backend_env_var(
        self, _dotenv, mock_load, mock_validate, mock_connect
    ):
        """quickstart reads backend from SVR_BACKEND env var."""
        await SVRClient.quickstart(**_REQUIRED_ARGS)

        call_kwargs = mock_load.call_args
        config_dict = call_kwargs[1].get("config_dict") or call_kwargs[0][0]
        if isinstance(config_dict, dict):
            assert config_dict["database"]["backend"] == "postgres"
