"""Configuration management for Semantic Vector Router."""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv

from semantic_vector_router.exceptions import ConfigurationError
from semantic_vector_router.models import SVRConfig
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)

# Default config file locations
DEFAULT_CONFIG_PATHS = [
    ".svr/config.json",
    "svr.config.json",
    ".svr.json",
]


def find_config_file(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find the configuration file by searching standard locations.

    Args:
        start_path: Starting directory to search from. Defaults to current directory.

    Returns:
        Path to the config file if found, None otherwise.
    """
    start_path = start_path or Path.cwd()

    # Search in current directory and parents
    current = start_path
    while current != current.parent:
        for config_name in DEFAULT_CONFIG_PATHS:
            config_path = current / config_name
            if config_path.exists():
                return config_path
        current = current.parent

    return None


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[dict[str, Any]] = None,
    load_env: bool = True,
) -> SVRConfig:
    """Load configuration from file or dictionary.

    Args:
        config_path: Path to config file. If None, searches standard locations.
        config_dict: Configuration dictionary. Takes precedence over config_path.
        load_env: Whether to load .env file for environment variables.

    Returns:
        Validated SVRConfig object.

    Raises:
        ConfigurationError: If configuration is invalid or not found.
    """
    if load_env:
        load_dotenv()

    if config_dict is not None:
        logger.debug("Loading configuration from dictionary")
        try:
            return SVRConfig.model_validate(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}")

    # Find config file
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
    else:
        found = find_config_file()
        if found is None:
            raise ConfigurationError(
                "No configuration file found. Run 'svr init' to create one."
            )
        path = found

    logger.debug(f"Loading configuration from {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {e}")
    except OSError as e:
        raise ConfigurationError(f"Error reading config file: {e}")

    try:
        return SVRConfig.model_validate(data)
    except Exception as e:
        raise ConfigurationError(f"Invalid configuration: {e}")


def save_config(
    config: SVRConfig,
    config_path: Optional[Union[str, Path]] = None,
    create_dir: bool = True,
) -> Path:
    """Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Path to save to. Defaults to .svr/config.json.
        create_dir: Whether to create the directory if it doesn't exist.

    Returns:
        Path where configuration was saved.

    Raises:
        ConfigurationError: If save fails.
    """
    if config_path is None:
        config_path = Path.cwd() / ".svr" / "config.json"
    else:
        config_path = Path(config_path)

    if create_dir:
        config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, "w") as f:
            json.dump(
                config.model_dump(mode="json", exclude_none=True),
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Configuration saved to {config_path}")
        return config_path
    except OSError as e:
        raise ConfigurationError(f"Error saving config file: {e}")


def resolve_env_var(env_var_name: str, required: bool = True) -> Optional[str]:
    """Resolve an environment variable.

    Args:
        env_var_name: Name of the environment variable.
        required: Whether the variable is required.

    Returns:
        Value of the environment variable, or None if not required and not set.

    Raises:
        ConfigurationError: If required variable is not set.
    """
    value = os.environ.get(env_var_name)
    if value is None and required:
        raise ConfigurationError(
            f"Required environment variable not set: {env_var_name}"
        )
    return value


def get_connection_string(config: SVRConfig) -> str:
    """Get the database connection string from config.

    Args:
        config: SVR configuration.

    Returns:
        Connection string.

    Raises:
        ConfigurationError: If connection string env var is not set.
    """
    env_var = config.database.connection_string_env
    value = resolve_env_var(env_var)
    if value is None:
        raise ConfigurationError(
            f"Database connection string not found in environment variable: {env_var}"
        )
    return value


def get_api_key(env_var_name: Optional[str], provider: str) -> Optional[str]:
    """Get an API key from environment.

    Args:
        env_var_name: Name of the environment variable.
        provider: Name of the provider (for error messages).

    Returns:
        API key value, or None if env_var_name is None.

    Raises:
        ConfigurationError: If env var is specified but not set.
    """
    if env_var_name is None:
        return None

    value = os.environ.get(env_var_name)
    if value is None:
        raise ConfigurationError(
            f"API key for {provider} not found in environment variable: {env_var_name}"
        )
    return value


# Re-export validate_config from config_validators for backward compatibility
from semantic_vector_router.config_validators import validate_config  # noqa: F401


# --- SVR env var support (Phase 15) ---

SVR_ENV_VARS = {
    "SVR_DATABASE": "Database name",
    "SVR_COLLECTION": "Source collection or table name",
    "SVR_PARTITION_FIELD": "Field to partition by",
    "SVR_BACKEND": "Database backend (mongodb or postgres)",
    "SVR_EMBEDDING_PROVIDER": "Embedding provider (openai, voyage, cohere)",
    "SVR_DIMENSIONS": "Embedding dimensions (integer)",
}


def detect_embedding_provider(
    explicit_provider: Optional[str] = None,
) -> tuple[str, str, int, Optional[str]]:
    """Auto-detect the best available embedding provider.

    Resolution order:
    1. Explicit provider name (if given)
    2. SVR_EMBEDDING_PROVIDER env var
    3. Auto-detect from available API keys (check in priority order)

    Returns:
        Tuple of (provider_name, model_name, dimensions, api_key_env_var).

    Raises:
        ConfigurationError: If no provider can be detected.
    """
    _PROVIDER_TABLE: list[tuple[str, str, str, int, str]] = [
        # (env_var_to_check, provider, model, dimensions, api_key_env)
        ("VOYAGE_API_KEY", "voyage", "voyage-3-lite", 1024, "VOYAGE_API_KEY"),
        ("OPENAI_API_KEY", "openai", "text-embedding-3-small", 1536, "OPENAI_API_KEY"),
        ("COHERE_API_KEY", "cohere", "embed-english-v3.0", 1024, "COHERE_API_KEY"),
    ]

    # 1. Explicit provider or env var
    provider_name = explicit_provider or os.environ.get("SVR_EMBEDDING_PROVIDER")

    if provider_name:
        for env_var, provider, model, dims, key_env in _PROVIDER_TABLE:
            if provider == provider_name:
                return (provider, model, dims, key_env)
        raise ConfigurationError(
            f"Unknown embedding provider: '{provider_name}'.\n\n"
            f"Supported providers: voyage, openai, cohere"
        )

    # 2. Auto-detect from available API keys
    for env_var, provider, model, dims, key_env in _PROVIDER_TABLE:
        if os.environ.get(env_var):
            logger.info(f"Auto-detected embedding provider: {provider} (from {env_var})")
            return (provider, model, dims, key_env)

    # 3. Nothing found
    raise ConfigurationError(
        "No embedding API key found.\n\n"
        "Set one of these environment variables:\n"
        "  export VOYAGE_API_KEY='pa-...'   (recommended — best search quality)\n"
        "  export OPENAI_API_KEY='sk-...'   (most common)\n"
        "  export COHERE_API_KEY='...'      (alternative)\n\n"
        "Or specify a provider explicitly:\n"
        "  svr = await SVRClient.quickstart(..., embedding_provider='openai')"
    )


def _parse_int_env(var_name: str) -> Optional[int]:
    """Parse an integer from an environment variable."""
    value = os.environ.get(var_name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ConfigurationError(
            f"Invalid value for {var_name}: '{value}' (expected integer)"
        )


def resolve_quickstart_params(
    database: Optional[str] = None,
    collection: Optional[str] = None,
    partition_field: Optional[str] = None,
    backend: Optional[str] = None,
    embedding_provider: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve quickstart parameters from args + env vars.

    Each parameter follows the resolution chain:
    1. Explicit argument (if not None)
    2. Corresponding SVR_* environment variable
    3. None (caller decides if required)

    Returns:
        Dict with resolved values. None values mean "not provided".
    """
    return {
        "database": database or os.environ.get("SVR_DATABASE"),
        "collection": collection or os.environ.get("SVR_COLLECTION"),
        "partition_field": partition_field or os.environ.get("SVR_PARTITION_FIELD"),
        "backend": backend or os.environ.get("SVR_BACKEND", "mongodb"),
        "embedding_provider": (
            embedding_provider or os.environ.get("SVR_EMBEDDING_PROVIDER")
        ),
        "dimensions": _parse_int_env("SVR_DIMENSIONS"),
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. Override values take precedence.

    Args:
        base: Base dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_default_config(
    database: str,
    source_collection: str,
    partition_field: str,
    connection_string_env: str = "MONGODB_URI",
) -> SVRConfig:
    """Create a default configuration with minimal required fields.

    Args:
        database: Database name.
        source_collection: Source collection name.
        partition_field: Field to partition by.
        connection_string_env: Environment variable for connection string.

    Returns:
        Default SVRConfig.
    """
    from semantic_vector_router.models import (
        DatabaseConfig,
        PartitioningConfig,
    )

    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env=connection_string_env,
            database=database,
            source_collection=source_collection,
        ),
        partitioning=PartitioningConfig(
            field=partition_field,
        ),
    )
