"""Backend factory — config-driven backend instantiation."""

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.exceptions import ConfigurationError
from semantic_vector_router.models.svr_config import SVRConfig


def create_backend(config: SVRConfig) -> BaseBackend:
    """Create a backend instance from configuration.

    Uses the ``config.database.backend`` field to determine which backend
    to instantiate. Currently only MongoDB is supported.

    Args:
        config: SVR configuration.

    Returns:
        Configured backend instance (not yet connected).

    Raises:
        ConfigurationError: If the backend type is unknown.
    """
    raw = config.database.backend
    backend_type = raw.value if hasattr(raw, "value") else str(raw)

    if backend_type == "mongodb":
        from semantic_vector_router.backends.mongodb import MongoDBBackend

        return MongoDBBackend(config)
    elif backend_type == "postgres":
        from semantic_vector_router.backends.postgres import PostgresBackend

        return PostgresBackend(config)
    else:
        raise ConfigurationError(
            f"Unknown backend type: '{backend_type}'. "
            f"Supported backends: mongodb, postgres"
        )
