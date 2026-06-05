"""Configuration presets for common SVR use cases."""

from typing import Any


def get_preset(name: str) -> dict[str, Any]:
    """Get a configuration preset by name.

    Available presets:
    - "minimal": Bare minimum — just search, no lifecycle
    - "production": Full features — lifecycle, metrics, caching, reranking
    - "development": Verbose logging, no rate limiting, small batches

    Args:
        name: Preset name.

    Returns:
        Config dict that can be merged with quickstart params.

    Raises:
        ValueError: If preset name is unknown.
    """
    presets = {
        "minimal": {
            "reranking": {"enabled": False},
            "lifecycle": {
                "auto_provision": False,
                "change_stream_enabled": False,
                "detection": {"enabled": False},
            },
            "metrics": {"enabled": False},
            "cache": {"enabled": False},
            "rate_limiting": {"enabled": False},
            "scheduler": {"enabled": False},
            "events": {"enabled": False},
        },
        "production": {
            "reranking": {"enabled": True},
            "lifecycle": {
                "auto_provision": True,
                "change_stream_enabled": True,
                "detection": {"enabled": True},
            },
            "resilience": {
                "max_retry_attempts": 3,
                "health_check_interval_s": 30,
            },
            "metrics": {"enabled": True},
            "cache": {"enabled": True, "max_size": 50_000, "ttl_seconds": 3600},
            "logging": {"json_format": True, "level": "WARNING"},
        },
        "development": {
            "reranking": {"enabled": False},
            "logging": {"level": "DEBUG", "log_query_text": True},
            "metrics": {"enabled": True, "include_query_tags": True},
            "cache": {"enabled": True, "max_size": 1000, "ttl_seconds": 300},
            "resilience": {"max_retry_attempts": 1},
            "rate_limiting": {"enabled": False},
            "ingestion": {"batch_size": 10, "write_batch_size": 50},
        },
    }

    if name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(f"Unknown preset: '{name}'. Available: {available}")

    return presets[name]
