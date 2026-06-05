"""Configuration validation for the Semantic Vector Router."""

import os
from typing import TYPE_CHECKING

from semantic_vector_router.exceptions import ConfigurationError

if TYPE_CHECKING:
    from semantic_vector_router.models import SVRConfig


def validate_config(config: "SVRConfig") -> list[str]:
    """Validate configuration and return list of warnings.

    Args:
        config: Configuration to validate.

    Returns:
        List of warning messages (empty if all is well).

    Raises:
        ConfigurationError: If configuration has critical issues.
    """
    warnings: list[str] = []
    _validate_embedding(config, warnings)
    _validate_storage(config, warnings)
    _validate_resilience(config, warnings)
    _validate_detection(config, warnings)
    _validate_repartition(config, warnings)
    _validate_auto_split(config, warnings)
    _validate_fields_mode(config, warnings)
    _validate_connection_pool(config, warnings)
    _validate_ingestion(config, warnings)
    _validate_rate_limiting(config, warnings)
    _validate_centroid(config, warnings)
    _validate_scheduler(config, warnings)
    _validate_events(config, warnings)
    return warnings


def _validate_embedding(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate embedding configuration."""
    from semantic_vector_router.models import (
        MongoDBIndexQuantization,
        VectorStorageFormat,
        VoyageQuantization,
    )

    # Check embedding mode consistency
    if config.embedding.mode == "auto" and config.embedding.provider not in [
        "atlas_voyage"
    ]:
        raise ConfigurationError(
            "Auto-embedding mode requires atlas_voyage provider"
        )

    # Check BYOM has valid provider
    if config.embedding.mode == "byom" and config.embedding.provider == "atlas_voyage":
        raise ConfigurationError(
            "BYOM mode cannot use atlas_voyage provider"
        )

    # Check dimensions match
    if config.embedding.dimensions != config.vector_search.dimensions:
        warnings.append(
            f"Embedding dimensions ({config.embedding.dimensions}) don't match "
            f"vector search dimensions ({config.vector_search.dimensions})"
        )

    # Check reranking provider has API key env var
    if config.reranking.enabled and not config.reranking.api_key_env:
        warnings.append(
            f"Reranking enabled but no API key env var specified for {config.reranking.provider}"
        )

    # Quantization compatibility validation
    storage_format = config.vector_storage.storage_format
    index_quant = config.vector_storage.index_quantization

    # Pre-quantized vectors (INT8, PACKED_BIT) must not use index-level quantization
    pre_quantized_formats = (
        VectorStorageFormat.BINDATA_INT8,
        VectorStorageFormat.BINDATA_PACKED_BIT,
    )
    if storage_format in pre_quantized_formats and index_quant != MongoDBIndexQuantization.NONE:
        raise ConfigurationError(
            f"Incompatible quantization: storage_format={storage_format.value} "
            f"is pre-quantized and cannot use index_quantization={index_quant.value}. "
            f"Pre-quantized vectors require index_quantization=none to avoid "
            f"double quantization."
        )

    # Voyage quantization output type should match storage format
    if config.embedding.provider == "voyage":
        voyage_quant = config.embedding.voyage_quantization
        is_int8_mismatch = (
            voyage_quant == VoyageQuantization.INT8
            and storage_format != VectorStorageFormat.BINDATA_INT8
        )
        if is_int8_mismatch:
            warnings.append(
                f"Voyage quantization is int8 but storage_format is {storage_format.value}. "
                f"Consider using bindata_int8 for optimal storage of pre-quantized vectors."
            )
        if voyage_quant in (VoyageQuantization.BINARY, VoyageQuantization.UBINARY):
            if storage_format != VectorStorageFormat.BINDATA_PACKED_BIT:
                warnings.append(
                    f"Voyage quantization is {voyage_quant.value} but storage_format is "
                    f"{storage_format.value}. Consider using bindata_packed_bit."
                )


def _validate_storage(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate vector storage configuration."""
    # Check separate storage has collection specified
    if (
        config.vector_storage.mode == "separate"
        and not config.vector_storage.embeddings_collection
    ):
        raise ConfigurationError(
            "Separate vector storage mode requires embeddings_collection to be set"
        )


def _validate_resilience(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate resilience configuration."""
    r = config.resilience
    if r.connection_timeout_ms <= 0:
        raise ConfigurationError(
            f"resilience.connection_timeout_ms must be > 0, got {r.connection_timeout_ms}"
        )
    if r.search_timeout_ms <= 0:
        raise ConfigurationError(
            f"resilience.search_timeout_ms must be > 0, got {r.search_timeout_ms}"
        )
    if r.max_retry_attempts < 0:
        raise ConfigurationError(
            f"resilience.max_retry_attempts must be >= 0, got {r.max_retry_attempts}"
        )
    if r.health_check_interval_s < 0:
        raise ConfigurationError(
            f"resilience.health_check_interval_s must be >= 0, got {r.health_check_interval_s}"
        )
    if r.search_timeout_ms < 5000:
        warnings.append(
            f"resilience.search_timeout_ms is very low ({r.search_timeout_ms}ms). "
            f"Vector searches may time out frequently."
        )


def _validate_detection(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate detection configuration."""
    d = config.lifecycle.detection
    if d.threshold_vectors <= 0:
        raise ConfigurationError(
            f"lifecycle.detection.threshold_vectors must be > 0, got {d.threshold_vectors}"
        )
    if d.min_threshold_vectors <= 0:
        raise ConfigurationError(
            f"lifecycle.detection.min_threshold_vectors must be > 0, got {d.min_threshold_vectors}"
        )
    if d.min_threshold_vectors >= d.threshold_vectors:
        raise ConfigurationError(
            f"lifecycle.detection.min_threshold_vectors ({d.min_threshold_vectors}) "
            f"must be < threshold_vectors ({d.threshold_vectors})"
        )
    if d.skew_ratio <= 1.0:
        raise ConfigurationError(
            f"lifecycle.detection.skew_ratio must be > 1.0, got {d.skew_ratio}"
        )
    if d.trend_window_days <= 0:
        raise ConfigurationError(
            f"lifecycle.detection.trend_window_days must be > 0, got {d.trend_window_days}"
        )


def _validate_repartition(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate repartition configuration."""
    rp = config.lifecycle.repartition
    if rp.index_wait_timeout_s <= 0:
        raise ConfigurationError(
            f"lifecycle.repartition.index_wait_timeout_s must be > 0, got {rp.index_wait_timeout_s}"
        )
    if rp.index_poll_interval_s <= 0:
        raise ConfigurationError(
            "lifecycle.repartition.index_poll_interval_s must be > 0, "
            f"got {rp.index_poll_interval_s}"
        )
    if rp.index_poll_interval_s >= rp.index_wait_timeout_s:
        raise ConfigurationError(
            f"lifecycle.repartition.index_poll_interval_s ({rp.index_poll_interval_s}) "
            f"must be < index_wait_timeout_s ({rp.index_wait_timeout_s})"
        )

    # Metadata connection string warning
    if config.lifecycle.metadata.connection_string_env:
        if not os.getenv(config.lifecycle.metadata.connection_string_env):
            warnings.append(
                f"lifecycle.metadata.connection_string_env is set to "
                f"'{config.lifecycle.metadata.connection_string_env}' but the "
                f"environment variable is not defined."
            )


def _validate_auto_split(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate auto-split configuration."""
    if config.lifecycle.auto_split and config.lifecycle.auto_split.enabled:
        split_config = config.lifecycle.auto_split
        if (
            split_config.split_strategy == "secondary_field"
            and not split_config.secondary_field
        ):
            raise ConfigurationError(
                "Secondary field split strategy requires secondary_field to be set"
            )
        if split_config.split_strategy == "time" and not split_config.time_field:
            raise ConfigurationError(
                "Time split strategy requires time_field to be set"
            )


def _validate_fields_mode(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate FIELDS mode configuration."""
    from semantic_vector_router.models import IndexLocation

    if config.vector_storage.index_on == IndexLocation.FIELDS:
        fields_count = sum(
            1 for p in config.partitions.registry.values()
            if p.index_location == IndexLocation.FIELDS
        )
        if fields_count > 50:
            raise ConfigurationError(
                f"FIELDS mode supports at most 50 partitions (Atlas 64-index limit). "
                f"Current count: {fields_count}. Switch to VIEWS mode for additional partitions."
            )
        if fields_count > 40:
            warnings.append(
                f"FIELDS mode has {fields_count}/50 partitions. "
                f"Consider switching to VIEWS mode before reaching the limit."
            )


def _validate_connection_pool(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate connection pool configuration."""
    db = config.database
    if db.max_pool_size <= 0:
        raise ConfigurationError(
            f"database.max_pool_size must be > 0, got {db.max_pool_size}"
        )
    if db.max_pool_size < 10:
        warnings.append(
            f"database.max_pool_size is {db.max_pool_size}, which is low for production. "
            f"Consider increasing to at least 10."
        )
    if db.min_pool_size < 0:
        raise ConfigurationError(
            f"database.min_pool_size must be >= 0, got {db.min_pool_size}"
        )
    if db.min_pool_size > db.max_pool_size:
        raise ConfigurationError(
            f"database.min_pool_size ({db.min_pool_size}) must be <= "
            f"max_pool_size ({db.max_pool_size})"
        )
    if db.max_idle_time_ms < 0:
        raise ConfigurationError(
            f"database.max_idle_time_ms must be >= 0, got {db.max_idle_time_ms}"
        )
    if db.wait_queue_timeout_ms < 0:
        raise ConfigurationError(
            f"database.wait_queue_timeout_ms must be >= 0, got {db.wait_queue_timeout_ms}"
        )


def _validate_ingestion(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate ingestion configuration."""
    ic = config.ingestion
    if ic.batch_size <= 0:
        raise ConfigurationError(
            f"ingestion.batch_size must be > 0, got {ic.batch_size}"
        )
    if ic.batch_size > 2048:
        warnings.append(
            f"ingestion.batch_size is {ic.batch_size}, which exceeds most "
            f"embedding provider batch limits. Consider reducing to <= 128."
        )
    if ic.write_batch_size <= 0:
        raise ConfigurationError(
            f"ingestion.write_batch_size must be > 0, got {ic.write_batch_size}"
        )
    if not ic.text_fields:
        warnings.append(
            "ingestion.text_fields is empty. Documents will have no text to embed."
        )


def _validate_rate_limiting(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate rate limiting configuration."""
    rl = config.rate_limiting
    if rl.default_tokens_per_second <= 0:
        raise ConfigurationError(
            f"rate_limiting.default_tokens_per_second must be > 0, "
            f"got {rl.default_tokens_per_second}"
        )
    if rl.default_burst <= 0:
        raise ConfigurationError(
            f"rate_limiting.default_burst must be > 0, got {rl.default_burst}"
        )
    for provider_name, provider_limit in rl.providers.items():
        if provider_limit.tokens_per_second <= 0:
            raise ConfigurationError(
                f"rate_limiting.providers.{provider_name}.tokens_per_second must be > 0"
            )
        if provider_limit.burst <= 0:
            raise ConfigurationError(
                f"rate_limiting.providers.{provider_name}.burst must be > 0"
            )


def _validate_centroid(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate centroid routing configuration."""
    cr = config.routing.centroid_routing
    if cr.relative_threshold <= 0 or cr.relative_threshold > 1:
        raise ConfigurationError(
            f"routing.centroid_routing.relative_threshold must be in (0, 1], "
            f"got {cr.relative_threshold}"
        )
    if cr.min_score < 0 or cr.min_score >= 1:
        raise ConfigurationError(
            f"routing.centroid_routing.min_score must be in [0, 1), "
            f"got {cr.min_score}"
        )
    if cr.min_score >= cr.relative_threshold:
        warnings.append(
            f"routing.centroid_routing.min_score ({cr.min_score}) >= "
            f"relative_threshold ({cr.relative_threshold}). "
            f"min_score should be lower than relative_threshold for proper pruning."
        )
    if cr.sample_size <= 0:
        raise ConfigurationError(
            f"routing.centroid_routing.sample_size must be > 0, got {cr.sample_size}"
        )
    if cr.centroid_ttl_seconds <= 0:
        raise ConfigurationError(
            f"routing.centroid_routing.centroid_ttl_seconds must be > 0, "
            f"got {cr.centroid_ttl_seconds}"
        )
    if cr.registry_ttl_seconds <= 0:
        raise ConfigurationError(
            f"routing.centroid_routing.registry_ttl_seconds must be > 0, "
            f"got {cr.registry_ttl_seconds}"
        )
    if cr.max_probe_partitions <= 0:
        raise ConfigurationError(
            f"routing.centroid_routing.max_probe_partitions must be > 0, "
            f"got {cr.max_probe_partitions}"
        )


def _validate_scheduler(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate scheduler configuration."""
    sc = config.scheduler
    if sc.enabled:
        if sc.tick_interval_seconds <= 0:
            raise ConfigurationError(
                f"scheduler.tick_interval_seconds must be > 0, got {sc.tick_interval_seconds}"
            )
        # Validate interval strings
        from semantic_vector_router.scheduler.interval import parse_interval

        for field_name in [
            "detection_interval",
            "centroid_refresh_interval",
            "count_update_interval",
            "repartition_check_interval",
            "index_health_interval",
        ]:
            val = getattr(sc, field_name, None)
            if val is not None:
                try:
                    parsed = parse_interval(val)
                    if parsed <= 0:
                        raise ConfigurationError(
                            f"scheduler.{field_name} must be positive"
                        )
                except ValueError as e:
                    raise ConfigurationError(
                        f"scheduler.{field_name}: {e}"
                    )

        # Validate maintenance window
        if sc.maintenance_window is not None:
            mw = sc.maintenance_window
            start_h = mw.allowed_hours.get("start", 0)
            end_h = mw.allowed_hours.get("end", 24)
            if not (0 <= start_h <= 24):
                raise ConfigurationError(
                    f"scheduler.maintenance_window.allowed_hours.start must be 0-24, "
                    f"got {start_h}"
                )
            if not (0 <= end_h <= 24):
                raise ConfigurationError(
                    f"scheduler.maintenance_window.allowed_hours.end must be 0-24, "
                    f"got {end_h}"
                )
            valid_days = {
                "monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday",
            }
            for day in mw.allowed_days:
                if day.lower() not in valid_days:
                    raise ConfigurationError(
                        f"scheduler.maintenance_window.allowed_days: "
                        f"invalid day '{day}'"
                    )
            # Validate timezone
            try:
                import zoneinfo
                zoneinfo.ZoneInfo(mw.timezone)
            except (KeyError, Exception):
                raise ConfigurationError(
                    f"scheduler.maintenance_window.timezone: "
                    f"invalid timezone '{mw.timezone}'"
                )


def _validate_events(config: "SVRConfig", warnings: list[str]) -> None:
    """Validate events configuration."""
    ev = config.events
    if ev.event_retention_days <= 0:
        raise ConfigurationError(
            f"events.event_retention_days must be > 0, got {ev.event_retention_days}"
        )
    for i, wh in enumerate(ev.webhooks):
        if not wh.url:
            raise ConfigurationError(
                f"events.webhooks[{i}].url must not be empty"
            )
        if not wh.url.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"events.webhooks[{i}].url must start with http:// or https://, "
                f"got '{wh.url}'"
            )
        if wh.timeout_seconds <= 0:
            raise ConfigurationError(
                f"events.webhooks[{i}].timeout_seconds must be > 0"
            )
        if wh.retry_count < 0:
            raise ConfigurationError(
                f"events.webhooks[{i}].retry_count must be >= 0"
            )
