"""Unit tests for semantic_vector_router.presets module."""

import pytest

from semantic_vector_router.presets import get_preset


class TestGetPresetMinimal:
    """Tests for the 'minimal' preset."""

    def test_minimal_returns_dict(self):
        result = get_preset("minimal")
        assert isinstance(result, dict)

    def test_minimal_has_expected_top_level_keys(self):
        result = get_preset("minimal")
        expected_keys = {
            "reranking",
            "lifecycle",
            "metrics",
            "cache",
            "rate_limiting",
            "scheduler",
            "events",
        }
        assert set(result.keys()) == expected_keys

    def test_minimal_disables_reranking(self):
        result = get_preset("minimal")
        assert result["reranking"]["enabled"] is False

    def test_minimal_disables_metrics(self):
        result = get_preset("minimal")
        assert result["metrics"]["enabled"] is False

    def test_minimal_disables_cache(self):
        result = get_preset("minimal")
        assert result["cache"]["enabled"] is False

    def test_minimal_disables_rate_limiting(self):
        result = get_preset("minimal")
        assert result["rate_limiting"]["enabled"] is False

    def test_minimal_disables_scheduler(self):
        result = get_preset("minimal")
        assert result["scheduler"]["enabled"] is False

    def test_minimal_disables_events(self):
        result = get_preset("minimal")
        assert result["events"]["enabled"] is False

    def test_minimal_disables_lifecycle_auto_provision(self):
        result = get_preset("minimal")
        assert result["lifecycle"]["auto_provision"] is False

    def test_minimal_disables_lifecycle_change_stream(self):
        result = get_preset("minimal")
        assert result["lifecycle"]["change_stream_enabled"] is False

    def test_minimal_disables_lifecycle_detection(self):
        result = get_preset("minimal")
        assert result["lifecycle"]["detection"]["enabled"] is False


class TestGetPresetProduction:
    """Tests for the 'production' preset."""

    def test_production_returns_dict(self):
        result = get_preset("production")
        assert isinstance(result, dict)

    def test_production_has_expected_top_level_keys(self):
        result = get_preset("production")
        expected_keys = {
            "reranking",
            "lifecycle",
            "resilience",
            "metrics",
            "cache",
            "logging",
        }
        assert set(result.keys()) == expected_keys

    def test_production_enables_reranking(self):
        result = get_preset("production")
        assert result["reranking"]["enabled"] is True

    def test_production_enables_metrics(self):
        result = get_preset("production")
        assert result["metrics"]["enabled"] is True

    def test_production_enables_cache(self):
        result = get_preset("production")
        assert result["cache"]["enabled"] is True

    def test_production_cache_max_size(self):
        result = get_preset("production")
        assert result["cache"]["max_size"] == 50_000

    def test_production_cache_ttl(self):
        result = get_preset("production")
        assert result["cache"]["ttl_seconds"] == 3600

    def test_production_resilience_max_retry_attempts(self):
        result = get_preset("production")
        assert result["resilience"]["max_retry_attempts"] == 3

    def test_production_resilience_health_check_interval(self):
        result = get_preset("production")
        assert result["resilience"]["health_check_interval_s"] == 30

    def test_production_enables_lifecycle_auto_provision(self):
        result = get_preset("production")
        assert result["lifecycle"]["auto_provision"] is True

    def test_production_enables_lifecycle_change_stream(self):
        result = get_preset("production")
        assert result["lifecycle"]["change_stream_enabled"] is True

    def test_production_enables_lifecycle_detection(self):
        result = get_preset("production")
        assert result["lifecycle"]["detection"]["enabled"] is True

    def test_production_logging_json_format(self):
        result = get_preset("production")
        assert result["logging"]["json_format"] is True

    def test_production_logging_level_warning(self):
        result = get_preset("production")
        assert result["logging"]["level"] == "WARNING"


class TestGetPresetDevelopment:
    """Tests for the 'development' preset."""

    def test_development_returns_dict(self):
        result = get_preset("development")
        assert isinstance(result, dict)

    def test_development_has_expected_top_level_keys(self):
        result = get_preset("development")
        expected_keys = {
            "reranking",
            "logging",
            "metrics",
            "cache",
            "resilience",
            "rate_limiting",
            "ingestion",
        }
        assert set(result.keys()) == expected_keys

    def test_development_debug_logging_level(self):
        result = get_preset("development")
        assert result["logging"]["level"] == "DEBUG"

    def test_development_logs_query_text(self):
        result = get_preset("development")
        assert result["logging"]["log_query_text"] is True

    def test_development_enables_metrics(self):
        result = get_preset("development")
        assert result["metrics"]["enabled"] is True

    def test_development_includes_query_tags_in_metrics(self):
        result = get_preset("development")
        assert result["metrics"]["include_query_tags"] is True

    def test_development_cache_enabled(self):
        result = get_preset("development")
        assert result["cache"]["enabled"] is True

    def test_development_cache_max_size(self):
        result = get_preset("development")
        assert result["cache"]["max_size"] == 1000

    def test_development_cache_ttl(self):
        result = get_preset("development")
        assert result["cache"]["ttl_seconds"] == 300

    def test_development_low_batch_size(self):
        result = get_preset("development")
        assert result["ingestion"]["batch_size"] == 10

    def test_development_low_write_batch_size(self):
        result = get_preset("development")
        assert result["ingestion"]["write_batch_size"] == 50

    def test_development_disables_reranking(self):
        result = get_preset("development")
        assert result["reranking"]["enabled"] is False

    def test_development_disables_rate_limiting(self):
        result = get_preset("development")
        assert result["rate_limiting"]["enabled"] is False

    def test_development_resilience_max_retry_one(self):
        result = get_preset("development")
        assert result["resilience"]["max_retry_attempts"] == 1


class TestGetPresetUnknown:
    """Tests for unknown preset names."""

    def test_unknown_preset_raises_value_error(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent")

    def test_unknown_preset_error_message_contains_name(self):
        with pytest.raises(ValueError, match="Unknown preset: 'nonexistent'"):
            get_preset("nonexistent")

    def test_unknown_preset_error_lists_available_presets(self):
        with pytest.raises(ValueError, match="Available: development, minimal, production"):
            get_preset("nonexistent")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown preset: ''"):
            get_preset("")

    def test_case_sensitive_raises_for_uppercase(self):
        with pytest.raises(ValueError):
            get_preset("Minimal")

    def test_case_sensitive_raises_for_all_caps(self):
        with pytest.raises(ValueError):
            get_preset("PRODUCTION")


class TestGetPresetIsolation:
    """Tests that each call returns independent data."""

    def test_minimal_returns_new_dict_each_call(self):
        first = get_preset("minimal")
        second = get_preset("minimal")
        assert first is not second

    def test_production_returns_new_dict_each_call(self):
        first = get_preset("production")
        second = get_preset("production")
        assert first is not second

    def test_development_returns_new_dict_each_call(self):
        first = get_preset("development")
        second = get_preset("development")
        assert first is not second

    def test_mutating_returned_dict_does_not_affect_next_call(self):
        first = get_preset("minimal")
        first["reranking"]["enabled"] = True
        first["extra_key"] = "injected"
        second = get_preset("minimal")
        assert second["reranking"]["enabled"] is False
        assert "extra_key" not in second

    def test_mutating_nested_dict_does_not_affect_next_call(self):
        first = get_preset("production")
        first["cache"]["max_size"] = 999
        first["lifecycle"]["auto_provision"] = False
        second = get_preset("production")
        assert second["cache"]["max_size"] == 50_000
        assert second["lifecycle"]["auto_provision"] is True


class TestGetPresetNestedStructure:
    """Tests that nested structures are correct dicts, not other types."""

    @pytest.mark.parametrize("preset_name", ["minimal", "production", "development"])
    def test_all_top_level_values_are_dicts(self, preset_name):
        result = get_preset(preset_name)
        for key, value in result.items():
            assert isinstance(value, dict), (
                f"Preset '{preset_name}', key '{key}' should be a dict, got {type(value)}"
            )

    def test_minimal_lifecycle_detection_is_nested_dict(self):
        result = get_preset("minimal")
        assert isinstance(result["lifecycle"]["detection"], dict)
        assert "enabled" in result["lifecycle"]["detection"]

    def test_production_lifecycle_detection_is_nested_dict(self):
        result = get_preset("production")
        assert isinstance(result["lifecycle"]["detection"], dict)
        assert "enabled" in result["lifecycle"]["detection"]

    @pytest.mark.parametrize("preset_name", ["minimal", "production", "development"])
    def test_reranking_has_enabled_key(self, preset_name):
        result = get_preset(preset_name)
        assert "enabled" in result["reranking"]

    @pytest.mark.parametrize("preset_name", ["production", "development"])
    def test_cache_has_full_config(self, preset_name):
        result = get_preset(preset_name)
        cache = result["cache"]
        assert "enabled" in cache
        assert "max_size" in cache
        assert "ttl_seconds" in cache


class TestGetPresetCrossPresetComparison:
    """Tests comparing values across different presets."""

    def test_production_cache_larger_than_development(self):
        prod = get_preset("production")
        dev = get_preset("development")
        assert prod["cache"]["max_size"] > dev["cache"]["max_size"]

    def test_production_cache_ttl_longer_than_development(self):
        prod = get_preset("production")
        dev = get_preset("development")
        assert prod["cache"]["ttl_seconds"] > dev["cache"]["ttl_seconds"]

    def test_production_more_retries_than_development(self):
        prod = get_preset("production")
        dev = get_preset("development")
        assert prod["resilience"]["max_retry_attempts"] > dev["resilience"]["max_retry_attempts"]

    def test_minimal_has_most_disabled_features(self):
        minimal = get_preset("minimal")
        disabled_count = sum(
            1
            for key in ["reranking", "metrics", "cache", "rate_limiting", "scheduler", "events"]
            if minimal[key].get("enabled") is False
        )
        assert disabled_count == 6

    def test_development_logging_more_verbose_than_production(self):
        prod = get_preset("production")
        dev = get_preset("development")
        log_levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        assert log_levels[dev["logging"]["level"]] < log_levels[prod["logging"]["level"]]
