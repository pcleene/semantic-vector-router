"""Tests for embedding detection, quickstart param resolution, and deep merge utilities."""

import pytest
from unittest.mock import patch

from semantic_vector_router.config import (
    detect_embedding_provider,
    resolve_quickstart_params,
    _parse_int_env,
    _deep_merge,
)
from semantic_vector_router.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Helper: build a dict of env vars, clearing all provider keys by default
# so tests are fully isolated from the real environment.
# ---------------------------------------------------------------------------
_ALL_PROVIDER_KEYS = [
    "VOYAGE_API_KEY",
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "SVR_EMBEDDING_PROVIDER",
    "SVR_DATABASE",
    "SVR_COLLECTION",
    "SVR_PARTITION_FIELD",
    "SVR_BACKEND",
    "SVR_DIMENSIONS",
]


def _clean_env(**overrides: str) -> dict[str, str]:
    """Return an env dict with all provider keys removed except those in *overrides*."""
    env = {k: v for k, v in __import__("os").environ.items() if k not in _ALL_PROVIDER_KEYS}
    env.update(overrides)
    return env


class TestDetectEmbeddingProvider:
    """Tests for detect_embedding_provider()."""

    # -- Auto-detection from API keys (priority order) ----------------------

    def test_voyage_key_present_returns_voyage_config(self):
        """Voyage key present -> returns voyage provider tuple."""
        with patch.dict("os.environ", _clean_env(VOYAGE_API_KEY="pa-test"), clear=True):
            provider, model, dims, key_env = detect_embedding_provider()
        assert provider == "voyage"
        assert model == "voyage-3-lite"
        assert dims == 1024
        assert key_env == "VOYAGE_API_KEY"

    def test_only_openai_key_returns_openai_config(self):
        """Only OpenAI key present -> returns openai provider tuple."""
        with patch.dict("os.environ", _clean_env(OPENAI_API_KEY="sk-test"), clear=True):
            provider, model, dims, key_env = detect_embedding_provider()
        assert provider == "openai"
        assert model == "text-embedding-3-small"
        assert dims == 1536
        assert key_env == "OPENAI_API_KEY"

    def test_only_cohere_key_returns_cohere_config(self):
        """Only Cohere key present -> returns cohere provider tuple."""
        with patch.dict("os.environ", _clean_env(COHERE_API_KEY="co-test"), clear=True):
            provider, model, dims, key_env = detect_embedding_provider()
        assert provider == "cohere"
        assert model == "embed-english-v3.0"
        assert dims == 1024
        assert key_env == "COHERE_API_KEY"

    def test_multiple_keys_uses_priority_order_voyage_first(self):
        """When Voyage + OpenAI keys both present, Voyage wins (higher priority)."""
        with patch.dict(
            "os.environ",
            _clean_env(VOYAGE_API_KEY="pa-test", OPENAI_API_KEY="sk-test"),
            clear=True,
        ):
            provider, model, dims, key_env = detect_embedding_provider()
        assert provider == "voyage"

    def test_multiple_keys_openai_over_cohere(self):
        """When OpenAI + Cohere keys present (no Voyage), OpenAI wins."""
        with patch.dict(
            "os.environ",
            _clean_env(OPENAI_API_KEY="sk-test", COHERE_API_KEY="co-test"),
            clear=True,
        ):
            provider, model, dims, key_env = detect_embedding_provider()
        assert provider == "openai"

    def test_all_three_keys_returns_voyage(self):
        """When all three keys are present, Voyage is selected (top priority)."""
        with patch.dict(
            "os.environ",
            _clean_env(
                VOYAGE_API_KEY="pa-test",
                OPENAI_API_KEY="sk-test",
                COHERE_API_KEY="co-test",
            ),
            clear=True,
        ):
            provider, *_ = detect_embedding_provider()
        assert provider == "voyage"

    # -- No keys at all -----------------------------------------------------

    def test_no_keys_raises_configuration_error(self):
        """No API keys set at all -> raises ConfigurationError with helpful message."""
        with patch.dict("os.environ", _clean_env(), clear=True):
            with pytest.raises(ConfigurationError, match="No embedding API key found"):
                detect_embedding_provider()

    # -- Explicit provider argument -----------------------------------------

    def test_explicit_provider_overrides_auto_detection(self):
        """Passing explicit_provider='openai' returns openai even when Voyage key is set."""
        with patch.dict("os.environ", _clean_env(VOYAGE_API_KEY="pa-test"), clear=True):
            provider, model, dims, key_env = detect_embedding_provider(
                explicit_provider="openai"
            )
        assert provider == "openai"
        assert model == "text-embedding-3-small"
        assert dims == 1536
        assert key_env == "OPENAI_API_KEY"

    def test_explicit_provider_cohere(self):
        """Explicit 'cohere' selects cohere regardless of other env vars."""
        with patch.dict("os.environ", _clean_env(), clear=True):
            provider, model, dims, key_env = detect_embedding_provider(
                explicit_provider="cohere"
            )
        assert provider == "cohere"
        assert model == "embed-english-v3.0"

    def test_explicit_unknown_provider_raises_error(self):
        """Unknown explicit provider name raises ConfigurationError."""
        with patch.dict("os.environ", _clean_env(), clear=True):
            with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
                detect_embedding_provider(explicit_provider="foobar")

    # -- SVR_EMBEDDING_PROVIDER env var -------------------------------------

    def test_svr_embedding_provider_env_var_selects_provider(self):
        """SVR_EMBEDDING_PROVIDER env var selects the named provider."""
        with patch.dict(
            "os.environ",
            _clean_env(SVR_EMBEDDING_PROVIDER="cohere"),
            clear=True,
        ):
            provider, model, dims, key_env = detect_embedding_provider()
        assert provider == "cohere"
        assert key_env == "COHERE_API_KEY"

    def test_svr_embedding_provider_env_var_unknown_raises_error(self):
        """SVR_EMBEDDING_PROVIDER set to unknown value raises ConfigurationError."""
        with patch.dict(
            "os.environ",
            _clean_env(SVR_EMBEDDING_PROVIDER="unknown_provider"),
            clear=True,
        ):
            with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
                detect_embedding_provider()

    def test_explicit_provider_overrides_env_var(self):
        """Explicit argument takes precedence over SVR_EMBEDDING_PROVIDER env var."""
        with patch.dict(
            "os.environ",
            _clean_env(SVR_EMBEDDING_PROVIDER="cohere"),
            clear=True,
        ):
            provider, *_ = detect_embedding_provider(explicit_provider="openai")
        assert provider == "openai"

    def test_svr_embedding_provider_env_var_overrides_api_key_autodetect(self):
        """SVR_EMBEDDING_PROVIDER env var takes precedence over API-key auto-detection."""
        with patch.dict(
            "os.environ",
            _clean_env(
                SVR_EMBEDDING_PROVIDER="cohere",
                VOYAGE_API_KEY="pa-test",
            ),
            clear=True,
        ):
            provider, *_ = detect_embedding_provider()
        assert provider == "cohere"


class TestParseIntEnv:
    """Tests for _parse_int_env()."""

    def test_valid_integer_string(self):
        """A valid integer string is parsed correctly."""
        with patch.dict("os.environ", {"SVR_DIMENSIONS": "512"}):
            assert _parse_int_env("SVR_DIMENSIONS") == 512

    def test_missing_env_var_returns_none(self):
        """Missing env var returns None."""
        with patch.dict("os.environ", _clean_env(), clear=True):
            assert _parse_int_env("SVR_DIMENSIONS") is None

    def test_invalid_non_numeric_string_raises_error(self):
        """Non-numeric value raises ConfigurationError."""
        with patch.dict("os.environ", {"SVR_DIMENSIONS": "not_a_number"}):
            with pytest.raises(ConfigurationError, match="Invalid value for SVR_DIMENSIONS"):
                _parse_int_env("SVR_DIMENSIONS")

    def test_empty_string_raises_error(self):
        """Empty string is not a valid integer and raises ConfigurationError."""
        with patch.dict("os.environ", {"SVR_DIMENSIONS": ""}):
            with pytest.raises(ConfigurationError, match="Invalid value for SVR_DIMENSIONS"):
                _parse_int_env("SVR_DIMENSIONS")

    def test_negative_integer(self):
        """Negative integer strings are parsed correctly."""
        with patch.dict("os.environ", {"SVR_DIMENSIONS": "-1"}):
            assert _parse_int_env("SVR_DIMENSIONS") == -1

    def test_float_string_raises_error(self):
        """Float string like '3.14' is not a valid integer."""
        with patch.dict("os.environ", {"SVR_DIMENSIONS": "3.14"}):
            with pytest.raises(ConfigurationError, match="Invalid value for SVR_DIMENSIONS"):
                _parse_int_env("SVR_DIMENSIONS")


class TestResolveQuickstartParams:
    """Tests for resolve_quickstart_params()."""

    def test_all_env_vars_resolved(self):
        """All SVR_* env vars are picked up when no explicit args are given."""
        env = _clean_env(
            SVR_DATABASE="mydb",
            SVR_COLLECTION="mycol",
            SVR_PARTITION_FIELD="category",
            SVR_BACKEND="postgresql",
            SVR_EMBEDDING_PROVIDER="openai",
            SVR_DIMENSIONS="768",
        )
        with patch.dict("os.environ", env, clear=True):
            result = resolve_quickstart_params()
        assert result["database"] == "mydb"
        assert result["collection"] == "mycol"
        assert result["partition_field"] == "category"
        assert result["backend"] == "postgresql"
        assert result["embedding_provider"] == "openai"
        assert result["dimensions"] == 768

    def test_explicit_args_override_env_vars(self):
        """Explicit keyword arguments override the corresponding env vars."""
        env = _clean_env(
            SVR_DATABASE="env_db",
            SVR_COLLECTION="env_col",
            SVR_PARTITION_FIELD="env_field",
            SVR_BACKEND="postgresql",
            SVR_EMBEDDING_PROVIDER="cohere",
        )
        with patch.dict("os.environ", env, clear=True):
            result = resolve_quickstart_params(
                database="arg_db",
                collection="arg_col",
                partition_field="arg_field",
                backend="mongodb",
                embedding_provider="voyage",
            )
        assert result["database"] == "arg_db"
        assert result["collection"] == "arg_col"
        assert result["partition_field"] == "arg_field"
        assert result["backend"] == "mongodb"
        assert result["embedding_provider"] == "voyage"

    def test_backend_defaults_to_mongodb(self):
        """When neither arg nor env var is set, backend defaults to 'mongodb'."""
        with patch.dict("os.environ", _clean_env(), clear=True):
            result = resolve_quickstart_params()
        assert result["backend"] == "mongodb"

    def test_missing_params_are_none(self):
        """Parameters not supplied via args or env vars resolve to None."""
        with patch.dict("os.environ", _clean_env(), clear=True):
            result = resolve_quickstart_params()
        assert result["database"] is None
        assert result["collection"] is None
        assert result["partition_field"] is None
        assert result["embedding_provider"] is None
        assert result["dimensions"] is None

    def test_dimensions_parsed_from_env(self):
        """SVR_DIMENSIONS is parsed as an integer via _parse_int_env."""
        with patch.dict("os.environ", _clean_env(SVR_DIMENSIONS="2048"), clear=True):
            result = resolve_quickstart_params()
        assert result["dimensions"] == 2048

    def test_invalid_dimensions_raises_error(self):
        """Non-numeric SVR_DIMENSIONS raises ConfigurationError via _parse_int_env."""
        with patch.dict("os.environ", _clean_env(SVR_DIMENSIONS="abc"), clear=True):
            with pytest.raises(ConfigurationError, match="Invalid value for SVR_DIMENSIONS"):
                resolve_quickstart_params()

    def test_partial_env_vars(self):
        """Only some env vars set — the rest resolve to None / defaults."""
        with patch.dict(
            "os.environ",
            _clean_env(SVR_DATABASE="only_db"),
            clear=True,
        ):
            result = resolve_quickstart_params()
        assert result["database"] == "only_db"
        assert result["collection"] is None
        assert result["backend"] == "mongodb"


class TestDeepMerge:
    """Tests for _deep_merge()."""

    def test_non_overlapping_keys(self):
        """Non-overlapping keys are combined into a single dict."""
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_override_scalar_replaces_base(self):
        """A scalar value in override replaces the same key in base."""
        base = {"a": 1}
        override = {"a": 99}
        assert _deep_merge(base, override) == {"a": 99}

    def test_nested_dicts_are_recursively_merged(self):
        """Nested dicts are merged recursively, not replaced wholesale."""
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_override_replaces_dict_with_scalar(self):
        """If override value is scalar but base value is dict, the scalar wins."""
        base = {"x": {"nested": True}}
        override = {"x": "flat"}
        assert _deep_merge(base, override) == {"x": "flat"}

    def test_override_replaces_scalar_with_dict(self):
        """If override value is dict but base value is scalar, the dict wins."""
        base = {"x": "flat"}
        override = {"x": {"nested": True}}
        assert _deep_merge(base, override) == {"x": {"nested": True}}

    def test_empty_base(self):
        """Merging into an empty base returns the override."""
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self):
        """Merging with an empty override returns the base unchanged."""
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_both_empty(self):
        """Merging two empty dicts returns empty dict."""
        assert _deep_merge({}, {}) == {}

    def test_deeply_nested_merge(self):
        """Three levels of nesting are merged correctly."""
        base = {"l1": {"l2": {"l3_a": 1}}}
        override = {"l1": {"l2": {"l3_b": 2}}}
        result = _deep_merge(base, override)
        assert result == {"l1": {"l2": {"l3_a": 1, "l3_b": 2}}}

    def test_original_dicts_not_mutated(self):
        """The input dicts are not modified in place."""
        base = {"a": 1, "nested": {"x": 10}}
        override = {"nested": {"y": 20}}
        base_copy = {"a": 1, "nested": {"x": 10}}
        override_copy = {"nested": {"y": 20}}
        _deep_merge(base, override)
        assert base == base_copy
        assert override == override_copy

    def test_list_values_are_replaced_not_merged(self):
        """Lists in override replace lists in base (no concatenation)."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        assert _deep_merge(base, override) == {"items": [4, 5]}
