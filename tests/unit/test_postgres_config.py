"""Comprehensive unit tests for PostgreSQL backend configuration models."""

import json

import pytest

from semantic_vector_router.backends.postgres.config import (
    HnswConfig,
    IvfflatConfig,
    PgDistanceMetric,
    PgIndexType,
    PostgresBackendConfig,
)
from semantic_vector_router.models.enums import BackendType
from semantic_vector_router.models.svr_config import SVRConfig


# ---------------------------------------------------------------------------
# PgIndexType enum
# ---------------------------------------------------------------------------


class TestPgIndexType:
    """Tests for PgIndexType enum values and string casting."""

    def test_hnsw_value(self):
        assert PgIndexType.HNSW.value == "hnsw"

    def test_ivfflat_value(self):
        assert PgIndexType.IVFFLAT.value == "ivfflat"

    def test_hnsw_str_cast(self):
        # On Python 3.12+, str() on (str, Enum) returns qualified name;
        # use .value for the raw string value.
        assert PgIndexType.HNSW.value == "hnsw"
        assert f"{PgIndexType.HNSW.value}" == "hnsw"

    def test_ivfflat_str_cast(self):
        assert PgIndexType.IVFFLAT.value == "ivfflat"
        assert f"{PgIndexType.IVFFLAT.value}" == "ivfflat"

    def test_hnsw_is_str(self):
        assert isinstance(PgIndexType.HNSW, str)

    def test_ivfflat_is_str(self):
        assert isinstance(PgIndexType.IVFFLAT, str)

    def test_from_value_hnsw(self):
        assert PgIndexType("hnsw") is PgIndexType.HNSW

    def test_from_value_ivfflat(self):
        assert PgIndexType("ivfflat") is PgIndexType.IVFFLAT

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            PgIndexType("btree")

    def test_member_count(self):
        assert len(PgIndexType) == 2


# ---------------------------------------------------------------------------
# PgDistanceMetric enum
# ---------------------------------------------------------------------------


class TestPgDistanceMetric:
    """Tests for PgDistanceMetric enum values and string casting."""

    def test_cosine_value(self):
        assert PgDistanceMetric.COSINE.value == "cosine"

    def test_l2_value(self):
        assert PgDistanceMetric.L2.value == "l2"

    def test_inner_product_value(self):
        assert PgDistanceMetric.INNER_PRODUCT.value == "ip"

    def test_cosine_str_cast(self):
        assert PgDistanceMetric.COSINE.value == "cosine"
        assert f"{PgDistanceMetric.COSINE.value}" == "cosine"

    def test_l2_str_cast(self):
        assert PgDistanceMetric.L2.value == "l2"
        assert f"{PgDistanceMetric.L2.value}" == "l2"

    def test_inner_product_str_cast(self):
        assert PgDistanceMetric.INNER_PRODUCT.value == "ip"
        assert f"{PgDistanceMetric.INNER_PRODUCT.value}" == "ip"

    def test_all_are_str_instances(self):
        for member in PgDistanceMetric:
            assert isinstance(member, str)

    def test_from_value_cosine(self):
        assert PgDistanceMetric("cosine") is PgDistanceMetric.COSINE

    def test_from_value_l2(self):
        assert PgDistanceMetric("l2") is PgDistanceMetric.L2

    def test_from_value_ip(self):
        assert PgDistanceMetric("ip") is PgDistanceMetric.INNER_PRODUCT

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            PgDistanceMetric("manhattan")

    def test_member_count(self):
        assert len(PgDistanceMetric) == 3


# ---------------------------------------------------------------------------
# HnswConfig
# ---------------------------------------------------------------------------


class TestHnswConfig:
    """Tests for HnswConfig default and custom values."""

    def test_defaults(self):
        cfg = HnswConfig()
        assert cfg.m == 16
        assert cfg.ef_construction == 64
        assert cfg.ef_search == 40

    def test_custom_m(self):
        cfg = HnswConfig(m=32)
        assert cfg.m == 32

    def test_custom_ef_construction(self):
        cfg = HnswConfig(ef_construction=128)
        assert cfg.ef_construction == 128

    def test_custom_ef_search(self):
        cfg = HnswConfig(ef_search=100)
        assert cfg.ef_search == 100

    def test_all_custom(self):
        cfg = HnswConfig(m=48, ef_construction=256, ef_search=200)
        assert cfg.m == 48
        assert cfg.ef_construction == 256
        assert cfg.ef_search == 200

    def test_dict_round_trip(self):
        cfg = HnswConfig(m=24, ef_construction=100, ef_search=60)
        data = cfg.model_dump()
        restored = HnswConfig(**data)
        assert restored == cfg

    def test_json_round_trip(self):
        cfg = HnswConfig(m=24, ef_construction=100, ef_search=60)
        json_str = cfg.model_dump_json()
        restored = HnswConfig.model_validate_json(json_str)
        assert restored == cfg


# ---------------------------------------------------------------------------
# IvfflatConfig
# ---------------------------------------------------------------------------


class TestIvfflatConfig:
    """Tests for IvfflatConfig default and custom values."""

    def test_defaults(self):
        cfg = IvfflatConfig()
        assert cfg.lists == 100
        assert cfg.probes == 10

    def test_custom_lists(self):
        cfg = IvfflatConfig(lists=200)
        assert cfg.lists == 200

    def test_custom_probes(self):
        cfg = IvfflatConfig(probes=20)
        assert cfg.probes == 20

    def test_all_custom(self):
        cfg = IvfflatConfig(lists=500, probes=50)
        assert cfg.lists == 500
        assert cfg.probes == 50

    def test_dict_round_trip(self):
        cfg = IvfflatConfig(lists=300, probes=30)
        data = cfg.model_dump()
        restored = IvfflatConfig(**data)
        assert restored == cfg

    def test_json_round_trip(self):
        cfg = IvfflatConfig(lists=300, probes=30)
        json_str = cfg.model_dump_json()
        restored = IvfflatConfig.model_validate_json(json_str)
        assert restored == cfg


# ---------------------------------------------------------------------------
# PostgresBackendConfig — Defaults
# ---------------------------------------------------------------------------


class TestPostgresBackendConfigDefaults:
    """Tests for PostgresBackendConfig default values."""

    def test_connection_string_env_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.connection_string_env == "POSTGRES_URI"

    def test_schema_name_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.schema_name == "public"

    def test_table_prefix_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.table_prefix == "svr_"

    def test_index_type_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.index_type == PgIndexType.HNSW

    def test_distance_metric_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.distance_metric == PgDistanceMetric.COSINE

    def test_pool_min_size_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.pool_min_size == 5

    def test_pool_max_size_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.pool_max_size == 20

    def test_hnsw_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.hnsw == HnswConfig()

    def test_ivfflat_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.ivfflat == IvfflatConfig()

    def test_statement_timeout_ms_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.statement_timeout_ms == 30_000

    def test_vector_dimensions_default(self):
        cfg = PostgresBackendConfig()
        assert cfg.vector_dimensions is None


# ---------------------------------------------------------------------------
# PostgresBackendConfig — Custom Values
# ---------------------------------------------------------------------------


class TestPostgresBackendConfigCustom:
    """Tests for PostgresBackendConfig with custom values."""

    def test_custom_connection_string_env(self):
        cfg = PostgresBackendConfig(connection_string_env="PG_CONN")
        assert cfg.connection_string_env == "PG_CONN"

    def test_custom_table_prefix(self):
        cfg = PostgresBackendConfig(table_prefix="myapp_")
        assert cfg.table_prefix == "myapp_"

    def test_custom_index_type_ivfflat(self):
        cfg = PostgresBackendConfig(index_type=PgIndexType.IVFFLAT)
        assert cfg.index_type == PgIndexType.IVFFLAT

    def test_custom_index_type_from_string(self):
        cfg = PostgresBackendConfig(index_type="ivfflat")
        assert cfg.index_type == PgIndexType.IVFFLAT

    def test_custom_distance_metric_l2(self):
        cfg = PostgresBackendConfig(distance_metric=PgDistanceMetric.L2)
        assert cfg.distance_metric == PgDistanceMetric.L2

    def test_custom_distance_metric_from_string(self):
        cfg = PostgresBackendConfig(distance_metric="ip")
        assert cfg.distance_metric == PgDistanceMetric.INNER_PRODUCT

    def test_custom_pool_sizes(self):
        cfg = PostgresBackendConfig(pool_min_size=2, pool_max_size=50)
        assert cfg.pool_min_size == 2
        assert cfg.pool_max_size == 50

    def test_custom_hnsw(self):
        custom_hnsw = HnswConfig(m=32, ef_construction=128, ef_search=80)
        cfg = PostgresBackendConfig(hnsw=custom_hnsw)
        assert cfg.hnsw.m == 32
        assert cfg.hnsw.ef_construction == 128
        assert cfg.hnsw.ef_search == 80

    def test_custom_ivfflat(self):
        custom_ivf = IvfflatConfig(lists=250, probes=25)
        cfg = PostgresBackendConfig(ivfflat=custom_ivf)
        assert cfg.ivfflat.lists == 250
        assert cfg.ivfflat.probes == 25

    def test_custom_statement_timeout(self):
        cfg = PostgresBackendConfig(statement_timeout_ms=60_000)
        assert cfg.statement_timeout_ms == 60_000

    def test_vector_dimensions_set(self):
        cfg = PostgresBackendConfig(vector_dimensions=1024)
        assert cfg.vector_dimensions == 1024

    def test_vector_dimensions_none_explicit(self):
        cfg = PostgresBackendConfig(vector_dimensions=None)
        assert cfg.vector_dimensions is None

    def test_vector_dimensions_zero(self):
        cfg = PostgresBackendConfig(vector_dimensions=0)
        assert cfg.vector_dimensions == 0

    def test_fully_custom_config(self):
        cfg = PostgresBackendConfig(
            connection_string_env="MY_PG",
            schema="analytics",
            table_prefix="data_",
            index_type=PgIndexType.IVFFLAT,
            distance_metric=PgDistanceMetric.INNER_PRODUCT,
            pool_min_size=1,
            pool_max_size=10,
            hnsw=HnswConfig(m=64, ef_construction=512, ef_search=256),
            ivfflat=IvfflatConfig(lists=500, probes=50),
            statement_timeout_ms=120_000,
            vector_dimensions=768,
        )
        assert cfg.connection_string_env == "MY_PG"
        assert cfg.schema_name == "analytics"
        assert cfg.table_prefix == "data_"
        assert cfg.index_type == PgIndexType.IVFFLAT
        assert cfg.distance_metric == PgDistanceMetric.INNER_PRODUCT
        assert cfg.pool_min_size == 1
        assert cfg.pool_max_size == 10
        assert cfg.hnsw.m == 64
        assert cfg.ivfflat.lists == 500
        assert cfg.statement_timeout_ms == 120_000
        assert cfg.vector_dimensions == 768


# ---------------------------------------------------------------------------
# PostgresBackendConfig — Schema Alias
# ---------------------------------------------------------------------------


class TestPostgresBackendConfigSchemaAlias:
    """Tests for schema alias ('schema') mapping to field 'schema_name'."""

    def test_create_with_alias_schema(self):
        cfg = PostgresBackendConfig(schema="custom_schema")
        assert cfg.schema_name == "custom_schema"

    def test_create_with_field_name_schema_name(self):
        cfg = PostgresBackendConfig(schema_name="other_schema")
        assert cfg.schema_name == "other_schema"

    def test_populate_by_name_enabled(self):
        """populate_by_name in model_config allows using field name directly."""
        cfg = PostgresBackendConfig(schema_name="via_field_name")
        assert cfg.schema_name == "via_field_name"

    def test_dict_with_alias(self):
        data = {"schema": "from_dict"}
        cfg = PostgresBackendConfig(**data)
        assert cfg.schema_name == "from_dict"

    def test_dict_with_field_name(self):
        data = {"schema_name": "from_dict_field"}
        cfg = PostgresBackendConfig(**data)
        assert cfg.schema_name == "from_dict_field"

    def test_model_dump_uses_field_name(self):
        cfg = PostgresBackendConfig(schema="myschema")
        dumped = cfg.model_dump()
        assert "schema_name" in dumped
        assert dumped["schema_name"] == "myschema"

    def test_model_dump_by_alias(self):
        cfg = PostgresBackendConfig(schema="myschema")
        dumped = cfg.model_dump(by_alias=True)
        assert "schema" in dumped
        assert dumped["schema"] == "myschema"


# ---------------------------------------------------------------------------
# PostgresBackendConfig — Pool size (no cross-field validation)
# ---------------------------------------------------------------------------


class TestPostgresBackendConfigPoolValidation:
    """Pool min/max cross-field validation."""

    def test_pool_min_greater_than_max_rejected(self):
        """Cross-field validation rejects pool_min_size > pool_max_size."""
        with pytest.raises(ValueError, match="pool_max_size.*must be.*pool_min_size"):
            PostgresBackendConfig(pool_min_size=50, pool_max_size=5)

    def test_pool_equal_min_max(self):
        cfg = PostgresBackendConfig(pool_min_size=10, pool_max_size=10)
        assert cfg.pool_min_size == 10
        assert cfg.pool_max_size == 10

    def test_pool_zero_min(self):
        cfg = PostgresBackendConfig(pool_min_size=0)
        assert cfg.pool_min_size == 0


# ---------------------------------------------------------------------------
# PostgresBackendConfig — Serialization Round-Trips
# ---------------------------------------------------------------------------


class TestPostgresBackendConfigSerialization:
    """JSON and dict serialization round-trips."""

    def test_json_round_trip_defaults(self):
        cfg = PostgresBackendConfig()
        json_str = cfg.model_dump_json()
        restored = PostgresBackendConfig.model_validate_json(json_str)
        assert restored == cfg

    def test_json_round_trip_custom(self):
        cfg = PostgresBackendConfig(
            connection_string_env="PG_URI",
            schema="test_schema",
            table_prefix="t_",
            index_type=PgIndexType.IVFFLAT,
            distance_metric=PgDistanceMetric.L2,
            pool_min_size=3,
            pool_max_size=15,
            hnsw=HnswConfig(m=8, ef_construction=32, ef_search=20),
            ivfflat=IvfflatConfig(lists=50, probes=5),
            statement_timeout_ms=10_000,
            vector_dimensions=384,
        )
        json_str = cfg.model_dump_json()
        restored = PostgresBackendConfig.model_validate_json(json_str)
        assert restored == cfg

    def test_dict_round_trip_defaults(self):
        cfg = PostgresBackendConfig()
        data = cfg.model_dump()
        restored = PostgresBackendConfig(**data)
        assert restored == cfg

    def test_dict_round_trip_custom(self):
        cfg = PostgresBackendConfig(
            schema="roundtrip",
            vector_dimensions=512,
            index_type="hnsw",
        )
        data = cfg.model_dump()
        restored = PostgresBackendConfig(**data)
        assert restored == cfg

    def test_json_contains_expected_keys(self):
        cfg = PostgresBackendConfig()
        data = json.loads(cfg.model_dump_json())
        expected_keys = {
            "connection_string_env",
            "schema_name",
            "table_prefix",
            "index_type",
            "distance_metric",
            "pool_min_size",
            "pool_max_size",
            "hnsw",
            "ivfflat",
            "statement_timeout_ms",
            "vector_dimensions",
        }
        assert set(data.keys()) == expected_keys

    def test_json_by_alias_uses_schema_key(self):
        cfg = PostgresBackendConfig()
        data = json.loads(cfg.model_dump_json(by_alias=True))
        assert "schema" in data
        assert "schema_name" not in data

    def test_dict_construction_with_alias(self):
        raw = {
            "schema": "aliased",
            "connection_string_env": "PG_URL",
            "table_prefix": "pfx_",
        }
        cfg = PostgresBackendConfig(**raw)
        assert cfg.schema_name == "aliased"
        assert cfg.connection_string_env == "PG_URL"
        assert cfg.table_prefix == "pfx_"

    def test_nested_hnsw_in_json(self):
        cfg = PostgresBackendConfig(hnsw=HnswConfig(m=48))
        data = json.loads(cfg.model_dump_json())
        assert data["hnsw"]["m"] == 48
        assert data["hnsw"]["ef_construction"] == 64
        assert data["hnsw"]["ef_search"] == 40

    def test_nested_ivfflat_in_json(self):
        cfg = PostgresBackendConfig(ivfflat=IvfflatConfig(lists=200))
        data = json.loads(cfg.model_dump_json())
        assert data["ivfflat"]["lists"] == 200
        assert data["ivfflat"]["probes"] == 10


# ---------------------------------------------------------------------------
# PostgresBackendConfig — table_prefix customization
# ---------------------------------------------------------------------------


class TestPostgresBackendConfigTablePrefix:
    """Table prefix customization tests."""

    def test_default_prefix(self):
        cfg = PostgresBackendConfig()
        assert cfg.table_prefix == "svr_"

    def test_empty_prefix(self):
        cfg = PostgresBackendConfig(table_prefix="")
        assert cfg.table_prefix == ""

    def test_custom_prefix_with_underscore(self):
        cfg = PostgresBackendConfig(table_prefix="myapp_")
        assert cfg.table_prefix == "myapp_"

    def test_custom_prefix_without_underscore(self):
        cfg = PostgresBackendConfig(table_prefix="data")
        assert cfg.table_prefix == "data"


# ---------------------------------------------------------------------------
# BackendType.POSTGRES enum value
# ---------------------------------------------------------------------------


class TestBackendTypePostgres:
    """Tests for BackendType.POSTGRES enum value."""

    def test_postgres_value(self):
        assert BackendType.POSTGRES.value == "postgres"

    def test_postgres_str_cast(self):
        assert BackendType.POSTGRES.value == "postgres"
        assert f"{BackendType.POSTGRES.value}" == "postgres"

    def test_postgres_is_str(self):
        assert isinstance(BackendType.POSTGRES, str)

    def test_from_value(self):
        assert BackendType("postgres") is BackendType.POSTGRES

    def test_not_equal_to_mongodb(self):
        assert BackendType.POSTGRES != BackendType.MONGODB


# ---------------------------------------------------------------------------
# SVRConfig integration with postgres field
# ---------------------------------------------------------------------------


def _minimal_svr_kwargs():
    """Return minimal kwargs to construct a valid SVRConfig."""
    return {
        "database": {
            "database": "test_db",
            "source_collection": "test_col",
        },
        "partitioning": {
            "field": "category",
        },
    }


class TestSVRConfigPostgresIntegration:
    """Tests for SVRConfig with postgres field."""

    def test_postgres_none_by_default(self):
        cfg = SVRConfig(**_minimal_svr_kwargs())
        assert cfg.postgres is None

    def test_postgres_accepts_config_instance(self):
        kwargs = _minimal_svr_kwargs()
        kwargs["postgres"] = PostgresBackendConfig()
        cfg = SVRConfig(**kwargs)
        assert cfg.postgres is not None
        assert isinstance(cfg.postgres, PostgresBackendConfig)

    def test_postgres_accepts_dict(self):
        kwargs = _minimal_svr_kwargs()
        kwargs["postgres"] = {"schema": "analytics", "vector_dimensions": 768}
        cfg = SVRConfig(**kwargs)
        assert cfg.postgres is not None
        assert cfg.postgres.schema_name == "analytics"
        assert cfg.postgres.vector_dimensions == 768

    def test_postgres_with_custom_config(self):
        kwargs = _minimal_svr_kwargs()
        pg_cfg = PostgresBackendConfig(
            connection_string_env="PROD_PG",
            schema="prod",
            index_type=PgIndexType.IVFFLAT,
        )
        kwargs["postgres"] = pg_cfg
        cfg = SVRConfig(**kwargs)
        assert cfg.postgres.connection_string_env == "PROD_PG"
        assert cfg.postgres.schema_name == "prod"
        assert cfg.postgres.index_type == PgIndexType.IVFFLAT

    def test_postgres_explicit_none(self):
        kwargs = _minimal_svr_kwargs()
        kwargs["postgres"] = None
        cfg = SVRConfig(**kwargs)
        assert cfg.postgres is None

    def test_svr_config_database_backend_postgres(self):
        kwargs = _minimal_svr_kwargs()
        kwargs["database"]["backend"] = "postgres"
        kwargs["postgres"] = PostgresBackendConfig()
        cfg = SVRConfig(**kwargs)
        assert cfg.database.backend == BackendType.POSTGRES
        assert cfg.postgres is not None

    def test_svr_postgres_round_trip_json(self):
        kwargs = _minimal_svr_kwargs()
        kwargs["postgres"] = PostgresBackendConfig(
            schema="rt_test", vector_dimensions=256
        )
        cfg = SVRConfig(**kwargs)
        json_str = cfg.model_dump_json()
        restored = SVRConfig.model_validate_json(json_str)
        assert restored.postgres is not None
        assert restored.postgres.schema_name == "rt_test"
        assert restored.postgres.vector_dimensions == 256
