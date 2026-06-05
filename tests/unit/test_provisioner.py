"""Comprehensive unit tests for PartitionProvisioner.

Tests the backend-agnostic provisioner which delegates all storage/index
operations to abstract backend methods (create_partition_storage,
create_partition_index, etc.).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.exceptions import (
    PartitionAlreadyExistsError,
    PartitionNotFoundError,
    PartitionProvisioningError,
)
from semantic_vector_router.lifecycle.provisioner import (
    MAX_FIELDS_PARTITIONS,
    SOURCE_INDEX_NAME,
    PartitionProvisioner,
)
from semantic_vector_router.models import (
    IndexLocation,
    PartitionInfo,
    PartitionStatus,
    SVRConfig,
)
from semantic_vector_router.models.backend import IndexStatus, PartitionStorageResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAVE_CONFIG_PATH = "semantic_vector_router.lifecycle.provisioner.save_config"


def _make_provisioner(
    backend: AsyncMock,
    config: SVRConfig,
    auto_save: bool = True,
) -> PartitionProvisioner:
    """Create a PartitionProvisioner with standard defaults."""
    return PartitionProvisioner(backend, config, auto_save_config=auto_save)


def _views_storage_result(name: str) -> PartitionStorageResult:
    """PartitionStorageResult for VIEWS mode (search on source collection)."""
    return PartitionStorageResult(
        storage_name=f"svr_test_partition_{name}",
        storage_type="view",
        view_name=f"svr_test_partition_{name}",
        search_collection="test_collection",
    )


def _views_81_storage_result(name: str) -> PartitionStorageResult:
    """PartitionStorageResult for VIEWS 8.1+ mode (search on view itself)."""
    return PartitionStorageResult(
        storage_name=f"svr_test_partition_{name}",
        storage_type="view",
        view_name=f"svr_test_partition_{name}",
        search_collection=f"svr_test_partition_{name}",
    )


def _source_storage_result(name: str) -> PartitionStorageResult:
    """PartitionStorageResult for SOURCE mode (optional view for browsing)."""
    return PartitionStorageResult(
        storage_name="test_collection",
        storage_type="source",
        view_name=f"svr_test_partition_{name}",
        search_collection="test_collection",
    )


def _source_storage_result_no_view() -> PartitionStorageResult:
    """PartitionStorageResult for SOURCE mode without view."""
    return PartitionStorageResult(
        storage_name="test_collection",
        storage_type="source",
        search_collection="test_collection",
    )


def _fields_storage_result(name: str) -> PartitionStorageResult:
    """PartitionStorageResult for FIELDS mode."""
    sanitized = name.replace("-", "_").replace(" ", "_")
    return PartitionStorageResult(
        storage_name="test_collection",
        storage_type="field",
        embedding_field=f"embedding_{sanitized}",
        search_collection="test_collection",
    )


def _setup_backend(
    backend: AsyncMock,
    storage_result: PartitionStorageResult,
    index_name: str,
    doc_count: int = 500,
) -> None:
    """Configure mock backend with abstract partition operations."""
    backend.create_partition_storage = AsyncMock(return_value=storage_result)
    backend.create_partition_index = AsyncMock(return_value=index_name)
    backend.delete_partition_index = AsyncMock()
    backend.delete_partition_storage = AsyncMock()
    backend.partition_storage_exists = AsyncMock(return_value=True)
    backend.get_partition_index_status = AsyncMock(return_value=IndexStatus.READY)
    backend.count_documents = AsyncMock(return_value=doc_count)


# ===========================================================================
# create_partition — VIEWS mode
# ===========================================================================


class TestCreatePartitionViewsMode:
    """VIEWS mode — backend creates view storage, provisioner assembles partition."""

    @pytest.mark.asyncio
    async def test_create_partition_views_mode(self, sample_config, mock_backend):
        """VIEWS mode: storage returns view, index applied to source."""
        storage = _views_storage_result("electronics")
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=500)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition("electronics")

        # Backend abstract methods called
        mock_backend.create_partition_storage.assert_awaited_once()
        mock_backend.create_partition_index.assert_awaited_once()

        # Partition assembled correctly
        assert partition.view_name == "svr_test_partition_electronics"
        assert partition.search_collection == "test_collection"
        assert partition.index_name == SOURCE_INDEX_NAME
        assert partition.index_location == IndexLocation.VIEWS
        assert partition.document_count == 500
        assert partition.filter_value == "electronics"
        assert partition.status == PartitionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_create_partition_views_81_plus(self, sample_config, mock_backend):
        """VIEWS 8.1+ mode: search_collection points to the view."""
        storage = _views_81_storage_result("electronics")
        _setup_backend(mock_backend, storage, "svr_test_idx_electronics", doc_count=750)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition("electronics")

        assert partition.search_collection == "svr_test_partition_electronics"
        assert partition.index_name == "svr_test_idx_electronics"
        assert partition.view_name == "svr_test_partition_electronics"
        assert partition.document_count == 750

    @pytest.mark.asyncio
    async def test_views_mode_count_uses_view(self, sample_config, mock_backend):
        """VIEWS mode: document count uses the view collection."""
        storage = _views_storage_result("electronics")
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=500)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            await prov.create_partition("electronics")

        # Count called on the view
        mock_backend.count_documents.assert_awaited_once_with(
            "svr_test_partition_electronics"
        )


# ===========================================================================
# create_partition — SOURCE mode
# ===========================================================================


class TestCreatePartitionSourceMode:
    """SOURCE mode — shared index on source, optional view for browsing."""

    @pytest.mark.asyncio
    async def test_create_partition_source_mode(
        self, sample_config_source, mock_backend
    ):
        storage = _source_storage_result("electronics")
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=300)

        prov = _make_provisioner(mock_backend, sample_config_source, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition("electronics")

        assert partition.search_collection == "test_collection"
        assert partition.index_name == SOURCE_INDEX_NAME
        assert partition.view_name == "svr_test_partition_electronics"
        assert partition.index_location == IndexLocation.SOURCE

    @pytest.mark.asyncio
    async def test_create_partition_source_mode_no_view(
        self, sample_config_source, mock_backend
    ):
        """SOURCE mode without view — backend returns no view_name."""
        storage = _source_storage_result_no_view()
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=200)

        prov = _make_provisioner(mock_backend, sample_config_source, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition("electronics")

        assert partition.search_collection == "test_collection"
        assert partition.view_name is None

    @pytest.mark.asyncio
    async def test_source_mode_count_uses_filter(
        self, sample_config_source, mock_backend
    ):
        """SOURCE mode: count uses partition filter on source collection."""
        storage = _source_storage_result_no_view()
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=200)

        prov = _make_provisioner(mock_backend, sample_config_source, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            await prov.create_partition("electronics")

        # Count called with filter on source collection
        mock_backend.count_documents.assert_awaited_once_with(
            "test_collection", {"category": "electronics"}
        )


# ===========================================================================
# create_partition — FIELDS mode
# ===========================================================================


class TestCreatePartitionFieldsMode:
    """FIELDS mode — per-partition embedding field + index on source."""

    @pytest.mark.asyncio
    async def test_create_partition_fields_mode(
        self, sample_config_fields, mock_backend
    ):
        storage = _fields_storage_result("electronics")
        _setup_backend(
            mock_backend, storage, "svr_test_idx_electronics", doc_count=120
        )

        prov = _make_provisioner(mock_backend, sample_config_fields, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition("electronics")

        assert partition.view_name is None
        assert partition.search_collection == "test_collection"
        assert partition.embedding_field == "embedding_electronics"
        assert partition.index_location == IndexLocation.FIELDS
        assert partition.document_count == 120

    @pytest.mark.asyncio
    async def test_fields_mode_count_uses_embedding_field(
        self, sample_config_fields, mock_backend
    ):
        """FIELDS mode: count uses $exists on the embedding field."""
        storage = _fields_storage_result("electronics")
        _setup_backend(
            mock_backend, storage, "svr_test_idx_electronics", doc_count=120
        )

        prov = _make_provisioner(mock_backend, sample_config_fields, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            await prov.create_partition("electronics")

        mock_backend.count_documents.assert_awaited_once_with(
            "test_collection",
            {"embedding_electronics": {"$exists": True}},
        )


# ===========================================================================
# create_partition — existence checks
# ===========================================================================


class TestCreatePartitionExistence:
    """Tests for skip_if_exists and duplicate detection."""

    @pytest.mark.asyncio
    async def test_create_partition_skip_if_exists(
        self, sample_config_with_partitions, mock_backend
    ):
        """Returns existing partition when skip_if_exists=True."""
        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=False
        )

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition(
                "electronics", skip_if_exists=True
            )

        assert partition.name == "electronics"
        assert partition.document_count == 150000
        # No backend calls should have been made
        mock_backend.create_partition_storage.assert_not_awaited()
        mock_backend.create_partition_index.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_partition_already_exists_raises(
        self, sample_config_with_partitions, mock_backend
    ):
        """Raises PartitionAlreadyExistsError when partition exists."""
        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=False
        )

        with pytest.raises(PartitionAlreadyExistsError, match="already exists"):
            await prov.create_partition("electronics", skip_if_exists=False)


# ===========================================================================
# FIELDS mode cap
# ===========================================================================


class TestFieldsModeCap:
    """FIELDS mode enforces a 50-partition cap."""

    @pytest.mark.asyncio
    async def test_fields_mode_50_partition_cap(
        self, sample_config_fields, mock_backend
    ):
        # Pre-populate registry with MAX_FIELDS_PARTITIONS entries
        for i in range(MAX_FIELDS_PARTITIONS):
            sample_config_fields.partitions.registry[f"part_{i}"] = PartitionInfo(
                name=f"part_{i}",
                index_name=f"svr_test_idx_part_{i}",
                filter_value=f"part_{i}",
                index_location=IndexLocation.FIELDS,
                search_collection="test_collection",
                embedding_field=f"embedding_part_{i}",
            )

        prov = _make_provisioner(mock_backend, sample_config_fields, auto_save=False)

        with pytest.raises(PartitionProvisioningError, match="limited to"):
            await prov.create_partition("one_too_many")


# ===========================================================================
# create_partitions_batch
# ===========================================================================


class TestCreatePartitionsBatch:
    """Batch creation suppresses saves and does a single save at end."""

    @pytest.mark.asyncio
    async def test_create_partitions_batch_single_save(
        self, sample_config, mock_backend
    ):
        """Regression: auto_save suppressed during batch, single save at end."""

        async def _storage_side_effect(preliminary, config):
            return _views_storage_result(preliminary.name)

        mock_backend.create_partition_storage = AsyncMock(
            side_effect=_storage_side_effect
        )
        mock_backend.create_partition_index = AsyncMock(
            return_value=SOURCE_INDEX_NAME
        )
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()
        mock_backend.count_documents = AsyncMock(return_value=100)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=True)

        with patch(SAVE_CONFIG_PATH) as mock_save:
            result = await prov.create_partitions_batch(
                ["electronics", "furniture", "clothing"],
                skip_existing=True,
            )

        assert len(result) == 3
        # Only ONE save_config call — at the end of the batch
        mock_save.assert_called_once_with(sample_config)

    @pytest.mark.asyncio
    async def test_create_partitions_batch_restores_auto_save_on_error(
        self, sample_config, mock_backend
    ):
        """auto_save_config is restored even if a partition creation fails."""
        call_count = 0

        async def _storage_side_effect(preliminary, config):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("boom")
            return _views_storage_result(preliminary.name)

        mock_backend.create_partition_storage = AsyncMock(
            side_effect=_storage_side_effect
        )
        mock_backend.create_partition_index = AsyncMock(
            return_value=SOURCE_INDEX_NAME
        )
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()
        mock_backend.count_documents = AsyncMock(return_value=50)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=True)

        with patch(SAVE_CONFIG_PATH):
            result = await prov.create_partitions_batch(
                ["a", "b", "c"], skip_existing=True
            )

        # auto_save_config must be restored to True
        assert prov.auto_save_config is True
        # Two succeeded (a and c), one failed (b)
        assert len(result) == 2


# ===========================================================================
# delete_partition
# ===========================================================================


class TestDeletePartitionConfigFirst:
    """Regression: config is removed FIRST, then backend cleanup happens."""

    @pytest.mark.asyncio
    async def test_delete_partition_config_first(
        self, sample_config_with_partitions, mock_backend
    ):
        """Config entry is removed and saved before any backend operations."""
        call_order = []

        async def _delete_index(partition):
            call_order.append("delete_index")

        async def _delete_storage(partition):
            call_order.append("delete_storage")

        mock_backend.delete_partition_index = AsyncMock(side_effect=_delete_index)
        mock_backend.delete_partition_storage = AsyncMock(side_effect=_delete_storage)

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=True
        )

        with patch(SAVE_CONFIG_PATH) as mock_save:

            def _save_side_effect(cfg):
                call_order.append("save_config")
                # At save time, partition must already be removed from registry
                assert "electronics" not in cfg.partitions.registry

            mock_save.side_effect = _save_side_effect
            await prov.delete_partition("electronics")

        # Verify ordering: save_config before any backend cleanup
        assert call_order.index("save_config") < call_order.index("delete_index")
        assert call_order.index("save_config") < call_order.index("delete_storage")
        # Partition no longer in registry
        assert "electronics" not in sample_config_with_partitions.partitions.registry


class TestDeletePartitionConfigRestoredOnSaveFailure:
    """Regression: if save_config fails, the registry entry is restored."""

    @pytest.mark.asyncio
    async def test_delete_partition_config_restored_on_save_failure(
        self, sample_config_with_partitions, mock_backend
    ):
        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=True
        )

        with patch(SAVE_CONFIG_PATH, side_effect=OSError("disk full")):
            with pytest.raises(PartitionProvisioningError, match="Failed to save"):
                await prov.delete_partition("electronics")

        # Registry entry MUST be restored
        assert "electronics" in sample_config_with_partitions.partitions.registry
        assert (
            sample_config_with_partitions.partitions.registry["electronics"].name
            == "electronics"
        )


class TestDeletePartitionAbstractMethods:
    """Verify delete_partition calls abstract backend methods."""

    @pytest.mark.asyncio
    async def test_delete_partition_calls_abstract_methods(
        self, sample_config_with_partitions, mock_backend
    ):
        """delete_partition calls backend.delete_partition_index and delete_partition_storage."""
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        partition = sample_config_with_partitions.partitions.registry["electronics"]

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=True
        )

        with patch(SAVE_CONFIG_PATH):
            await prov.delete_partition("electronics")

        mock_backend.delete_partition_index.assert_awaited_once_with(partition)
        mock_backend.delete_partition_storage.assert_awaited_once_with(partition)

    @pytest.mark.asyncio
    async def test_delete_partition_skip_index_delete(
        self, sample_config_with_partitions, mock_backend
    ):
        """delete_index=False skips index cleanup."""
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=True
        )

        with patch(SAVE_CONFIG_PATH):
            await prov.delete_partition("electronics", delete_index=False)

        mock_backend.delete_partition_index.assert_not_awaited()
        mock_backend.delete_partition_storage.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_partition_skip_storage_delete(
        self, sample_config_with_partitions, mock_backend
    ):
        """delete_view=False skips storage cleanup."""
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=True
        )

        with patch(SAVE_CONFIG_PATH):
            await prov.delete_partition("electronics", delete_view=False)

        mock_backend.delete_partition_index.assert_awaited_once()
        mock_backend.delete_partition_storage.assert_not_awaited()


class TestDeletePartitionBestEffort:
    """Backend cleanup failures are logged but do not raise."""

    @pytest.mark.asyncio
    async def test_delete_partition_best_effort_cleanup(
        self, sample_config_with_partitions, mock_backend
    ):
        """Backend cleanup failures are logged but do not raise."""
        mock_backend.delete_partition_index = AsyncMock(
            side_effect=RuntimeError("index gone")
        )
        mock_backend.delete_partition_storage = AsyncMock(
            side_effect=RuntimeError("storage gone")
        )

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=True
        )

        with patch(SAVE_CONFIG_PATH):
            # Should NOT raise despite backend errors
            await prov.delete_partition("electronics")

        assert "electronics" not in sample_config_with_partitions.partitions.registry


class TestDeletePartitionNotFound:
    """delete_partition raises PartitionNotFoundError for unknown name."""

    @pytest.mark.asyncio
    async def test_delete_nonexistent_partition(self, sample_config, mock_backend):
        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with pytest.raises(PartitionNotFoundError, match="not found"):
            await prov.delete_partition("nonexistent")


# ===========================================================================
# update_partition_count
# ===========================================================================


class TestUpdatePartitionCountViews:
    """VIEWS mode counts documents on the view."""

    @pytest.mark.asyncio
    async def test_update_partition_count_views(
        self, sample_config_with_partitions, mock_backend
    ):
        mock_backend.count_documents = AsyncMock(return_value=42000)

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=False
        )

        count = await prov.update_partition_count("electronics")

        assert count == 42000
        # Called with view_name (not source collection)
        mock_backend.count_documents.assert_awaited_once_with(
            "svr_test_partition_electronics"
        )
        partition = sample_config_with_partitions.partitions.registry["electronics"]
        assert partition.document_count == 42000
        assert partition.last_count_update is not None


class TestUpdatePartitionCountFields:
    """FIELDS mode counts by embedding field existence."""

    @pytest.mark.asyncio
    async def test_update_partition_count_fields(
        self, sample_config_with_fields_partitions, mock_backend
    ):
        mock_backend.count_documents = AsyncMock(return_value=8800)

        prov = _make_provisioner(
            mock_backend, sample_config_with_fields_partitions, auto_save=False
        )

        count = await prov.update_partition_count("electronics")

        assert count == 8800
        mock_backend.count_documents.assert_awaited_once_with(
            "test_collection",
            {"embedding_electronics": {"$exists": True}},
        )


class TestUpdatePartitionCountSource:
    """SOURCE mode counts by filter on source collection."""

    @pytest.mark.asyncio
    async def test_update_partition_count_source(
        self, sample_config_with_source_partitions, mock_backend
    ):
        mock_backend.count_documents = AsyncMock(return_value=6600)

        prov = _make_provisioner(
            mock_backend, sample_config_with_source_partitions, auto_save=False
        )

        # SOURCE partition with view_name = None triggers filter-based count
        partition = sample_config_with_source_partitions.partitions.registry[
            "electronics"
        ]
        partition.view_name = None  # Force no view so filter path is taken

        count = await prov.update_partition_count("electronics")

        assert count == 6600
        mock_backend.count_documents.assert_awaited_once_with(
            "test_collection",
            {"category": "electronics"},
        )

    @pytest.mark.asyncio
    async def test_update_partition_count_source_with_filter_expression(
        self, sample_config_with_source_partitions, mock_backend
    ):
        mock_backend.count_documents = AsyncMock(return_value=3300)

        prov = _make_provisioner(
            mock_backend, sample_config_with_source_partitions, auto_save=False
        )

        partition = sample_config_with_source_partitions.partitions.registry[
            "electronics"
        ]
        partition.view_name = None
        partition.filter_expression = {"category": {"$in": ["electronics", "tech"]}}

        count = await prov.update_partition_count("electronics")

        assert count == 3300
        mock_backend.count_documents.assert_awaited_once_with(
            "test_collection",
            {"category": {"$in": ["electronics", "tech"]}},
        )


class TestUpdatePartitionCountNotFound:
    """update_partition_count raises PartitionNotFoundError for unknown name."""

    @pytest.mark.asyncio
    async def test_update_count_nonexistent_partition(
        self, sample_config, mock_backend
    ):
        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with pytest.raises(PartitionNotFoundError, match="not found"):
            await prov.update_partition_count("nonexistent")


# ===========================================================================
# update_all_partition_counts
# ===========================================================================


class TestUpdateAllPartitionCounts:
    """Batch update suppresses individual saves, does single save at end."""

    @pytest.mark.asyncio
    async def test_update_all_partition_counts_batches_saves(
        self, sample_config_with_partitions, mock_backend
    ):
        mock_backend.count_documents = AsyncMock(return_value=999)

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=True
        )

        with patch(SAVE_CONFIG_PATH) as mock_save:
            counts = await prov.update_all_partition_counts()

        # All three partitions updated
        assert len(counts) == 3
        assert all(v == 999 for v in counts.values())

        # Only ONE save call at the end (not three)
        mock_save.assert_called_once_with(sample_config_with_partitions)

        # auto_save_config restored
        assert prov.auto_save_config is True


# ===========================================================================
# ensure_source_index
# ===========================================================================


class TestEnsureSourceIndex:
    """Tests for ensure_source_index delegation to backend."""

    @pytest.mark.asyncio
    async def test_ensure_source_index_delegates_to_backend(
        self, sample_config_source, mock_backend
    ):
        mock_backend.ensure_source_index = AsyncMock(
            return_value=SOURCE_INDEX_NAME
        )

        prov = _make_provisioner(mock_backend, sample_config_source, auto_save=False)
        name = await prov.ensure_source_index()

        assert name == SOURCE_INDEX_NAME
        mock_backend.ensure_source_index.assert_awaited_once_with(
            auto_detect_filters=False,
            extra_filter_fields=None,
        )

    @pytest.mark.asyncio
    async def test_ensure_source_index_passes_kwargs(
        self, sample_config_source, mock_backend
    ):
        mock_backend.ensure_source_index = AsyncMock(
            return_value=SOURCE_INDEX_NAME
        )

        prov = _make_provisioner(mock_backend, sample_config_source, auto_save=False)
        await prov.ensure_source_index(
            auto_detect_filters=True,
            extra_filter_fields=["brand", "price_range"],
        )

        mock_backend.ensure_source_index.assert_awaited_once_with(
            auto_detect_filters=True,
            extra_filter_fields=["brand", "price_range"],
        )

    @pytest.mark.asyncio
    async def test_ensure_source_index_returns_empty_if_backend_lacks_method(
        self, sample_config_source,
    ):
        """If backend doesn't have ensure_source_index, return empty string."""
        # Use spec to prevent auto-creation of attributes
        backend = MagicMock(spec=[])  # Empty spec — no methods

        prov = _make_provisioner(backend, sample_config_source, auto_save=False)
        name = await prov.ensure_source_index()

        assert name == ""


# ===========================================================================
# verify_partition
# ===========================================================================


class TestVerifyPartition:
    """Tests for partition verification."""

    @pytest.mark.asyncio
    async def test_verify_existing_partition(
        self, sample_config_with_partitions, mock_backend
    ):
        mock_backend.partition_storage_exists = AsyncMock(return_value=True)
        mock_backend.get_partition_index_status = AsyncMock(
            return_value=IndexStatus.READY
        )

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=False
        )
        result = await prov.verify_partition("electronics")

        assert result["name"] == "electronics"
        assert result["exists"] is True
        assert result["view_exists"] is True
        assert result["index_status"] == "ready"
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_verify_nonexistent_partition(self, sample_config, mock_backend):
        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)
        result = await prov.verify_partition("nonexistent")

        assert result["exists"] is False
        assert result["status"] == "not_registered"

    @pytest.mark.asyncio
    async def test_verify_all_partitions(
        self, sample_config_with_partitions, mock_backend
    ):
        mock_backend.partition_storage_exists = AsyncMock(return_value=True)
        mock_backend.get_partition_index_status = AsyncMock(
            return_value=IndexStatus.READY
        )

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=False
        )
        results = await prov.verify_all_partitions()

        assert len(results) == 3
        assert all(r["exists"] for r in results)


# ===========================================================================
# Auto-save behavior
# ===========================================================================


class TestAutoSave:
    """Verify auto_save_config triggers save_config at the right times."""

    @pytest.mark.asyncio
    async def test_create_partition_auto_saves(self, sample_config, mock_backend):
        storage = _views_storage_result("new_partition")
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=10)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=True)

        with patch(SAVE_CONFIG_PATH) as mock_save:
            await prov.create_partition("new_partition")

        mock_save.assert_called_once_with(sample_config)

    @pytest.mark.asyncio
    async def test_create_partition_no_save_when_disabled(
        self, sample_config, mock_backend
    ):
        storage = _views_storage_result("new_partition")
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=10)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH) as mock_save:
            await prov.create_partition("new_partition")

        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_partition_count_auto_saves(
        self, sample_config_with_partitions, mock_backend
    ):
        mock_backend.count_documents = AsyncMock(return_value=5000)

        prov = _make_provisioner(
            mock_backend, sample_config_with_partitions, auto_save=True
        )

        with patch(SAVE_CONFIG_PATH) as mock_save:
            await prov.update_partition_count("electronics")

        mock_save.assert_called_once_with(sample_config_with_partitions)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    @pytest.mark.asyncio
    async def test_create_partition_uses_name_as_filter_value(
        self, sample_config, mock_backend
    ):
        """When filter_value is omitted it defaults to name."""
        storage = _views_storage_result("shoes")
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=0)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition("shoes")

        assert partition.filter_value == "shoes"

    @pytest.mark.asyncio
    async def test_create_partition_with_custom_filter_expression(
        self, sample_config, mock_backend
    ):
        """Custom filter_expression is passed through to backend."""
        storage = _views_storage_result("custom")
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=0)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)
        expr = {"category": {"$in": ["a", "b"]}}

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition(
                "custom", filter_expression=expr
            )

        assert partition.filter_expression == expr
        # Verify backend received the expression in the preliminary
        call_args = mock_backend.create_partition_storage.call_args
        preliminary = call_args[0][0]
        assert preliminary.filter_expression == expr

    @pytest.mark.asyncio
    async def test_fields_mode_name_sanitization(
        self, sample_config_fields, mock_backend
    ):
        """FIELDS mode: embedding field reflects backend's sanitization."""
        storage = _fields_storage_result("my-cool partition")
        _setup_backend(
            mock_backend, storage, "svr_test_idx_my_cool_partition", doc_count=0
        )

        prov = _make_provisioner(mock_backend, sample_config_fields, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            partition = await prov.create_partition("my-cool partition")

        # The embedding field is set from backend's storage result
        assert partition.embedding_field == "embedding_my_cool_partition"
        assert partition.index_name == "svr_test_idx_my_cool_partition"

    @pytest.mark.asyncio
    async def test_create_partition_preliminary_has_correct_fields(
        self, sample_config, mock_backend
    ):
        """Verify the preliminary PartitionInfo passed to backend."""
        storage = _views_storage_result("electronics")
        _setup_backend(mock_backend, storage, SOURCE_INDEX_NAME, doc_count=100)

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            await prov.create_partition(
                "electronics", filter_value="elec", filter_expression=None
            )

        call_args = mock_backend.create_partition_storage.call_args
        preliminary = call_args[0][0]
        assert preliminary.name == "electronics"
        assert preliminary.filter_value == "elec"
        assert preliminary.index_location == IndexLocation.VIEWS


# ===========================================================================
# Phase 3: Rollback on create_partition failure
# ===========================================================================


class TestRollbackOnStorageFailure:
    """When create_partition_storage fails, nothing was created — just raise."""

    @pytest.mark.asyncio
    async def test_storage_failure_raises_provisioning_error(
        self, sample_config, mock_backend
    ):
        """If create_partition_storage fails, PartitionProvisioningError is raised
        and no cleanup calls are needed (nothing was created yet)."""
        mock_backend.create_partition_storage = AsyncMock(
            side_effect=RuntimeError("storage creation failed")
        )
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            with pytest.raises(
                PartitionProvisioningError, match="storage creation failed"
            ):
                await prov.create_partition("electronics")

        # No cleanup needed — nothing was created
        mock_backend.delete_partition_index.assert_not_awaited()
        mock_backend.delete_partition_storage.assert_not_awaited()
        # Partition must not appear in the registry
        assert "electronics" not in sample_config.partitions.registry

    @pytest.mark.asyncio
    async def test_storage_failure_different_config(
        self, sample_config_fields, mock_backend
    ):
        """Same test for FIELDS mode config."""
        mock_backend.create_partition_storage = AsyncMock(
            side_effect=RuntimeError("namespace error")
        )
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(mock_backend, sample_config_fields, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            with pytest.raises(PartitionProvisioningError, match="namespace error"):
                await prov.create_partition("electronics")

        mock_backend.delete_partition_index.assert_not_awaited()
        mock_backend.delete_partition_storage.assert_not_awaited()
        assert "electronics" not in sample_config_fields.partitions.registry


class TestRollbackOnIndexCreationFailure:
    """When create_partition_index fails after storage created → rollback storage."""

    @pytest.mark.asyncio
    async def test_index_failure_rolls_back_storage(
        self, sample_config, mock_backend
    ):
        """When create_partition_index fails after storage,
        storage must be cleaned up via delete_partition_storage."""
        storage = _views_81_storage_result("electronics")
        mock_backend.create_partition_storage = AsyncMock(return_value=storage)
        mock_backend.create_partition_index = AsyncMock(
            side_effect=RuntimeError("index creation timeout")
        )
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            with pytest.raises(
                PartitionProvisioningError, match="index creation timeout"
            ):
                await prov.create_partition("electronics")

        # Rollback should delete the storage that was successfully created
        mock_backend.delete_partition_storage.assert_awaited_once()
        # Index was NOT created, so no index cleanup
        mock_backend.delete_partition_index.assert_not_awaited()
        # Partition must NOT be in registry
        assert "electronics" not in sample_config.partitions.registry

    @pytest.mark.asyncio
    async def test_source_mode_index_failure_rolls_back_storage(
        self, sample_config_source, mock_backend
    ):
        """SOURCE mode: index fails → rollback storage."""
        storage = _source_storage_result("electronics")
        mock_backend.create_partition_storage = AsyncMock(return_value=storage)
        mock_backend.create_partition_index = AsyncMock(
            side_effect=RuntimeError("source index failed")
        )
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(mock_backend, sample_config_source, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            with pytest.raises(
                PartitionProvisioningError, match="source index failed"
            ):
                await prov.create_partition("electronics")

        # Storage should be cleaned up
        mock_backend.delete_partition_storage.assert_awaited_once()
        assert "electronics" not in sample_config_source.partitions.registry


class TestRollbackOnConfigSaveFailure:
    """If everything creates fine but save_config() fails, rollback removes from registry."""

    @pytest.mark.asyncio
    async def test_save_config_failure_triggers_rollback(
        self, sample_config, mock_backend
    ):
        """When save_config raises, the partition is removed from registry
        and a PartitionProvisioningError is raised."""
        storage = _views_81_storage_result("electronics")
        _setup_backend(
            mock_backend, storage, "svr_test_idx_electronics", doc_count=500
        )

        prov = _make_provisioner(mock_backend, sample_config, auto_save=True)

        with patch(SAVE_CONFIG_PATH, side_effect=OSError("disk full")):
            with pytest.raises(PartitionProvisioningError, match="disk full"):
                await prov.create_partition("electronics")

        # Registry must NOT contain the partition after rollback
        assert "electronics" not in sample_config.partitions.registry

    @pytest.mark.asyncio
    async def test_save_config_failure_rolls_back_created_resources(
        self, sample_config, mock_backend
    ):
        """When save_config fails, rollback also cleans up storage and index."""
        storage = _views_81_storage_result("electronics")
        _setup_backend(
            mock_backend, storage, "svr_test_idx_electronics", doc_count=500
        )

        prov = _make_provisioner(mock_backend, sample_config, auto_save=True)

        with patch(SAVE_CONFIG_PATH, side_effect=OSError("disk full")):
            with pytest.raises(PartitionProvisioningError):
                await prov.create_partition("electronics")

        # Both storage and index should be cleaned up
        mock_backend.delete_partition_storage.assert_awaited_once()
        mock_backend.delete_partition_index.assert_awaited_once()


class TestRollbackFailureDoesntMaskOriginalError:
    """If rollback itself fails, the original error is still raised."""

    @pytest.mark.asyncio
    async def test_rollback_storage_delete_failure_still_raises_original(
        self, sample_config, mock_backend
    ):
        """If delete_partition_storage raises during rollback, the original
        PartitionProvisioningError (wrapping the index failure) is still raised."""
        storage = _views_storage_result("electronics")
        mock_backend.create_partition_storage = AsyncMock(return_value=storage)
        mock_backend.create_partition_index = AsyncMock(
            side_effect=RuntimeError("index creation failed")
        )
        # Make rollback itself fail
        mock_backend.delete_partition_storage = AsyncMock(
            side_effect=RuntimeError("storage delete also failed")
        )
        mock_backend.delete_partition_index = AsyncMock()

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            with pytest.raises(
                PartitionProvisioningError, match="index creation failed"
            ):
                await prov.create_partition("electronics")

        # The rollback still attempted the storage cleanup
        mock_backend.delete_partition_storage.assert_awaited_once()
        # Partition must not be in registry
        assert "electronics" not in sample_config.partitions.registry

    @pytest.mark.asyncio
    async def test_rollback_index_delete_failure_still_raises_original(
        self, sample_config, mock_backend
    ):
        """If delete_partition_index raises during rollback,
        the original error is still surfaced."""
        storage = _views_81_storage_result("electronics")
        mock_backend.create_partition_storage = AsyncMock(return_value=storage)
        mock_backend.create_partition_index = AsyncMock(
            return_value="svr_test_idx_electronics"
        )
        mock_backend.count_documents = AsyncMock(
            side_effect=RuntimeError("count failed")
        )
        # Make rollback fail for index cleanup
        mock_backend.delete_partition_index = AsyncMock(
            side_effect=RuntimeError("cannot delete index")
        )
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(mock_backend, sample_config, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            with pytest.raises(PartitionProvisioningError, match="count failed"):
                await prov.create_partition("electronics")

        # Rollback attempted both index and storage cleanup
        mock_backend.delete_partition_index.assert_awaited_once()
        mock_backend.delete_partition_storage.assert_awaited_once()
        assert "electronics" not in sample_config.partitions.registry


class TestRollbackFieldsMode:
    """FIELDS mode rollback scenarios."""

    @pytest.mark.asyncio
    async def test_fields_index_creation_failure_rolls_back_storage(
        self, sample_config_fields, mock_backend
    ):
        """FIELDS mode: create_partition_index fails → rollback storage."""
        storage = _fields_storage_result("electronics")
        mock_backend.create_partition_storage = AsyncMock(return_value=storage)
        mock_backend.create_partition_index = AsyncMock(
            side_effect=RuntimeError("fields index failed")
        )
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(mock_backend, sample_config_fields, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            with pytest.raises(
                PartitionProvisioningError, match="fields index failed"
            ):
                await prov.create_partition("electronics")

        # Storage was created, so rollback cleans it up
        mock_backend.delete_partition_storage.assert_awaited_once()
        # Index was NOT created
        mock_backend.delete_partition_index.assert_not_awaited()
        # Partition must not be in registry
        assert "electronics" not in sample_config_fields.partitions.registry

    @pytest.mark.asyncio
    async def test_fields_count_documents_failure_rolls_back_index(
        self, sample_config_fields, mock_backend
    ):
        """FIELDS mode: index created but count_documents fails →
        rollback should delete the index and storage."""
        storage = _fields_storage_result("electronics")
        mock_backend.create_partition_storage = AsyncMock(return_value=storage)
        mock_backend.create_partition_index = AsyncMock(
            return_value="svr_test_idx_electronics"
        )
        mock_backend.count_documents = AsyncMock(
            side_effect=RuntimeError("count failed")
        )
        mock_backend.delete_partition_index = AsyncMock()
        mock_backend.delete_partition_storage = AsyncMock()

        prov = _make_provisioner(mock_backend, sample_config_fields, auto_save=False)

        with patch(SAVE_CONFIG_PATH):
            with pytest.raises(PartitionProvisioningError, match="count failed"):
                await prov.create_partition("electronics")

        # Rollback should delete both index and storage
        mock_backend.delete_partition_index.assert_awaited_once()
        mock_backend.delete_partition_storage.assert_awaited_once()
        # Partition must not be in registry
        assert "electronics" not in sample_config_fields.partitions.registry

    @pytest.mark.asyncio
    async def test_fields_save_config_failure_rolls_back_index(
        self, sample_config_fields, mock_backend
    ):
        """FIELDS mode: everything creates but save_config fails →
        rollback removes registry entry and deletes index + storage."""
        storage = _fields_storage_result("electronics")
        _setup_backend(
            mock_backend, storage, "svr_test_idx_electronics", doc_count=100
        )

        prov = _make_provisioner(mock_backend, sample_config_fields, auto_save=True)

        with patch(SAVE_CONFIG_PATH, side_effect=OSError("disk full")):
            with pytest.raises(PartitionProvisioningError, match="disk full"):
                await prov.create_partition("electronics")

        # Both index and storage should be rolled back
        mock_backend.delete_partition_index.assert_awaited_once()
        mock_backend.delete_partition_storage.assert_awaited_once()
        # Registry cleaned up
        assert "electronics" not in sample_config_fields.partitions.registry
