"""Comprehensive unit tests for MetadataStore class."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from pymongo.errors import DuplicateKeyError

from semantic_vector_router.backends.metadata import MetadataStore
from semantic_vector_router.exceptions import MetadataError
from semantic_vector_router.models import (
    IndexLocation,
    PartitionInfo,
    PartitionStatus,
    SVRConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_collection():
    """Create a mock MongoDB collection with async cursor support."""
    coll = AsyncMock()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    coll.find = MagicMock(return_value=cursor)
    # For sorted cursors (list_operations uses .sort())
    cursor.sort = MagicMock(return_value=cursor)
    return coll


@pytest.fixture
def metadata_store(sample_config, mock_collection):
    """Create a MetadataStore with mocked collection and db."""
    store = MetadataStore(sample_config)
    store._collection = mock_collection
    store._db = MagicMock()
    return store


def _make_partition_doc(
    name="test",
    index_name="idx_test",
    status="active",
    filter_value=None,
    document_count=0,
    view_name=None,
):
    """Build a partition document that looks like what MongoDB returns."""
    return {
        "_id": f"partition:{name}",
        "type": "partition",
        "name": name,
        "index_name": index_name,
        "status": status,
        "filter_value": filter_value,
        "document_count": document_count,
        "view_name": view_name,
        "health_history": [],
        "created_at": datetime.utcnow().isoformat(),
    }


# ===================================================================
# 1. Partition CRUD
# ===================================================================


class TestPartitionCRUD:
    """Tests for partition CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_partition_found(self, metadata_store, mock_collection):
        """When find_one returns a doc, get_partition should return PartitionInfo."""
        doc = _make_partition_doc(name="electronics", index_name="svr_idx_electronics")
        mock_collection.find_one = AsyncMock(return_value=doc)

        result = await metadata_store.get_partition("electronics")

        assert result is not None
        assert isinstance(result, PartitionInfo)
        assert result.name == "electronics"
        assert result.index_name == "svr_idx_electronics"
        mock_collection.find_one.assert_awaited_once_with(
            {"_id": "partition:electronics", "type": "partition"}
        )

    @pytest.mark.asyncio
    async def test_get_partition_not_found(self, metadata_store, mock_collection):
        """When find_one returns None, get_partition should return None."""
        mock_collection.find_one = AsyncMock(return_value=None)

        result = await metadata_store.get_partition("nonexistent")

        assert result is None
        mock_collection.find_one.assert_awaited_once_with(
            {"_id": "partition:nonexistent", "type": "partition"}
        )

    @pytest.mark.asyncio
    async def test_list_partitions(self, metadata_store, mock_collection):
        """list_partitions should return a list of PartitionInfo from cursor docs."""
        docs = [
            _make_partition_doc(name="electronics", index_name="idx_electronics"),
            _make_partition_doc(name="furniture", index_name="idx_furniture"),
        ]
        cursor = AsyncMock()
        cursor.to_list = AsyncMock(return_value=docs)
        mock_collection.find = MagicMock(return_value=cursor)

        result = await metadata_store.list_partitions()

        assert len(result) == 2
        assert all(isinstance(p, PartitionInfo) for p in result)
        assert result[0].name == "electronics"
        assert result[1].name == "furniture"
        mock_collection.find.assert_called_once_with({"type": "partition"})
        cursor.to_list.assert_awaited_once_with(length=None)

    @pytest.mark.asyncio
    async def test_list_partitions_with_status_filter(self, metadata_store, mock_collection):
        """list_partitions with status filter should include status in the query."""
        cursor = AsyncMock()
        cursor.to_list = AsyncMock(return_value=[])
        mock_collection.find = MagicMock(return_value=cursor)

        result = await metadata_store.list_partitions(status="active")

        assert result == []
        mock_collection.find.assert_called_once_with(
            {"type": "partition", "status": "active"}
        )

    @pytest.mark.asyncio
    async def test_save_partition(self, metadata_store, mock_collection):
        """save_partition should call replace_one with upsert=True."""
        partition = PartitionInfo(
            name="test_partition",
            index_name="svr_idx_test",
            filter_value="test",
            status=PartitionStatus.ACTIVE,
        )

        await metadata_store.save_partition(partition)

        mock_collection.replace_one.assert_awaited_once()
        call_args = mock_collection.replace_one.call_args
        assert call_args[0][0] == {"_id": "partition:test_partition"}
        # The doc should have _id and type fields
        saved_doc = call_args[0][1]
        assert saved_doc["_id"] == "partition:test_partition"
        assert saved_doc["type"] == "partition"
        assert saved_doc["name"] == "test_partition"
        assert saved_doc["index_name"] == "svr_idx_test"
        assert call_args[1]["upsert"] is True

    @pytest.mark.asyncio
    async def test_delete_partition_success(self, metadata_store, mock_collection):
        """delete_partition should return True when a document is deleted."""
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        mock_collection.delete_one = AsyncMock(return_value=mock_result)

        result = await metadata_store.delete_partition("electronics")

        assert result is True
        mock_collection.delete_one.assert_awaited_once_with(
            {"_id": "partition:electronics", "type": "partition"}
        )

    @pytest.mark.asyncio
    async def test_delete_partition_not_found(self, metadata_store, mock_collection):
        """delete_partition should return False when no document matched."""
        mock_result = MagicMock()
        mock_result.deleted_count = 0
        mock_collection.delete_one = AsyncMock(return_value=mock_result)

        result = await metadata_store.delete_partition("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_update_partition_count(self, metadata_store, mock_collection):
        """update_partition_count should call update_one with $set for count and timestamp."""
        await metadata_store.update_partition_count("electronics", 42000)

        mock_collection.update_one.assert_awaited_once()
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {
            "_id": "partition:electronics",
            "type": "partition",
        }
        update_doc = call_args[0][1]
        assert "$set" in update_doc
        assert update_doc["$set"]["document_count"] == 42000
        assert "last_count_update" in update_doc["$set"]
        assert isinstance(update_doc["$set"]["last_count_update"], datetime)


# ===================================================================
# 2. Health History
# ===================================================================


class TestHealthHistory:
    """Tests for health history operations."""

    @pytest.mark.asyncio
    async def test_append_health_history(self, metadata_store, mock_collection):
        """append_health_history should use $push with $each and $slice:-30."""
        await metadata_store.append_health_history("electronics", 50000)

        mock_collection.update_one.assert_awaited_once()
        call_args = mock_collection.update_one.call_args
        # Check the filter
        assert call_args[0][0] == {
            "_id": "partition:electronics",
            "type": "partition",
        }
        # Check the update uses $push with $each and $slice
        update_doc = call_args[0][1]
        assert "$push" in update_doc
        push_spec = update_doc["$push"]["health_history"]
        assert "$each" in push_spec
        assert len(push_spec["$each"]) == 1
        assert push_spec["$each"][0]["count"] == 50000
        assert "ts" in push_spec["$each"][0]
        assert push_spec["$slice"] == -30

    @pytest.mark.asyncio
    async def test_get_health_history_exists(self, metadata_store, mock_collection):
        """get_health_history should return the history array when doc exists."""
        history = [
            {"ts": datetime.utcnow(), "count": 100},
            {"ts": datetime.utcnow(), "count": 200},
        ]
        mock_collection.find_one = AsyncMock(
            return_value={"_id": "partition:electronics", "health_history": history}
        )

        result = await metadata_store.get_health_history("electronics")

        assert result == history
        assert len(result) == 2
        mock_collection.find_one.assert_awaited_once_with(
            {"_id": "partition:electronics", "type": "partition"},
            {"health_history": 1},
        )

    @pytest.mark.asyncio
    async def test_get_health_history_empty(self, metadata_store, mock_collection):
        """get_health_history should return empty list when doc is not found."""
        mock_collection.find_one = AsyncMock(return_value=None)

        result = await metadata_store.get_health_history("nonexistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_health_history_no_history_field(self, metadata_store, mock_collection):
        """get_health_history should return empty list when doc exists but has no history field."""
        mock_collection.find_one = AsyncMock(
            return_value={"_id": "partition:test"}
        )

        result = await metadata_store.get_health_history("test")

        assert result == []


# ===================================================================
# 3. Operation CRUD
# ===================================================================


class TestOperationCRUD:
    """Tests for operation CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_operation(self, metadata_store, mock_collection):
        """create_operation should call insert_one and return the op_id."""
        operation = {
            "_id": "op:split-electronics-20240101",
            "type": "operation",
            "status": "pending",
            "created_at": datetime.utcnow(),
        }

        result = await metadata_store.create_operation(operation)

        assert result == "op:split-electronics-20240101"
        mock_collection.insert_one.assert_awaited_once()
        inserted_doc = mock_collection.insert_one.call_args[0][0]
        assert inserted_doc["type"] == "operation"
        assert inserted_doc["_id"] == "op:split-electronics-20240101"

    @pytest.mark.asyncio
    async def test_create_operation_auto_sets_type(self, metadata_store, mock_collection):
        """create_operation should set type to 'operation' if not provided."""
        operation = {
            "_id": "op:merge-test-20240101",
            "status": "pending",
        }

        await metadata_store.create_operation(operation)

        inserted_doc = mock_collection.insert_one.call_args[0][0]
        assert inserted_doc["type"] == "operation"

    @pytest.mark.asyncio
    async def test_get_operation(self, metadata_store, mock_collection):
        """get_operation should call find_one with the op_id and type filter."""
        expected_doc = {
            "_id": "op:split-electronics-20240101",
            "type": "operation",
            "status": "pending",
        }
        mock_collection.find_one = AsyncMock(return_value=expected_doc)

        result = await metadata_store.get_operation("op:split-electronics-20240101")

        assert result == expected_doc
        mock_collection.find_one.assert_awaited_once_with(
            {"_id": "op:split-electronics-20240101", "type": "operation"}
        )

    @pytest.mark.asyncio
    async def test_get_operation_not_found(self, metadata_store, mock_collection):
        """get_operation should return None when no operation matches."""
        mock_collection.find_one = AsyncMock(return_value=None)

        result = await metadata_store.get_operation("op:nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_operations(self, metadata_store, mock_collection):
        """list_operations should call find().sort().to_list()."""
        ops = [
            {"_id": "op:1", "type": "operation", "status": "pending", "created_at": datetime.utcnow()},
            {"_id": "op:2", "type": "operation", "status": "done", "created_at": datetime.utcnow()},
        ]
        cursor = AsyncMock()
        cursor.to_list = AsyncMock(return_value=ops)
        cursor.sort = MagicMock(return_value=cursor)
        mock_collection.find = MagicMock(return_value=cursor)

        result = await metadata_store.list_operations()

        assert len(result) == 2
        mock_collection.find.assert_called_once_with({"type": "operation"})
        cursor.sort.assert_called_once_with("created_at", -1)
        cursor.to_list.assert_awaited_once_with(length=None)

    @pytest.mark.asyncio
    async def test_list_operations_with_status_filter(self, metadata_store, mock_collection):
        """list_operations with status filter should include it in the query."""
        cursor = AsyncMock()
        cursor.to_list = AsyncMock(return_value=[])
        cursor.sort = MagicMock(return_value=cursor)
        mock_collection.find = MagicMock(return_value=cursor)

        result = await metadata_store.list_operations(status="pending")

        assert result == []
        mock_collection.find.assert_called_once_with(
            {"type": "operation", "status": "pending"}
        )

    @pytest.mark.asyncio
    async def test_update_operation_status(self, metadata_store, mock_collection):
        """update_operation_status should use $set with status and extra kwargs."""
        await metadata_store.update_operation_status(
            "op:split-1", "in_progress", started_at=datetime(2024, 1, 15)
        )

        mock_collection.update_one.assert_awaited_once()
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {"_id": "op:split-1", "type": "operation"}
        update_doc = call_args[0][1]
        assert update_doc["$set"]["status"] == "in_progress"
        assert update_doc["$set"]["started_at"] == datetime(2024, 1, 15)

    @pytest.mark.asyncio
    async def test_update_operation_status_no_kwargs(self, metadata_store, mock_collection):
        """update_operation_status with no extra kwargs should only set status."""
        await metadata_store.update_operation_status("op:split-1", "done")

        update_doc = mock_collection.update_one.call_args[0][1]
        assert update_doc == {"$set": {"status": "done"}}

    @pytest.mark.asyncio
    async def test_update_operation_step(self, metadata_store, mock_collection):
        """update_operation_step should use array_filters to target a specific step."""
        await metadata_store.update_operation_step(
            "op:split-1", "create_view", "in_progress"
        )

        mock_collection.update_one.assert_awaited_once()
        call_args = mock_collection.update_one.call_args
        # Check filter
        assert call_args[0][0] == {"_id": "op:split-1", "type": "operation"}
        # Check update sets the step status and started_at
        update_doc = call_args[0][1]
        assert "steps.$[elem].status" in update_doc["$set"]
        assert update_doc["$set"]["steps.$[elem].status"] == "in_progress"
        assert "steps.$[elem].started_at" in update_doc["$set"]
        # Check array_filters
        assert call_args[1]["array_filters"] == [{"elem.action": "create_view"}]

    @pytest.mark.asyncio
    async def test_update_operation_step_done(self, metadata_store, mock_collection):
        """update_operation_step with 'done' status should set completed_at."""
        await metadata_store.update_operation_step(
            "op:split-1", "create_index", "done"
        )

        update_doc = mock_collection.update_one.call_args[0][1]
        assert update_doc["$set"]["steps.$[elem].status"] == "done"
        assert "steps.$[elem].completed_at" in update_doc["$set"]

    @pytest.mark.asyncio
    async def test_update_operation_step_failed_with_error(self, metadata_store, mock_collection):
        """update_operation_step with 'failed' status and error should set error field."""
        await metadata_store.update_operation_step(
            "op:split-1", "create_index", "failed", error="Index creation timeout"
        )

        update_doc = mock_collection.update_one.call_args[0][1]
        assert update_doc["$set"]["steps.$[elem].status"] == "failed"
        assert update_doc["$set"]["steps.$[elem].error"] == "Index creation timeout"
        assert "steps.$[elem].completed_at" in update_doc["$set"]


# ===================================================================
# 4. Distributed Lock
# ===================================================================


class TestDistributedLock:
    """Tests for distributed lock operations."""

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, metadata_store, mock_collection):
        """acquire_lock should return True when find_one_and_update returns doc with matching holder."""
        mock_collection.find_one_and_update = AsyncMock(
            return_value={
                "_id": "lock:repartition",
                "type": "lock",
                "holder": "worker-1",
                "acquired_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(seconds=300),
            }
        )

        result = await metadata_store.acquire_lock("repartition", "worker-1", ttl_seconds=300)

        assert result is True
        mock_collection.find_one_and_update.assert_awaited_once()
        call_args = mock_collection.find_one_and_update.call_args
        # Check filter includes _id and $or for expired/same-holder
        filter_doc = call_args[0][0]
        assert filter_doc["_id"] == "lock:repartition"
        assert "$or" in filter_doc
        # Check update sets holder, type, acquired_at, expires_at
        update_doc = call_args[0][1]
        assert update_doc["$set"]["type"] == "lock"
        assert update_doc["$set"]["holder"] == "worker-1"
        assert "acquired_at" in update_doc["$set"]
        assert "expires_at" in update_doc["$set"]
        # Check upsert=True
        assert call_args[1]["upsert"] is True

    @pytest.mark.asyncio
    async def test_acquire_lock_fails_duplicate(self, metadata_store, mock_collection):
        """acquire_lock should return False when DuplicateKeyError is raised (race condition)."""
        mock_collection.find_one_and_update = AsyncMock(
            side_effect=DuplicateKeyError("E11000 duplicate key error")
        )

        result = await metadata_store.acquire_lock("repartition", "worker-2")

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_lock_returns_false_when_holder_mismatch(self, metadata_store, mock_collection):
        """acquire_lock should return False when result holder does not match."""
        mock_collection.find_one_and_update = AsyncMock(
            return_value={
                "_id": "lock:repartition",
                "type": "lock",
                "holder": "worker-other",
            }
        )

        result = await metadata_store.acquire_lock("repartition", "worker-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_lock_returns_false_when_result_is_none(self, metadata_store, mock_collection):
        """acquire_lock should return False when find_one_and_update returns None."""
        mock_collection.find_one_and_update = AsyncMock(return_value=None)

        result = await metadata_store.acquire_lock("repartition", "worker-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_release_lock(self, metadata_store, mock_collection):
        """release_lock should call delete_one with holder filter and return True."""
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        mock_collection.delete_one = AsyncMock(return_value=mock_result)

        result = await metadata_store.release_lock("repartition", "worker-1")

        assert result is True
        mock_collection.delete_one.assert_awaited_once_with(
            {"_id": "lock:repartition", "type": "lock", "holder": "worker-1"}
        )

    @pytest.mark.asyncio
    async def test_release_lock_wrong_holder(self, metadata_store, mock_collection):
        """release_lock should return False when holder does not match (deleted_count=0)."""
        mock_result = MagicMock()
        mock_result.deleted_count = 0
        mock_collection.delete_one = AsyncMock(return_value=mock_result)

        result = await metadata_store.release_lock("repartition", "wrong-worker")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_lock_held_active(self, metadata_store, mock_collection):
        """is_lock_held should return True when lock exists and is not expired."""
        mock_collection.find_one = AsyncMock(
            return_value={
                "_id": "lock:repartition",
                "type": "lock",
                "holder": "worker-1",
                "expires_at": datetime.utcnow() + timedelta(seconds=300),
            }
        )

        result = await metadata_store.is_lock_held("repartition")

        assert result is True
        mock_collection.find_one.assert_awaited_once_with(
            {"_id": "lock:repartition", "type": "lock"}
        )

    @pytest.mark.asyncio
    async def test_is_lock_held_expired(self, metadata_store, mock_collection):
        """is_lock_held should return False when lock exists but is expired."""
        mock_collection.find_one = AsyncMock(
            return_value={
                "_id": "lock:repartition",
                "type": "lock",
                "holder": "worker-1",
                "expires_at": datetime.utcnow() - timedelta(seconds=60),
            }
        )

        result = await metadata_store.is_lock_held("repartition")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_lock_held_no_lock(self, metadata_store, mock_collection):
        """is_lock_held should return False when no lock document exists."""
        mock_collection.find_one = AsyncMock(return_value=None)

        result = await metadata_store.is_lock_held("repartition")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_lock_held_no_expires_at(self, metadata_store, mock_collection):
        """is_lock_held should return False when lock doc has no expires_at field."""
        mock_collection.find_one = AsyncMock(
            return_value={
                "_id": "lock:repartition",
                "type": "lock",
                "holder": "worker-1",
                # no expires_at field
            }
        )

        result = await metadata_store.is_lock_held("repartition")

        assert result is False


# ===================================================================
# 5. Migration
# ===================================================================


class TestMigration:
    """Tests for config-to-metadata migration."""

    @pytest.mark.asyncio
    async def test_migrate_from_config(self, sample_config_with_partitions, mock_collection):
        """migrate_from_config should migrate partitions that don't already exist in metadata."""
        store = MetadataStore(sample_config_with_partitions)
        store._collection = mock_collection
        store._db = MagicMock()

        # "electronics" already exists; "furniture" and "clothing" do not
        async def mock_find_one(query):
            if query == {"_id": "partition:electronics", "type": "partition"}:
                return _make_partition_doc(name="electronics", index_name="svr_test_idx_electronics")
            return None

        mock_collection.find_one = AsyncMock(side_effect=mock_find_one)

        migrated = await store.migrate_from_config(sample_config_with_partitions)

        assert migrated == 2
        # save_partition calls replace_one, should be called twice
        assert mock_collection.replace_one.await_count == 2

    @pytest.mark.asyncio
    async def test_migrate_idempotent(self, sample_config_with_partitions, mock_collection):
        """migrate_from_config should migrate 0 when all partitions already exist."""
        store = MetadataStore(sample_config_with_partitions)
        store._collection = mock_collection
        store._db = MagicMock()

        # All partitions already exist in metadata
        async def mock_find_one(query):
            # Extract name from _id field like "partition:electronics"
            partition_id = query.get("_id", "")
            if partition_id.startswith("partition:"):
                name = partition_id.split(":", 1)[1]
                return _make_partition_doc(name=name, index_name=f"svr_test_idx_{name}")
            return None

        mock_collection.find_one = AsyncMock(side_effect=mock_find_one)

        migrated = await store.migrate_from_config(sample_config_with_partitions)

        assert migrated == 0
        mock_collection.replace_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_migrate_from_config_empty_registry(self, sample_config, mock_collection):
        """migrate_from_config with empty registry should migrate 0."""
        store = MetadataStore(sample_config)
        store._collection = mock_collection
        store._db = MagicMock()

        migrated = await store.migrate_from_config(sample_config)

        assert migrated == 0
        mock_collection.find_one.assert_not_awaited()


# ===================================================================
# 6. Connection
# ===================================================================


class TestConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_set_shared_db(self, sample_config):
        """_set_shared_db should set _db when not using separate client."""
        store = MetadataStore(sample_config)
        assert store._db is None

        mock_db = MagicMock()
        store._set_shared_db(mock_db)

        assert store._db is mock_db

    @pytest.mark.asyncio
    async def test_set_shared_db_ignored_with_separate_client(self, sample_config):
        """_set_shared_db should be a no-op when using separate client."""
        # Configure a separate metadata connection string
        sample_config.lifecycle.metadata.connection_string_env = "METADATA_MONGODB_URI"
        store = MetadataStore(sample_config)
        assert store._use_separate_client is True
        assert store._db is None

        mock_db = MagicMock()
        store._set_shared_db(mock_db)

        # Should remain None because separate client is configured
        assert store._db is None

    @pytest.mark.asyncio
    async def test_connect_raises_without_db(self, sample_config):
        """connect() should raise MetadataError when no db is set and no separate client."""
        store = MetadataStore(sample_config)
        # _db is None and _use_separate_client is False (default), so connect should fail

        with pytest.raises(MetadataError, match="Metadata store not connected"):
            await store.connect()

    @pytest.mark.asyncio
    async def test_connect_with_shared_db(self, sample_config):
        """connect() should succeed when _db has been set via _set_shared_db."""
        store = MetadataStore(sample_config)
        mock_db = MagicMock()
        store._set_shared_db(mock_db)

        await store.connect()

        assert store._collection is not None
        # Collection should be set to db[collection_name]
        mock_db.__getitem__.assert_called_once_with("svr_metadata")

    @pytest.mark.asyncio
    async def test_connect_with_separate_client_missing_env(self, sample_config):
        """connect() with separate client should raise when env var is not set."""
        sample_config.lifecycle.metadata.connection_string_env = "METADATA_MONGODB_URI"
        store = MetadataStore(sample_config)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(MetadataError, match="connection string env var not set"):
                await store.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, sample_config, mock_collection):
        """disconnect() should clear _db and _collection."""
        store = MetadataStore(sample_config)
        store._collection = mock_collection
        store._db = MagicMock()

        await store.disconnect()

        assert store._db is None
        assert store._collection is None

    @pytest.mark.asyncio
    async def test_disconnect_with_separate_client(self, sample_config):
        """disconnect() with separate client should close the client."""
        sample_config.lifecycle.metadata.connection_string_env = "METADATA_MONGODB_URI"
        store = MetadataStore(sample_config)
        mock_client = AsyncMock()
        store._client = mock_client
        store._db = MagicMock()
        store._collection = AsyncMock()

        await store.disconnect()

        mock_client.close.assert_awaited_once()
        assert store._client is None
        assert store._db is None
        assert store._collection is None


# ===================================================================
# 7. Conversion Helpers
# ===================================================================


class TestConversionHelpers:
    """Tests for _partition_to_doc and _doc_to_partition."""

    def test_partition_to_doc(self, metadata_store):
        """_partition_to_doc should produce a document with _id, type, and health_history."""
        partition = PartitionInfo(
            name="electronics",
            index_name="svr_idx_electronics",
            filter_value="electronics",
            status=PartitionStatus.ACTIVE,
            document_count=150000,
        )

        doc = metadata_store._partition_to_doc(partition)

        assert doc["_id"] == "partition:electronics"
        assert doc["type"] == "partition"
        assert doc["name"] == "electronics"
        assert doc["index_name"] == "svr_idx_electronics"
        assert doc["status"] == "active"
        assert doc["document_count"] == 150000
        assert doc["health_history"] == []

    def test_doc_to_partition(self, metadata_store):
        """_doc_to_partition should strip _id, type, health_history and return PartitionInfo."""
        doc = _make_partition_doc(
            name="furniture",
            index_name="svr_idx_furniture",
            status="active",
            filter_value="furniture",
            document_count=85000,
        )

        result = metadata_store._doc_to_partition(doc)

        assert isinstance(result, PartitionInfo)
        assert result.name == "furniture"
        assert result.index_name == "svr_idx_furniture"
        assert result.status == PartitionStatus.ACTIVE
        assert result.document_count == 85000

    def test_doc_to_partition_strips_internal_fields(self, metadata_store):
        """_doc_to_partition should remove _id, type, and health_history from the data."""
        doc = {
            "_id": "partition:test",
            "type": "partition",
            "name": "test",
            "index_name": "idx_test",
            "health_history": [{"ts": datetime.utcnow(), "count": 100}],
            "created_at": datetime.utcnow().isoformat(),
        }

        result = metadata_store._doc_to_partition(doc)

        assert isinstance(result, PartitionInfo)
        assert result.name == "test"
        # PartitionInfo should not have _id, type, or health_history attributes from the doc

    def test_partition_to_doc_roundtrip(self, metadata_store):
        """Converting partition -> doc -> partition should preserve core data."""
        original = PartitionInfo(
            name="roundtrip",
            index_name="svr_idx_roundtrip",
            filter_value="roundtrip_val",
            status=PartitionStatus.ACTIVE,
            document_count=999,
            index_location=IndexLocation.SOURCE,
            search_collection="test_collection",
        )

        doc = metadata_store._partition_to_doc(original)
        restored = metadata_store._doc_to_partition(doc)

        assert restored.name == original.name
        assert restored.index_name == original.index_name
        assert restored.filter_value == original.filter_value
        assert restored.status == original.status
        assert restored.document_count == original.document_count
        assert restored.index_location == original.index_location
        assert restored.search_collection == original.search_collection
