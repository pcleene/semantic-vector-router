"""Unit tests for the RepartitionEngine class."""

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.exceptions import RepartitionError
from semantic_vector_router.lifecycle.repartition import RepartitionEngine
from semantic_vector_router.models import (
    IndexLocation,
    PartitionInfo,
    PartitionStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metadata():
    """Create a mock MetadataStore."""
    metadata = AsyncMock()
    metadata.get_operation = AsyncMock()
    metadata.get_partition = AsyncMock()
    metadata.save_partition = AsyncMock()
    metadata.update_operation_status = AsyncMock()
    metadata.update_operation_step = AsyncMock()
    return metadata


@pytest.fixture
def sample_op():
    """Create a sample repartition operation dict."""
    return {
        "_id": "op:split-electronics-123",
        "type": "operation",
        "action": "split",
        "target_partition": "electronics",
        "strategy": "secondary_field",
        "strategy_config": {"secondary_field": "subcategory"},
        "status": "pending",
        "steps": [
            {"step": 1, "action": "create_children", "status": "pending"},
            {"step": 2, "action": "build_indexes", "status": "pending"},
            {"step": 3, "action": "wait_indexes", "status": "pending"},
            {"step": 4, "action": "switch_routing", "status": "pending"},
            {"step": 5, "action": "cleanup_parent", "status": "pending"},
        ],
    }


@pytest.fixture
def parent_partition():
    """Create a parent PartitionInfo."""
    return PartitionInfo(
        name="electronics",
        index_name="svr_test_idx_electronics",
        view_name="svr_test_partition_electronics",
        filter_value="electronics",
        status=PartitionStatus.ACTIVE,
        child_partitions=[],
    )


@pytest.fixture
def child_partition_a():
    """Create child partition A (subcategory=phones)."""
    return PartitionInfo(
        name="electronics_subcategory_phones",
        index_name="svr_test_idx_electronics_subcategory_phones",
        view_name="svr_test_partition_electronics_subcategory_phones",
        filter_value="phones",
        status=PartitionStatus.ACTIVE,
        parent_partition="electronics",
        search_collection=None,
    )


@pytest.fixture
def child_partition_b():
    """Create child partition B (subcategory=laptops)."""
    return PartitionInfo(
        name="electronics_subcategory_laptops",
        index_name="svr_test_idx_electronics_subcategory_laptops",
        view_name="svr_test_partition_electronics_subcategory_laptops",
        filter_value="laptops",
        status=PartitionStatus.ACTIVE,
        parent_partition="electronics",
        search_collection=None,
    )


@pytest.fixture
def mock_backend():
    """Create a mock backend with sensible defaults for repartition tests."""
    backend = AsyncMock()
    backend.get_index_status = AsyncMock(return_value={"queryable": True})
    backend.get_distinct_values = AsyncMock(return_value=["phones", "laptops"])
    backend.delete_index = AsyncMock()
    backend.delete_view = AsyncMock()
    return backend


@pytest.fixture
def engine(sample_config, mock_backend, mock_metadata):
    """Create a RepartitionEngine with mocked provisioner."""
    with patch(
        "semantic_vector_router.lifecycle.repartition.PartitionProvisioner"
    ) as MockProv:
        prov = AsyncMock()
        prov.create_partition = AsyncMock()
        prov.delete_partition = AsyncMock()
        MockProv.return_value = prov
        eng = RepartitionEngine(mock_backend, mock_metadata, sample_config)
        eng.provisioner = prov
        return eng


# ---------------------------------------------------------------------------
# execute_operation tests
# ---------------------------------------------------------------------------


class TestExecuteOperation:
    """Tests for RepartitionEngine.execute_operation."""

    @pytest.mark.asyncio
    async def test_execute_full_workflow(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
        child_partition_a,
        child_partition_b,
    ):
        """All 5 steps succeed, operation marked done, returns True."""
        mock_metadata.get_operation.return_value = sample_op

        # Parent starts ACTIVE; children are returned after creation
        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]

        # get_partition is called many times across steps.
        # Step 1 (create_children): returns parent (no children yet)
        # Steps 2-5: returns parent with children, plus each child
        call_count = {"n": 0}
        parent_splitting = copy.deepcopy(parent_partition)
        parent_splitting.status = PartitionStatus.SPLITTING
        parent_splitting.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                # First call is from _step_create_children; subsequent from later steps
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return copy.deepcopy(parent_partition)
                return copy.deepcopy(parent_splitting)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            elif name == "electronics_subcategory_laptops":
                return copy.deepcopy(child_partition_b)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect
        mock_backend.get_distinct_values.return_value = ["phones", "laptops"]
        mock_backend.get_index_status.return_value = {"queryable": True}

        result = await engine.execute_operation("op:split-electronics-123")

        assert result is True

        # Verify operation status updated to done
        done_calls = [
            c
            for c in mock_metadata.update_operation_status.call_args_list
            if c.args[1] == "done"
        ]
        assert len(done_calls) == 1

        # Verify all 5 steps were marked in_progress then done
        step_calls = mock_metadata.update_operation_step.call_args_list
        actions_in_progress = [
            c.args[1] for c in step_calls if c.args[2] == "in_progress"
        ]
        actions_done = [c.args[1] for c in step_calls if c.args[2] == "done"]
        assert "create_children" in actions_in_progress
        assert "build_indexes" in actions_in_progress
        assert "wait_indexes" in actions_in_progress
        assert "switch_routing" in actions_in_progress
        assert "cleanup_parent" in actions_in_progress
        assert "create_children" in actions_done
        assert "build_indexes" in actions_done
        assert "wait_indexes" in actions_done
        assert "switch_routing" in actions_done
        assert "cleanup_parent" in actions_done

    @pytest.mark.asyncio
    async def test_execute_skips_done_steps(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
        child_partition_a,
        child_partition_b,
    ):
        """Steps already marked 'done' are skipped; only remaining steps run."""
        # Mark first two steps as done
        sample_op["steps"][0]["status"] = "done"  # create_children
        sample_op["steps"][1]["status"] = "done"  # build_indexes

        mock_metadata.get_operation.return_value = sample_op

        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.status = PartitionStatus.SPLITTING
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            elif name == "electronics_subcategory_laptops":
                return copy.deepcopy(child_partition_b)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect
        mock_backend.get_index_status.return_value = {"queryable": True}

        result = await engine.execute_operation("op:split-electronics-123")

        assert result is True

        # Only steps 3, 4, 5 should have been executed
        step_calls = mock_metadata.update_operation_step.call_args_list
        actions_in_progress = [
            c.args[1] for c in step_calls if c.args[2] == "in_progress"
        ]
        assert "create_children" not in actions_in_progress
        assert "build_indexes" not in actions_in_progress
        assert "wait_indexes" in actions_in_progress
        assert "switch_routing" in actions_in_progress
        assert "cleanup_parent" in actions_in_progress

    @pytest.mark.asyncio
    async def test_execute_fails_on_step(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
        child_partition_a,
        child_partition_b,
    ):
        """When a step fails, operation and step are marked failed, returns False."""
        # Mark first two done, let step 3 (wait_indexes) fail
        sample_op["steps"][0]["status"] = "done"
        sample_op["steps"][1]["status"] = "done"

        mock_metadata.get_operation.return_value = sample_op

        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.status = PartitionStatus.SPLITTING
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            elif name == "electronics_subcategory_laptops":
                return copy.deepcopy(child_partition_b)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect

        # Make get_index_status return None to trigger an error in wait_indexes
        mock_backend.get_index_status.return_value = None

        result = await engine.execute_operation("op:split-electronics-123")

        assert result is False

        # Verify operation marked as failed
        failed_calls = [
            c
            for c in mock_metadata.update_operation_status.call_args_list
            if c.args[1] == "failed"
        ]
        assert len(failed_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_not_found(self, engine, mock_metadata):
        """Operation not found raises RepartitionError."""
        mock_metadata.get_operation.return_value = None

        with pytest.raises(RepartitionError, match="not found"):
            await engine.execute_operation("op:nonexistent-456")


# ---------------------------------------------------------------------------
# rollback_operation tests
# ---------------------------------------------------------------------------


class TestRollbackOperation:
    """Tests for RepartitionEngine.rollback_operation."""

    @pytest.mark.asyncio
    async def test_rollback_deletes_children(
        self,
        engine,
        mock_metadata,
        sample_op,
        parent_partition,
    ):
        """Rollback calls provisioner.delete_partition for each child."""
        mock_metadata.get_operation.return_value = sample_op

        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]
        mock_metadata.get_partition.return_value = parent_with_children

        await engine.rollback_operation("op:split-electronics-123")

        # Verify delete_partition called for each child
        delete_calls = engine.provisioner.delete_partition.call_args_list
        deleted_names = [c.args[0] for c in delete_calls]
        assert "electronics_subcategory_phones" in deleted_names
        assert "electronics_subcategory_laptops" in deleted_names
        assert len(delete_calls) == 2

    @pytest.mark.asyncio
    async def test_rollback_resets_parent(
        self,
        engine,
        mock_metadata,
        sample_op,
        parent_partition,
    ):
        """Rollback resets parent status to ACTIVE and clears child_partitions."""
        mock_metadata.get_operation.return_value = sample_op

        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.status = PartitionStatus.SPLITTING
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]
        mock_metadata.get_partition.return_value = parent_with_children

        await engine.rollback_operation("op:split-electronics-123")

        # Verify save_partition was called with parent reset
        save_calls = mock_metadata.save_partition.call_args_list
        assert len(save_calls) == 1
        saved_partition = save_calls[0].args[0]
        assert saved_partition.status == PartitionStatus.ACTIVE
        assert saved_partition.child_partitions == []

        # Verify operation marked as failed with rollback note
        status_calls = mock_metadata.update_operation_status.call_args_list
        assert len(status_calls) == 1
        assert status_calls[0].args[1] == "failed"
        assert "Rolled back" in status_calls[0].kwargs.get("error", "")

    @pytest.mark.asyncio
    async def test_rollback_not_found(self, engine, mock_metadata):
        """Rollback raises RepartitionError when operation not found."""
        mock_metadata.get_operation.return_value = None

        with pytest.raises(RepartitionError, match="not found"):
            await engine.rollback_operation("op:nonexistent-789")

    @pytest.mark.asyncio
    async def test_rollback_no_target_partition(self, engine, mock_metadata):
        """Rollback raises RepartitionError when target_partition is missing."""
        mock_metadata.get_operation.return_value = {
            "_id": "op:bad-op",
            "type": "operation",
        }

        with pytest.raises(RepartitionError, match="no target_partition"):
            await engine.rollback_operation("op:bad-op")

    @pytest.mark.asyncio
    async def test_rollback_parent_not_found(
        self, engine, mock_metadata, sample_op
    ):
        """Rollback raises RepartitionError when parent partition not found."""
        mock_metadata.get_operation.return_value = sample_op
        mock_metadata.get_partition.return_value = None

        with pytest.raises(RepartitionError, match="Parent partition.*not found"):
            await engine.rollback_operation("op:split-electronics-123")

    @pytest.mark.asyncio
    async def test_rollback_child_delete_failure_continues(
        self,
        engine,
        mock_metadata,
        sample_op,
        parent_partition,
    ):
        """Rollback continues even if deleting a child fails."""
        mock_metadata.get_operation.return_value = sample_op

        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]
        mock_metadata.get_partition.return_value = parent_with_children

        # First delete fails, second succeeds
        engine.provisioner.delete_partition.side_effect = [
            Exception("delete failed"),
            None,
        ]

        # Should not raise, but log warning
        await engine.rollback_operation("op:split-electronics-123")

        # Parent should still be reset
        save_calls = mock_metadata.save_partition.call_args_list
        assert len(save_calls) == 1
        saved_partition = save_calls[0].args[0]
        assert saved_partition.status == PartitionStatus.ACTIVE
        assert saved_partition.child_partitions == []


# ---------------------------------------------------------------------------
# Step handler tests
# ---------------------------------------------------------------------------


class TestStepCreateChildren:
    """Tests for _step_create_children."""

    @pytest.mark.asyncio
    async def test_step_create_children(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
    ):
        """Creates children from distinct values, marks parent SPLITTING."""
        mock_backend.get_distinct_values.return_value = ["phones", "laptops"]

        # get_partition returns parent initially, then children after provisioning
        child_phones = PartitionInfo(
            name="electronics_subcategory_phones",
            index_name="svr_test_idx_electronics_subcategory_phones",
            view_name="svr_test_partition_electronics_subcategory_phones",
            filter_value="phones",
            status=PartitionStatus.ACTIVE,
        )
        child_laptops = PartitionInfo(
            name="electronics_subcategory_laptops",
            index_name="svr_test_idx_electronics_subcategory_laptops",
            view_name="svr_test_partition_electronics_subcategory_laptops",
            filter_value="laptops",
            status=PartitionStatus.ACTIVE,
        )

        call_map = {
            "electronics": copy.deepcopy(parent_partition),
            "electronics_subcategory_phones": child_phones,
            "electronics_subcategory_laptops": child_laptops,
        }

        async def get_partition_side_effect(name):
            return call_map.get(name)

        mock_metadata.get_partition.side_effect = get_partition_side_effect

        await engine._step_create_children(sample_op)

        # Verify parent marked as SPLITTING (first save_partition call)
        save_calls = mock_metadata.save_partition.call_args_list
        first_save = save_calls[0].args[0]
        assert first_save.status == PartitionStatus.SPLITTING
        assert first_save.name == "electronics"

        # Verify provisioner.create_partition called for each child
        create_calls = engine.provisioner.create_partition.call_args_list
        assert len(create_calls) == 2
        created_names = [c.kwargs.get("name", c.args[0] if c.args else None) for c in create_calls]
        assert "electronics_subcategory_phones" in created_names
        assert "electronics_subcategory_laptops" in created_names

        # Verify parent child_partitions updated (last save for parent)
        # Find last save call for the parent
        parent_saves = [c for c in save_calls if c.args[0].name == "electronics"]
        last_parent_save = parent_saves[-1].args[0]
        assert "electronics_subcategory_phones" in last_parent_save.child_partitions
        assert "electronics_subcategory_laptops" in last_parent_save.child_partitions

    @pytest.mark.asyncio
    async def test_step_create_children_unknown_strategy(
        self, engine, mock_metadata, parent_partition
    ):
        """Unknown strategy raises RepartitionError."""
        op = {
            "_id": "op:split-test",
            "target_partition": "electronics",
            "strategy": "unknown_strategy",
            "strategy_config": {},
        }
        mock_metadata.get_partition.return_value = copy.deepcopy(parent_partition)

        with pytest.raises(RepartitionError, match="Unknown splitting strategy"):
            await engine._step_create_children(op)

    @pytest.mark.asyncio
    async def test_step_create_children_missing_secondary_field(
        self, engine, mock_metadata, parent_partition
    ):
        """secondary_field strategy without field config raises RepartitionError."""
        op = {
            "_id": "op:split-test",
            "target_partition": "electronics",
            "strategy": "secondary_field",
            "strategy_config": {},
        }
        mock_metadata.get_partition.return_value = copy.deepcopy(parent_partition)

        with pytest.raises(RepartitionError, match="secondary_field.*requires"):
            await engine._step_create_children(op)


class TestStepBuildIndexes:
    """Tests for _step_build_indexes."""

    @pytest.mark.asyncio
    async def test_step_build_indexes(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
        child_partition_a,
        child_partition_b,
    ):
        """Verifies get_index_status called for each child partition."""
        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            elif name == "electronics_subcategory_laptops":
                return copy.deepcopy(child_partition_b)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect
        mock_backend.get_index_status.return_value = {"queryable": True}

        await engine._step_build_indexes(sample_op)

        # get_index_status should be called once per child
        assert mock_backend.get_index_status.call_count == 2

    @pytest.mark.asyncio
    async def test_step_build_indexes_index_not_found(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
        child_partition_a,
    ):
        """Missing index raises RepartitionError."""
        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect
        mock_backend.get_index_status.return_value = None

        with pytest.raises(RepartitionError, match="Index.*not found"):
            await engine._step_build_indexes(sample_op)

    @pytest.mark.asyncio
    async def test_step_build_indexes_child_not_found(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
    ):
        """Child partition not found raises RepartitionError."""
        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect

        with pytest.raises(RepartitionError, match="Child partition.*not found"):
            await engine._step_build_indexes(sample_op)


class TestStepWaitIndexes:
    """Tests for _step_wait_indexes."""

    @pytest.mark.asyncio
    async def test_step_wait_indexes(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
        child_partition_a,
        child_partition_b,
    ):
        """Indexes immediately queryable completes without timeout."""
        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            elif name == "electronics_subcategory_laptops":
                return copy.deepcopy(child_partition_b)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect
        mock_backend.get_index_status.return_value = {"queryable": True}

        # Should complete immediately without error
        await engine._step_wait_indexes(sample_op)

        # get_index_status called for both children
        assert mock_backend.get_index_status.call_count == 2

    @pytest.mark.asyncio
    async def test_step_wait_indexes_timeout(
        self,
        sample_config,
        mock_backend,
        mock_metadata,
        sample_op,
        parent_partition,
        child_partition_a,
    ):
        """Indexes never become queryable triggers timeout."""
        # Use very short timeout
        sample_config.lifecycle.repartition.index_wait_timeout_s = 0.1
        sample_config.lifecycle.repartition.index_poll_interval_s = 0.05

        with patch(
            "semantic_vector_router.lifecycle.repartition.PartitionProvisioner"
        ) as MockProv:
            MockProv.return_value = AsyncMock()
            eng = RepartitionEngine(mock_backend, mock_metadata, sample_config)

        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect
        mock_backend.get_index_status.return_value = {"queryable": False}

        with pytest.raises(RepartitionError, match="Timeout"):
            await eng._step_wait_indexes(sample_op)

    @pytest.mark.asyncio
    async def test_step_wait_indexes_becomes_queryable_after_poll(
        self,
        sample_config,
        mock_backend,
        mock_metadata,
        sample_op,
        parent_partition,
        child_partition_a,
    ):
        """Index becomes queryable after one poll cycle."""
        sample_config.lifecycle.repartition.index_wait_timeout_s = 5
        sample_config.lifecycle.repartition.index_poll_interval_s = 0.01

        with patch(
            "semantic_vector_router.lifecycle.repartition.PartitionProvisioner"
        ) as MockProv:
            MockProv.return_value = AsyncMock()
            eng = RepartitionEngine(mock_backend, mock_metadata, sample_config)

        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect

        # First call: not queryable. Second call: queryable.
        mock_backend.get_index_status.side_effect = [
            {"queryable": False},
            {"queryable": True},
        ]

        await eng._step_wait_indexes(sample_op)

        assert mock_backend.get_index_status.call_count == 2

    @pytest.mark.asyncio
    async def test_step_wait_indexes_index_disappears(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
        child_partition_a,
    ):
        """Index disappearing during wait raises RepartitionError."""
        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_partition_a)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect
        mock_backend.get_index_status.return_value = None

        with pytest.raises(RepartitionError, match="disappeared"):
            await engine._step_wait_indexes(sample_op)


class TestStepSwitchRouting:
    """Tests for _step_switch_routing."""

    @pytest.mark.asyncio
    async def test_step_switch_routing(
        self,
        engine,
        mock_metadata,
        sample_op,
        parent_partition,
        child_partition_a,
        child_partition_b,
    ):
        """Parent marked RETIRED, children marked ACTIVE."""
        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
            "electronics_subcategory_laptops",
        ]

        # Children start as non-ACTIVE to verify they get set
        child_a = copy.deepcopy(child_partition_a)
        child_a.status = PartitionStatus.PENDING_SPLIT
        child_b = copy.deepcopy(child_partition_b)
        child_b.status = PartitionStatus.PENDING_SPLIT

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            elif name == "electronics_subcategory_phones":
                return copy.deepcopy(child_a)
            elif name == "electronics_subcategory_laptops":
                return copy.deepcopy(child_b)
            return None

        mock_metadata.get_partition.side_effect = get_partition_side_effect

        await engine._step_switch_routing(sample_op)

        # Verify save_partition called for parent + 2 children = 3 calls
        save_calls = mock_metadata.save_partition.call_args_list
        assert len(save_calls) == 3

        # First save is parent -> RETIRED
        parent_save = save_calls[0].args[0]
        assert parent_save.name == "electronics"
        assert parent_save.status == PartitionStatus.RETIRED

        # Remaining saves are children -> ACTIVE
        child_saves = [c.args[0] for c in save_calls[1:]]
        child_save_names = [c.name for c in child_saves]
        assert "electronics_subcategory_phones" in child_save_names
        assert "electronics_subcategory_laptops" in child_save_names
        for child_save in child_saves:
            assert child_save.status == PartitionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_step_switch_routing_parent_not_found(
        self, engine, mock_metadata, sample_op
    ):
        """Parent not found raises RepartitionError."""
        mock_metadata.get_partition.return_value = None

        with pytest.raises(RepartitionError, match="Parent partition.*not found"):
            await engine._step_switch_routing(sample_op)

    @pytest.mark.asyncio
    async def test_step_switch_routing_child_not_found(
        self, engine, mock_metadata, sample_op, parent_partition
    ):
        """Child not found during routing switch raises RepartitionError."""
        parent_with_children = copy.deepcopy(parent_partition)
        parent_with_children.child_partitions = [
            "electronics_subcategory_phones",
        ]

        async def get_partition_side_effect(name):
            if name == "electronics":
                return copy.deepcopy(parent_with_children)
            return None  # child not found

        mock_metadata.get_partition.side_effect = get_partition_side_effect

        with pytest.raises(RepartitionError, match="Child partition.*not found"):
            await engine._step_switch_routing(sample_op)


class TestStepCleanupParent:
    """Tests for _step_cleanup_parent."""

    @pytest.mark.asyncio
    async def test_step_cleanup_parent(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
    ):
        """Auto cleanup deletes index and view (VIEWS mode)."""
        # Default config has auto_cleanup_retired = True and index_on = VIEWS
        engine.config.lifecycle.repartition.auto_cleanup_retired = True
        engine.config.vector_storage.index_on = IndexLocation.VIEWS

        mock_metadata.get_partition.return_value = copy.deepcopy(parent_partition)

        await engine._step_cleanup_parent(sample_op)

        # Verify index deletion
        mock_backend.delete_index.assert_called_once_with(
            parent_partition.view_name,
            parent_partition.index_name,
        )

        # Verify view deletion (VIEWS mode)
        mock_backend.delete_view.assert_called_once_with(
            parent_partition.view_name
        )

    @pytest.mark.asyncio
    async def test_step_cleanup_parent_disabled(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
    ):
        """Auto cleanup disabled skips all cleanup."""
        engine.config.lifecycle.repartition.auto_cleanup_retired = False

        await engine._step_cleanup_parent(sample_op)

        # No partition lookup or backend calls should happen
        mock_metadata.get_partition.assert_not_called()
        mock_backend.delete_index.assert_not_called()
        mock_backend.delete_view.assert_not_called()

    @pytest.mark.asyncio
    async def test_step_cleanup_parent_source_mode_no_view_delete(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
    ):
        """In SOURCE mode, view is NOT deleted (only index)."""
        engine.config.lifecycle.repartition.auto_cleanup_retired = True
        engine.config.vector_storage.index_on = IndexLocation.SOURCE

        mock_metadata.get_partition.return_value = copy.deepcopy(parent_partition)

        await engine._step_cleanup_parent(sample_op)

        # Index deletion still happens
        mock_backend.delete_index.assert_called_once()

        # View deletion should NOT happen in SOURCE mode
        mock_backend.delete_view.assert_not_called()

    @pytest.mark.asyncio
    async def test_step_cleanup_parent_with_search_collection(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
    ):
        """Cleanup uses search_collection when set instead of view_name."""
        engine.config.lifecycle.repartition.auto_cleanup_retired = True
        engine.config.vector_storage.index_on = IndexLocation.VIEWS

        parent = PartitionInfo(
            name="electronics",
            index_name="svr_test_idx_electronics",
            view_name="svr_test_partition_electronics",
            filter_value="electronics",
            status=PartitionStatus.RETIRED,
            search_collection="custom_collection",
        )
        mock_metadata.get_partition.return_value = parent

        await engine._step_cleanup_parent(sample_op)

        # Should use search_collection for index deletion
        mock_backend.delete_index.assert_called_once_with(
            "custom_collection",
            "svr_test_idx_electronics",
        )

    @pytest.mark.asyncio
    async def test_step_cleanup_parent_not_found_no_error(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
    ):
        """Cleanup with missing parent just logs warning, no error."""
        engine.config.lifecycle.repartition.auto_cleanup_retired = True
        mock_metadata.get_partition.return_value = None

        # Should not raise
        await engine._step_cleanup_parent(sample_op)

        mock_backend.delete_index.assert_not_called()
        mock_backend.delete_view.assert_not_called()

    @pytest.mark.asyncio
    async def test_step_cleanup_parent_delete_index_error_nonfatal(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
    ):
        """Index deletion failure is non-fatal (best effort)."""
        engine.config.lifecycle.repartition.auto_cleanup_retired = True
        engine.config.vector_storage.index_on = IndexLocation.VIEWS

        mock_metadata.get_partition.return_value = copy.deepcopy(parent_partition)
        mock_backend.delete_index.side_effect = Exception("index delete failed")

        # Should not raise — cleanup is best-effort
        await engine._step_cleanup_parent(sample_op)

        # View deletion should still be attempted
        mock_backend.delete_view.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_cleanup_parent_delete_view_error_nonfatal(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
        parent_partition,
    ):
        """View deletion failure is non-fatal (best effort)."""
        engine.config.lifecycle.repartition.auto_cleanup_retired = True
        engine.config.vector_storage.index_on = IndexLocation.VIEWS

        mock_metadata.get_partition.return_value = copy.deepcopy(parent_partition)
        mock_backend.delete_view.side_effect = Exception("view delete failed")

        # Should not raise
        await engine._step_cleanup_parent(sample_op)

        # Index deletion should still have been attempted
        mock_backend.delete_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_cleanup_parent_no_index_name(
        self,
        engine,
        mock_metadata,
        mock_backend,
        sample_op,
    ):
        """Parent with no index_name skips index deletion."""
        engine.config.lifecycle.repartition.auto_cleanup_retired = True
        engine.config.vector_storage.index_on = IndexLocation.VIEWS

        parent = PartitionInfo(
            name="electronics",
            index_name="",  # Empty index_name — falsy
            view_name="svr_test_partition_electronics",
            filter_value="electronics",
            status=PartitionStatus.RETIRED,
        )
        # PartitionInfo requires index_name, so we manually set it
        parent.index_name = ""
        mock_metadata.get_partition.return_value = parent

        await engine._step_cleanup_parent(sample_op)

        mock_backend.delete_index.assert_not_called()
