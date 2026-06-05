"""Unit tests for the PartitionSplitter class."""

import warnings
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from semantic_vector_router.exceptions import SplitError
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.lifecycle.splitter import PartitionSplitter
from semantic_vector_router.models import (
    AutoSplitConfig,
    PartitionInfo,
    PartitionStatus,
    SplitStrategy,
    SVRConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def splitter_config(sample_config: SVRConfig) -> SVRConfig:
    """Build a config wired for split tests.

    Starts with the shared ``sample_config`` fixture (partitioning.field = "category")
    and adds lifecycle / auto_split settings plus a large "electronics" partition.
    """
    sample_config.lifecycle.auto_split = AutoSplitConfig(
        enabled=True,
        split_strategy=SplitStrategy.SECONDARY_FIELD,
        secondary_field="subcategory",
        threshold_vectors=10_000_000,
    )

    sample_config.partitions.registry["electronics"] = PartitionInfo(
        name="electronics",
        view_name="svr_partition_electronics",
        index_name="svr_idx_electronics",
        filter_value="electronics",
        document_count=15_000_000,
        status=PartitionStatus.ACTIVE,
    )

    return sample_config


@pytest.fixture
def mock_provisioner() -> AsyncMock:
    """Return an async mock of PartitionProvisioner."""
    provisioner = AsyncMock(spec=PartitionProvisioner)
    provisioner.create_partition = AsyncMock()
    return provisioner


@pytest.fixture
def splitter(
    mock_backend: AsyncMock,
    splitter_config: SVRConfig,
    mock_provisioner: AsyncMock,
) -> PartitionSplitter:
    """Return a PartitionSplitter wired with mocks."""
    return PartitionSplitter(
        backend=mock_backend,
        config=splitter_config,
        provisioner=mock_provisioner,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSplitBySecondaryField:
    """Tests for the SECONDARY_FIELD split strategy."""

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_split_by_secondary_field_passes_partition_filter(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
    ) -> None:
        """Regression Phase 1.9: get_distinct_values must be called with a
        filter_expression scoped to the parent partition so that child values
        are not pulled from unrelated partitions.
        """
        mock_backend.get_distinct_values = AsyncMock(
            return_value=["phones", "laptops"]
        )

        await splitter.execute_split("electronics")

        mock_backend.get_distinct_values.assert_called_once_with(
            "subcategory",
            filter_expression={"category": "electronics"},
        )

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_split_by_secondary_field_uses_filter_expression_when_present(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
    ) -> None:
        """When a partition already carries a custom filter_expression,
        get_distinct_values should forward it rather than building one from
        filter_value.
        """
        custom_filter = {"category": "electronics", "brand": "acme"}
        splitter.config.partitions.registry["electronics"].filter_expression = (
            custom_filter
        )
        mock_backend.get_distinct_values = AsyncMock(
            return_value=["phones"]
        )

        await splitter.execute_split("electronics")

        mock_backend.get_distinct_values.assert_called_once_with(
            "subcategory",
            filter_expression=custom_filter,
        )

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_split_by_secondary_field_creates_children(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
        mock_provisioner: AsyncMock,
    ) -> None:
        """Child partitions are named {parent}__{value} and provisioned."""
        mock_backend.get_distinct_values = AsyncMock(
            return_value=["phones", "laptops"]
        )

        children = await splitter.execute_split("electronics")

        assert children == ["electronics__phones", "electronics__laptops"]
        assert mock_provisioner.create_partition.call_count == 2

        # Verify the first child's filter expression
        first_call_kwargs = mock_provisioner.create_partition.call_args_list[0].kwargs
        assert first_call_kwargs["name"] == "electronics__phones"
        assert first_call_kwargs["filter_expression"] == {
            "category": "electronics",
            "subcategory": "phones",
        }


class TestSplitByHash:
    """Tests for the (deprecated) HASH split strategy."""

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_hash_split_emits_deprecation_warning(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
    ) -> None:
        """Regression Phase 1.2: executing a hash split must emit a
        DeprecationWarning mentioning 'Hash split'.
        """
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.HASH
        splitter.config.lifecycle.auto_split.num_shards = 2

        with pytest.warns(DeprecationWarning, match="Hash split"):
            await splitter.execute_split("electronics")

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_hash_split_creates_correct_shard_names(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
    ) -> None:
        """Hash shards are named {parent}__shard_{n}."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.HASH
        splitter.config.lifecycle.auto_split.num_shards = 3

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            children = await splitter.execute_split("electronics")

        assert children == [
            "electronics__shard_0",
            "electronics__shard_1",
            "electronics__shard_2",
        ]


class TestSplitByTime:
    """Tests for the TIME split strategy."""

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_time_split_yearly_creates_correct_children(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
        mock_provisioner: AsyncMock,
    ) -> None:
        """Yearly time buckets produce children named {parent}_{year}."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.TIME
        splitter.config.lifecycle.auto_split.time_field = "created_at"
        splitter.config.lifecycle.auto_split.bucket = "yearly"

        # Mock aggregation result
        agg_result = [{"_id": None, "min_date": datetime(2023, 3, 15, tzinfo=timezone.utc),
                        "max_date": datetime(2025, 8, 20, tzinfo=timezone.utc), "count": 5000}]
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=agg_result)
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_backend.get_collection = lambda: mock_collection

        children = await splitter.execute_split("electronics")

        assert children == ["electronics_2023", "electronics_2024", "electronics_2025"]
        assert mock_provisioner.create_partition.call_count == 3

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_time_split_monthly_creates_correct_children(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
        mock_provisioner: AsyncMock,
    ) -> None:
        """Monthly time buckets produce children named {parent}_{YYYY_MM}."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.TIME
        splitter.config.lifecycle.auto_split.time_field = "published_at"
        splitter.config.lifecycle.auto_split.bucket = "monthly"

        agg_result = [{"_id": None, "min_date": datetime(2025, 10, 5, tzinfo=timezone.utc),
                        "max_date": datetime(2026, 2, 15, tzinfo=timezone.utc), "count": 3000}]
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=agg_result)
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_backend.get_collection = lambda: mock_collection

        children = await splitter.execute_split("electronics")

        expected = [
            "electronics_2025_10", "electronics_2025_11", "electronics_2025_12",
            "electronics_2026_01", "electronics_2026_02",
        ]
        assert children == expected

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_time_split_quarterly_creates_correct_children(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
        mock_provisioner: AsyncMock,
    ) -> None:
        """Quarterly time buckets produce children named {parent}_{YYYY_QN}."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.TIME
        splitter.config.lifecycle.auto_split.time_field = "created_at"
        splitter.config.lifecycle.auto_split.bucket = "quarterly"

        agg_result = [{"_id": None, "min_date": datetime(2025, 1, 10, tzinfo=timezone.utc),
                        "max_date": datetime(2025, 9, 20, tzinfo=timezone.utc), "count": 4000}]
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=agg_result)
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_backend.get_collection = lambda: mock_collection

        children = await splitter.execute_split("electronics")

        expected = ["electronics_2025_Q1", "electronics_2025_Q2", "electronics_2025_Q3"]
        assert children == expected

    @pytest.mark.asyncio
    async def test_time_split_empty_data_raises(
        self,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
    ) -> None:
        """SplitError when no documents have the time field."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.TIME
        splitter.config.lifecycle.auto_split.time_field = "created_at"
        splitter.config.lifecycle.auto_split.bucket = "yearly"

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_backend.get_collection = lambda: mock_collection

        with pytest.raises(SplitError, match="No documents"):
            await splitter.execute_split("electronics")

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_time_split_filter_expression_includes_parent(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
        mock_provisioner: AsyncMock,
    ) -> None:
        """Child filter expressions include the parent partition filter."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.TIME
        splitter.config.lifecycle.auto_split.time_field = "created_at"
        splitter.config.lifecycle.auto_split.bucket = "yearly"

        agg_result = [{"_id": None, "min_date": datetime(2025, 6, 1, tzinfo=timezone.utc),
                        "max_date": datetime(2025, 6, 1, tzinfo=timezone.utc), "count": 100}]
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=agg_result)
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_backend.get_collection = lambda: mock_collection

        await splitter.execute_split("electronics")

        call_kwargs = mock_provisioner.create_partition.call_args_list[0].kwargs
        assert call_kwargs["filter_expression"]["category"] == "electronics"
        assert "$gte" in call_kwargs["filter_expression"]["created_at"]
        assert "$lt" in call_kwargs["filter_expression"]["created_at"]

    @pytest.mark.asyncio
    async def test_time_split_null_min_date_raises(
        self,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
    ) -> None:
        """SplitError when aggregation returns null min_date."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.TIME
        splitter.config.lifecycle.auto_split.time_field = "created_at"
        splitter.config.lifecycle.auto_split.bucket = "yearly"

        agg_result = [{"_id": None, "min_date": None, "max_date": None, "count": 0}]
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=agg_result)
        mock_collection = AsyncMock()
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)
        mock_backend.get_collection = lambda: mock_collection

        with pytest.raises(SplitError, match="No documents"):
            await splitter.execute_split("electronics")


class TestStrategyValidation:
    """Tests for configuration validation when executing splits."""

    @pytest.mark.asyncio
    async def test_split_strategy_validation_secondary_field_requires_field(
        self,
        splitter: PartitionSplitter,
    ) -> None:
        """SplitError is raised when secondary_field strategy is used but no
        secondary_field is configured.
        """
        splitter.config.lifecycle.auto_split.secondary_field = None
        splitter.config.lifecycle.auto_split.split_strategy = (
            SplitStrategy.SECONDARY_FIELD
        )

        with pytest.raises(SplitError, match="Secondary field not configured"):
            await splitter.execute_split("electronics")

    @pytest.mark.asyncio
    async def test_split_strategy_validation_time_requires_time_field(
        self,
        splitter: PartitionSplitter,
    ) -> None:
        """SplitError is raised when time strategy is used but no time_field
        is configured.
        """
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.TIME
        splitter.config.lifecycle.auto_split.time_field = None

        with pytest.raises(SplitError, match="Time field not configured"):
            await splitter.execute_split("electronics")


class TestAlertOnly:
    """Tests for the ALERT_ONLY strategy."""

    @pytest.mark.asyncio
    async def test_alert_only_returns_empty(
        self,
        splitter: PartitionSplitter,
    ) -> None:
        """ALERT_ONLY strategy returns an empty list without creating children."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.ALERT_ONLY

        result = await splitter.execute_split("electronics")

        assert result == []


class TestParentStatus:
    """Tests for parent partition status after split."""

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_split_marks_parent_as_split(
        self,
        _mock_save: AsyncMock,
        splitter: PartitionSplitter,
        mock_backend: AsyncMock,
    ) -> None:
        """After a successful split the parent partition status must be
        PartitionStatus.SPLIT and save_config must be called to persist.
        """
        mock_backend.get_distinct_values = AsyncMock(
            return_value=["phones", "laptops"]
        )

        await splitter.execute_split("electronics")

        parent = splitter.config.partitions.registry["electronics"]
        assert parent.status == PartitionStatus.SPLIT
        assert parent.child_partitions == [
            "electronics__phones",
            "electronics__laptops",
        ]
        _mock_save.assert_called_once_with(splitter.config)

    @pytest.mark.asyncio
    async def test_alert_only_does_not_mark_parent_as_split(
        self,
        splitter: PartitionSplitter,
    ) -> None:
        """ALERT_ONLY should leave the parent partition status unchanged."""
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.ALERT_ONLY

        await splitter.execute_split("electronics")

        parent = splitter.config.partitions.registry["electronics"]
        assert parent.status == PartitionStatus.ACTIVE


class TestExecuteSplitEdgeCases:
    """Edge-case tests for execute_split."""

    @pytest.mark.asyncio
    async def test_split_unknown_partition_raises(
        self,
        splitter: PartitionSplitter,
    ) -> None:
        """Splitting a partition that does not exist raises SplitError."""
        with pytest.raises(SplitError, match="not found"):
            await splitter.execute_split("nonexistent")


# ===========================================================================
# check_and_split
# ===========================================================================


class TestCheckAndSplit:
    @pytest.mark.asyncio
    async def test_check_and_split_disabled_no_config(self, splitter):
        splitter.config.lifecycle.auto_split = None
        result = await splitter.check_and_split()
        assert result == []

    @pytest.mark.asyncio
    async def test_check_and_split_disabled_not_enabled(self, splitter):
        splitter.config.lifecycle.auto_split.enabled = False
        result = await splitter.check_and_split()
        assert result == []

    @pytest.mark.asyncio
    async def test_check_and_split_below_threshold(self, splitter, mock_backend):
        mock_backend.count_documents = AsyncMock(return_value=5_000_000)
        result = await splitter.check_and_split()
        assert result == []

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_check_and_split_above_threshold_within_schedule(
        self, _mock_save, splitter, mock_backend,
    ):
        mock_backend.count_documents = AsyncMock(return_value=15_000_000)
        mock_backend.get_distinct_values = AsyncMock(return_value=["phones"])

        with patch.object(splitter, "_is_within_schedule", return_value=True):
            result = await splitter.check_and_split()

        assert "electronics" in result

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_check_and_split_above_threshold_outside_schedule(
        self, _mock_save, splitter, mock_backend,
    ):
        mock_backend.count_documents = AsyncMock(return_value=15_000_000)

        with patch.object(splitter, "_is_within_schedule", return_value=False), \
             patch.object(splitter, "_mark_pending_split", new_callable=AsyncMock) as mock_mark:
            result = await splitter.check_and_split()

        mock_mark.assert_awaited_once_with("electronics")
        assert result == []

    @pytest.mark.asyncio
    async def test_check_and_split_skips_split_partitions(self, splitter, mock_backend):
        splitter.config.partitions.registry["electronics"].status = PartitionStatus.SPLIT
        mock_backend.count_documents = AsyncMock(return_value=15_000_000)
        result = await splitter.check_and_split()
        assert result == []
        mock_backend.count_documents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_check_and_split_skips_disabled_partitions(self, splitter, mock_backend):
        splitter.config.partitions.registry["electronics"].status = PartitionStatus.DISABLED
        mock_backend.count_documents = AsyncMock(return_value=15_000_000)
        result = await splitter.check_and_split()
        assert result == []


# ===========================================================================
# _mark_pending_split
# ===========================================================================


class TestMarkPendingSplit:
    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_mark_pending_split(self, mock_save, splitter):
        await splitter._mark_pending_split("electronics")
        partition = splitter.config.partitions.registry["electronics"]
        assert partition.status == PartitionStatus.PENDING_SPLIT
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_mark_pending_split_nonexistent(self, mock_save, splitter):
        await splitter._mark_pending_split("nonexistent")
        mock_save.assert_not_called()


# ===========================================================================
# _is_within_schedule
# ===========================================================================


class TestIsWithinSchedule:
    def test_no_auto_split_config(self, splitter):
        splitter.config.lifecycle.auto_split = None
        assert splitter._is_within_schedule() is True

    def test_no_schedule(self, splitter):
        splitter.config.lifecycle.auto_split.schedule = None
        assert splitter._is_within_schedule() is True

    def test_allowed_day_matches(self, splitter):
        from unittest.mock import MagicMock as MM
        schedule = MM()
        schedule.allowed_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        schedule.allowed_hours = None
        splitter.config.lifecycle.auto_split.schedule = schedule
        assert splitter._is_within_schedule() is True

    def test_disallowed_day(self, splitter):
        from unittest.mock import MagicMock as MM
        schedule = MM()
        # Use a day name that can never match the current day
        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        current_day = datetime.now(timezone.utc).strftime("%A").lower()
        schedule.allowed_days = [d for d in all_days if d != current_day]
        schedule.allowed_hours = None
        splitter.config.lifecycle.auto_split.schedule = schedule
        assert splitter._is_within_schedule() is False

    def test_hour_in_range(self, splitter):
        from unittest.mock import MagicMock as MM
        schedule = MM()
        schedule.allowed_days = None
        schedule.allowed_hours = {"start": 0, "end": 24}  # All hours
        splitter.config.lifecycle.auto_split.schedule = schedule
        assert splitter._is_within_schedule() is True

    def test_hour_out_of_range(self, splitter):
        from unittest.mock import MagicMock as MM
        schedule = MM()
        schedule.allowed_days = None
        current_hour = datetime.now(timezone.utc).hour
        # Set a range that excludes the current hour
        if current_hour < 12:
            schedule.allowed_hours = {"start": 13, "end": 14}
        else:
            schedule.allowed_hours = {"start": 1, "end": 2}
        splitter.config.lifecycle.auto_split.schedule = schedule
        assert splitter._is_within_schedule() is False


# ===========================================================================
# get_pending_splits
# ===========================================================================


class TestGetPendingSplits:
    @pytest.mark.asyncio
    async def test_no_pending(self, splitter):
        result = await splitter.get_pending_splits()
        assert result == []

    @pytest.mark.asyncio
    async def test_one_pending(self, splitter):
        splitter.config.partitions.registry["electronics"].status = PartitionStatus.PENDING_SPLIT
        result = await splitter.get_pending_splits()
        assert result == ["electronics"]

    @pytest.mark.asyncio
    async def test_mixed_statuses(self, splitter):
        splitter.config.partitions.registry["electronics"].status = PartitionStatus.PENDING_SPLIT
        splitter.config.partitions.registry["active_one"] = PartitionInfo(
            name="active_one", view_name="v", index_name="i",
            filter_value="active_one", status=PartitionStatus.ACTIVE,
        )
        result = await splitter.get_pending_splits()
        assert result == ["electronics"]


# ===========================================================================
# execute_pending_splits
# ===========================================================================


class TestExecutePendingSplits:
    @pytest.mark.asyncio
    async def test_execute_pending_empty(self, splitter):
        result = await splitter.execute_pending_splits()
        assert result == []

    @pytest.mark.asyncio
    @patch("semantic_vector_router.lifecycle.splitter.save_config")
    async def test_execute_pending_success(self, _mock_save, splitter, mock_backend):
        splitter.config.partitions.registry["electronics"].status = PartitionStatus.PENDING_SPLIT
        mock_backend.get_distinct_values = AsyncMock(return_value=["phones"])

        result = await splitter.execute_pending_splits()
        assert "electronics" in result

    @pytest.mark.asyncio
    async def test_execute_pending_handles_error(self, splitter):
        splitter.config.partitions.registry["electronics"].status = PartitionStatus.PENDING_SPLIT

        with patch.object(splitter, "execute_split", side_effect=Exception("fail")):
            result = await splitter.execute_pending_splits()

        assert result == []

    @pytest.mark.asyncio
    async def test_execute_pending_alert_only_no_children(self, splitter):
        splitter.config.partitions.registry["electronics"].status = PartitionStatus.PENDING_SPLIT
        splitter.config.lifecycle.auto_split.split_strategy = SplitStrategy.ALERT_ONLY

        result = await splitter.execute_pending_splits()
        assert result == []  # ALERT_ONLY returns empty children


# ===========================================================================
# Time split: non-yearly bucket
# ===========================================================================


class TestGenerateTimeBuckets:
    """Tests for the static _generate_time_buckets method."""

    def test_yearly_single_year(self):
        buckets = PartitionSplitter._generate_time_buckets(
            datetime(2025, 3, 1, tzinfo=timezone.utc),
            datetime(2025, 9, 1, tzinfo=timezone.utc),
            "yearly",
        )
        assert len(buckets) == 1
        start, end, label = buckets[0]
        assert label == "2025"
        assert start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert end == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_yearly_multi_year(self):
        buckets = PartitionSplitter._generate_time_buckets(
            datetime(2022, 6, 1, tzinfo=timezone.utc),
            datetime(2025, 2, 1, tzinfo=timezone.utc),
            "yearly",
        )
        labels = [b[2] for b in buckets]
        assert labels == ["2022", "2023", "2024", "2025"]

    def test_monthly_across_year_boundary(self):
        buckets = PartitionSplitter._generate_time_buckets(
            datetime(2025, 11, 1, tzinfo=timezone.utc),
            datetime(2026, 2, 15, tzinfo=timezone.utc),
            "monthly",
        )
        labels = [b[2] for b in buckets]
        assert labels == ["2025_11", "2025_12", "2026_01", "2026_02"]

    def test_quarterly_full_year(self):
        buckets = PartitionSplitter._generate_time_buckets(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 31, tzinfo=timezone.utc),
            "quarterly",
        )
        labels = [b[2] for b in buckets]
        assert labels == ["2025_Q1", "2025_Q2", "2025_Q3", "2025_Q4"]

    def test_quarterly_cross_year(self):
        buckets = PartitionSplitter._generate_time_buckets(
            datetime(2025, 10, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 15, tzinfo=timezone.utc),
            "quarterly",
        )
        labels = [b[2] for b in buckets]
        assert labels == ["2025_Q4", "2026_Q1"]

    def test_monthly_single_month(self):
        buckets = PartitionSplitter._generate_time_buckets(
            datetime(2025, 6, 5, tzinfo=timezone.utc),
            datetime(2025, 6, 20, tzinfo=timezone.utc),
            "monthly",
        )
        assert len(buckets) == 1
        assert buckets[0][2] == "2025_06"

    def test_bucket_boundaries_are_utc(self):
        buckets = PartitionSplitter._generate_time_buckets(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 3, 1, tzinfo=timezone.utc),
            "monthly",
        )
        for start, end, _ in buckets:
            assert start.tzinfo == timezone.utc
            assert end.tzinfo == timezone.utc
