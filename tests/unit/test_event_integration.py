"""Unit tests for event integration across SVR modules.

Verifies that events are correctly emitted from provisioner, repartition
engine, detector, and client modules when an EventBus is wired in.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.events.models import SVREvent, SVREventType


class MockEventBus:
    """Simple test double that collects emitted events."""

    def __init__(self):
        self.events: list[SVREvent] = []

    async def emit(self, event: SVREvent) -> None:
        self.events.append(event)


# ─── Provisioner Event Tests ─────────────────────────────────────────────────


class TestProvisionerEvents:
    """Events emitted by PartitionProvisioner."""

    def _make_provisioner(self, event_bus=None):
        """Create a PartitionProvisioner with mocked dependencies."""
        from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
        from semantic_vector_router.models import (
            DatabaseConfig,
            EmbeddingConfig,
            EmbeddingMode,
            EmbeddingProvider,
            IndexLocation,
            PartitioningConfig,
            SVRConfig,
            VectorSearchConfig,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=1536),
            embedding=EmbeddingConfig(
                mode=EmbeddingMode.BYOM,
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
                api_key_env="OPENAI_API_KEY",
                dimensions=1536,
            ),
        )

        from semantic_vector_router.models.backend import PartitionStorageResult

        backend = AsyncMock()
        backend.create_partition_storage = AsyncMock(
            return_value=PartitionStorageResult(
                storage_name="vw_products_electronics",
                storage_type="view",
                view_name="vw_products_electronics",
                search_collection="products",
            )
        )
        backend.create_partition_index = AsyncMock(
            return_value="svr_vector_idx_source"
        )
        backend.count_documents = AsyncMock(return_value=1000)
        backend.delete_partition_index = AsyncMock()
        backend.delete_partition_storage = AsyncMock()

        provisioner = PartitionProvisioner(backend, config, auto_save_config=False)
        if event_bus is not None:
            provisioner.set_event_bus(event_bus)
        return provisioner

    @pytest.mark.asyncio
    async def test_create_partition_emits_partition_created(self):
        """create_partition emits a PARTITION_CREATED event on success."""
        bus = MockEventBus()
        provisioner = self._make_provisioner(event_bus=bus)

        await provisioner.create_partition("electronics")

        assert len(bus.events) == 1
        event = bus.events[0]
        assert event.event_type == SVREventType.PARTITION_CREATED
        assert event.partition == "electronics"
        assert event.details["document_count"] == 1000

    @pytest.mark.asyncio
    async def test_delete_partition_emits_partition_deleted(self):
        """delete_partition emits a PARTITION_DELETED event after cleanup."""
        bus = MockEventBus()
        provisioner = self._make_provisioner(event_bus=bus)

        # First create a partition so it exists in registry
        await provisioner.create_partition("electronics")
        bus.events.clear()

        await provisioner.delete_partition("electronics")

        assert len(bus.events) == 1
        event = bus.events[0]
        assert event.event_type == SVREventType.PARTITION_DELETED
        assert event.partition == "electronics"

    @pytest.mark.asyncio
    async def test_no_event_bus_no_error(self):
        """When no event bus is set, create_partition succeeds without error."""
        provisioner = self._make_provisioner(event_bus=None)

        partition = await provisioner.create_partition("furniture")

        assert partition.name == "furniture"
        assert partition.document_count == 1000

    @pytest.mark.asyncio
    async def test_event_bus_exception_does_not_propagate(self):
        """If the event bus raises, the provisioner logs a warning but continues."""
        broken_bus = AsyncMock()
        broken_bus.emit = AsyncMock(side_effect=RuntimeError("bus broken"))

        provisioner = self._make_provisioner(event_bus=broken_bus)

        # Should not raise
        partition = await provisioner.create_partition("electronics")
        assert partition.name == "electronics"


# ─── Repartition Engine Event Tests ──────────────────────────────────────────


class TestRepartitionEvents:
    """Events emitted by RepartitionEngine."""

    def _make_engine(self, event_bus=None, steps=None):
        """Create a RepartitionEngine with mocked dependencies."""
        from semantic_vector_router.lifecycle.repartition import RepartitionEngine
        from semantic_vector_router.models import (
            DatabaseConfig,
            EmbeddingConfig,
            EmbeddingMode,
            EmbeddingProvider,
            PartitioningConfig,
            SVRConfig,
            VectorSearchConfig,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=1536),
            embedding=EmbeddingConfig(
                mode=EmbeddingMode.BYOM,
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
                api_key_env="OPENAI_API_KEY",
                dimensions=1536,
            ),
        )

        backend = AsyncMock()
        metadata = AsyncMock()

        if steps is None:
            steps = []

        metadata.get_operation = AsyncMock(return_value={
            "target_partition": "electronics",
            "strategy": "secondary_field",
            "strategy_config": {},
            "steps": steps,
        })
        metadata.update_operation_step = AsyncMock()
        metadata.update_operation_status = AsyncMock()

        engine = RepartitionEngine(backend, metadata, config, event_bus=event_bus)
        return engine

    @pytest.mark.asyncio
    async def test_execute_operation_emits_started(self):
        """execute_operation emits REPARTITION_STARTED at the beginning."""
        bus = MockEventBus()
        engine = self._make_engine(event_bus=bus, steps=[])

        await engine.execute_operation("op-123")

        started_events = [
            e for e in bus.events
            if e.event_type == SVREventType.REPARTITION_STARTED
        ]
        assert len(started_events) == 1
        assert started_events[0].partition == "electronics"
        assert started_events[0].details["operation_id"] == "op-123"

    @pytest.mark.asyncio
    async def test_execute_operation_emits_completed_on_success(self):
        """execute_operation emits REPARTITION_COMPLETED when all steps succeed."""
        bus = MockEventBus()
        engine = self._make_engine(event_bus=bus, steps=[])

        result = await engine.execute_operation("op-123")

        assert result is True
        completed_events = [
            e for e in bus.events
            if e.event_type == SVREventType.REPARTITION_COMPLETED
        ]
        assert len(completed_events) == 1
        assert completed_events[0].details["operation_id"] == "op-123"

    @pytest.mark.asyncio
    async def test_execute_operation_emits_failed_on_error(self):
        """execute_operation emits REPARTITION_FAILED when a step raises."""
        bus = MockEventBus()
        steps = [{"action": "create_children", "status": "pending"}]
        engine = self._make_engine(event_bus=bus, steps=steps)

        # Make the step handler fail
        engine._step_create_children = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        result = await engine.execute_operation("op-456")

        assert result is False
        failed_events = [
            e for e in bus.events
            if e.event_type == SVREventType.REPARTITION_FAILED
        ]
        assert len(failed_events) == 1
        assert "boom" in failed_events[0].details.get("error", "")

    @pytest.mark.asyncio
    async def test_rollback_operation_emits_rolled_back(self):
        """rollback_operation emits REPARTITION_ROLLED_BACK event."""
        bus = MockEventBus()
        engine = self._make_engine(event_bus=bus)

        # Mock the parent partition with child_partitions
        mock_parent = MagicMock()
        mock_parent.child_partitions = []
        engine.metadata.get_partition = AsyncMock(return_value=mock_parent)
        engine.metadata.get_operation = AsyncMock(return_value={
            "target_partition": "electronics",
        })
        engine.metadata.save_partition = AsyncMock()

        await engine.rollback_operation("op-789")

        rolled_back_events = [
            e for e in bus.events
            if e.event_type == SVREventType.REPARTITION_ROLLED_BACK
        ]
        assert len(rolled_back_events) == 1
        assert rolled_back_events[0].partition == "electronics"
        assert rolled_back_events[0].details["operation_id"] == "op-789"

    @pytest.mark.asyncio
    async def test_compute_centroids_emits_centroid_computed(self):
        """_step_compute_centroids emits CENTROID_COMPUTED for each child."""
        bus = MockEventBus()
        engine = self._make_engine(event_bus=bus)

        # Setup parent with two children
        from semantic_vector_router.models import PartitionStatus

        mock_parent = MagicMock()
        mock_parent.child_partitions = ["electronics_brand_a", "electronics_brand_b"]

        mock_child_a = MagicMock()
        mock_child_a.status = PartitionStatus.ACTIVE
        mock_child_a.embedding_field = None
        mock_child_a.filter_value = "brand_a"

        mock_child_b = MagicMock()
        mock_child_b.status = PartitionStatus.ACTIVE
        mock_child_b.embedding_field = None
        mock_child_b.filter_value = "brand_b"

        engine.metadata.get_partition = AsyncMock(side_effect=lambda name: {
            "electronics": mock_parent,
            "electronics_brand_a": mock_child_a,
            "electronics_brand_b": mock_child_b,
        }.get(name))
        engine.metadata.update_centroid = AsyncMock()

        # Mock the db collection
        engine.backend.db = {"products": MagicMock()}

        op = {"target_partition": "electronics"}

        with patch(
            "semantic_vector_router.routing.centroid.compute_partition_centroid",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3],
        ):
            await engine._step_compute_centroids(op)

        centroid_events = [
            e for e in bus.events
            if e.event_type == SVREventType.CENTROID_COMPUTED
        ]
        assert len(centroid_events) == 2
        partitions = {e.partition for e in centroid_events}
        assert partitions == {"electronics_brand_a", "electronics_brand_b"}

    @pytest.mark.asyncio
    async def test_no_event_bus_repartition_no_error(self):
        """When no event bus is set, execute_operation works fine."""
        engine = self._make_engine(event_bus=None, steps=[])

        result = await engine.execute_operation("op-100")
        assert result is True


# ─── Detector Event Tests ────────────────────────────────────────────────────


class TestDetectorEvents:
    """Events emitted by PartitionDetector."""

    def _make_detector(self, event_bus=None, partitions=None, counts=None):
        """Create a PartitionDetector with mocked dependencies."""
        from semantic_vector_router.lifecycle.detector import PartitionDetector
        from semantic_vector_router.models import (
            DatabaseConfig,
            EmbeddingConfig,
            EmbeddingMode,
            EmbeddingProvider,
            PartitioningConfig,
            SVRConfig,
            VectorSearchConfig,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=1536),
            embedding=EmbeddingConfig(
                mode=EmbeddingMode.BYOM,
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
                api_key_env="OPENAI_API_KEY",
                dimensions=1536,
            ),
        )
        # Use a low threshold for testing
        config.lifecycle.detection.threshold_vectors = 10_000

        backend = AsyncMock()
        metadata = AsyncMock()

        if partitions is None:
            partitions = []
        metadata.list_partitions = AsyncMock(return_value=partitions)
        metadata.get_health_history = AsyncMock(return_value=[])
        metadata.append_health_history = AsyncMock()
        metadata.create_operation = AsyncMock()

        # Set up count_documents to return values from counts dict
        if counts:
            async def _count(coll_name, filter_expr=None):
                # Map back to partition name from the arguments
                for name, count in counts.items():
                    return count
                return 0
            backend.count_documents = AsyncMock(side_effect=lambda *a, **kw: 0)

        detector = PartitionDetector(backend, metadata, config, event_bus=event_bus)
        return detector, backend

    @pytest.mark.asyncio
    async def test_threshold_breach_emits_event(self):
        """Detection finding THRESHOLD_BREACH emits health.threshold_breach."""
        from semantic_vector_router.models import (
            IndexLocation,
            PartitionInfo,
            PartitionStatus,
        )

        bus = MockEventBus()
        partition = PartitionInfo(
            name="electronics",
            view_name="vw_products_electronics",
            index_name="idx_electronics",
            filter_value="electronics",
            document_count=15_000,
            status=PartitionStatus.ACTIVE,
            search_collection="products",
            index_location=IndexLocation.SOURCE,
        )

        detector, backend = self._make_detector(event_bus=bus, partitions=[partition])
        # Return count over threshold
        backend.count_documents = AsyncMock(return_value=15_000)

        results = await detector.run_detection()

        breach_events = [
            e for e in bus.events
            if e.event_type == SVREventType.HEALTH_THRESHOLD_BREACH
        ]
        assert len(breach_events) >= 1
        assert breach_events[0].partition == "electronics"

    @pytest.mark.asyncio
    async def test_approaching_threshold_emits_event(self):
        """Detection finding APPROACHING_THRESHOLD emits health.approaching_threshold."""
        from semantic_vector_router.models import (
            IndexLocation,
            PartitionInfo,
            PartitionStatus,
        )

        bus = MockEventBus()
        partition = PartitionInfo(
            name="furniture",
            view_name="vw_products_furniture",
            index_name="idx_furniture",
            filter_value="furniture",
            document_count=8_000,
            status=PartitionStatus.ACTIVE,
            search_collection="products",
            index_location=IndexLocation.SOURCE,
        )

        detector, backend = self._make_detector(event_bus=bus, partitions=[partition])
        backend.count_documents = AsyncMock(return_value=8_000)

        # Create history with growth trend (will breach within window)
        now = datetime.utcnow()
        history = [
            {"ts": datetime(2025, 1, 1), "count": 5000},
            {"ts": datetime(2025, 1, 10), "count": 6500},
            {"ts": datetime(2025, 1, 20), "count": 8000},
        ]
        detector.metadata.get_health_history = AsyncMock(return_value=history)

        results = await detector.run_detection()

        approach_events = [
            e for e in bus.events
            if e.event_type == SVREventType.HEALTH_APPROACHING_THRESHOLD
        ]
        assert len(approach_events) >= 1
        assert approach_events[0].partition == "furniture"

    @pytest.mark.asyncio
    async def test_severe_skew_emits_event(self):
        """Detection finding SEVERE_SKEW emits health.skew_detected."""
        from semantic_vector_router.models import (
            IndexLocation,
            PartitionInfo,
            PartitionStatus,
        )

        bus = MockEventBus()
        p1 = PartitionInfo(
            name="electronics",
            view_name="vw_electronics",
            index_name="idx_electronics",
            filter_value="electronics",
            document_count=9_000,
            status=PartitionStatus.ACTIVE,
            search_collection="products",
            index_location=IndexLocation.SOURCE,
            parent_partition="_root",
        )
        p2 = PartitionInfo(
            name="furniture",
            view_name="vw_furniture",
            index_name="idx_furniture",
            filter_value="furniture",
            document_count=500,
            status=PartitionStatus.ACTIVE,
            search_collection="products",
            index_location=IndexLocation.SOURCE,
            parent_partition="_root",
        )

        detector, backend = self._make_detector(event_bus=bus, partitions=[p1, p2])

        # Set index_on to SOURCE so _build_filter uses partition field filters
        detector.config.vector_storage.index_on = IndexLocation.SOURCE

        # Return highly skewed counts based on the filter expression
        # electronics=9000, furniture=100 -> avg=4550, max/avg=9000/4550~1.98
        # Use extreme skew: 9000 vs 50 -> avg=4525, ratio=9000/4525~1.99
        # Need ratio > skew_ratio. With counts 9500 vs 100 -> avg=4800, ratio=9500/4800~1.98
        # Better: 9000 vs 10 -> avg=4505, ratio=9000/4505~2.0 (borderline)
        # Use 9000 vs 1 -> avg=4500.5, ratio=9000/4500.5~2.0 (still borderline)
        # Simplest: set skew_ratio to 1.5, keep 9000 vs 500
        async def _mock_count(coll_name, filter_expr=None):
            if filter_expr and filter_expr.get("category") == "electronics":
                return 9000
            elif filter_expr and filter_expr.get("category") == "furniture":
                return 500
            return 0

        backend.count_documents = AsyncMock(side_effect=_mock_count)

        # avg = (9000 + 500) / 2 = 4750, ratio = 9000 / 4750 = 1.89
        # Set skew ratio low enough to trigger
        detector.config.lifecycle.detection.skew_ratio = 1.5

        results = await detector.run_detection()

        skew_events = [
            e for e in bus.events
            if e.event_type == SVREventType.HEALTH_SKEW_DETECTED
        ]
        assert len(skew_events) >= 1

    @pytest.mark.asyncio
    async def test_no_event_bus_detector_no_error(self):
        """When no event bus is set, run_detection succeeds without error."""
        from semantic_vector_router.models import (
            IndexLocation,
            PartitionInfo,
            PartitionStatus,
        )

        partition = PartitionInfo(
            name="electronics",
            view_name="vw_products_electronics",
            index_name="idx_electronics",
            filter_value="electronics",
            document_count=15_000,
            status=PartitionStatus.ACTIVE,
            search_collection="products",
            index_location=IndexLocation.SOURCE,
        )

        detector, backend = self._make_detector(event_bus=None, partitions=[partition])
        backend.count_documents = AsyncMock(return_value=15_000)

        # Should not raise
        results = await detector.run_detection()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_detection_no_partitions_no_events(self):
        """When no partitions exist, run_detection returns [] and no events."""
        bus = MockEventBus()
        detector, backend = self._make_detector(event_bus=bus, partitions=[])

        results = await detector.run_detection()

        assert results == []
        assert len(bus.events) == 0

    @pytest.mark.asyncio
    async def test_detection_emits_health_alert_for_unknown_signals(self):
        """Signals not in the mapping emit health.alert as fallback."""
        from semantic_vector_router.models import (
            IndexLocation,
            PartitionInfo,
            PartitionStatus,
        )

        bus = MockEventBus()
        partition = PartitionInfo(
            name="tiny",
            view_name="vw_tiny",
            index_name="idx_tiny",
            filter_value="tiny",
            document_count=5,
            status=PartitionStatus.ACTIVE,
            search_collection="products",
            index_location=IndexLocation.SOURCE,
        )

        detector, backend = self._make_detector(event_bus=bus, partitions=[partition])
        backend.count_documents = AsyncMock(return_value=5)

        # Set min_threshold high enough to trigger UNDERPOPULATED
        detector.config.lifecycle.detection.min_threshold_vectors = 100

        results = await detector.run_detection()

        # UNDERPOPULATED is not in the signal_to_event mapping,
        # so it falls back to health.alert
        alert_events = [
            e for e in bus.events
            if e.event_type == SVREventType.HEALTH_ALERT
        ]
        assert len(alert_events) >= 1


# ─── Client Event Tests ──────────────────────────────────────────────────────


class TestClientEvents:
    """Events emitted by SVRClient."""

    @pytest.mark.asyncio
    async def test_ingest_success_emits_ingest_completed(self):
        """ingest() emits INGEST_COMPLETED event when documents are inserted."""
        bus = MockEventBus()

        with patch(
            "semantic_vector_router.client.validate_config",
            return_value=[],
        ):
            from semantic_vector_router.models import (
                DatabaseConfig,
                EmbeddingConfig,
                EmbeddingMode,
                EmbeddingProvider,
                PartitioningConfig,
                SVRConfig,
                VectorSearchConfig,
            )

            config = SVRConfig(
                database=DatabaseConfig(
                    connection_string_env="MONGODB_URI",
                    database="test_db",
                    source_collection="products",
                ),
                partitioning=PartitioningConfig(field="category"),
                vector_search=VectorSearchConfig(dimensions=1536),
                embedding=EmbeddingConfig(
                    mode=EmbeddingMode.BYOM,
                    provider=EmbeddingProvider.OPENAI,
                    model="text-embedding-3-small",
                    api_key_env="OPENAI_API_KEY",
                    dimensions=1536,
                ),
            )

            from semantic_vector_router.client import SVRClient

            client = SVRClient(config=config, auto_connect=False)
            client._connected = True
            client._event_bus = bus
            client._backend = MagicMock()
            client._metadata = None  # No metadata store

            # Mock the embedder
            mock_embedder = AsyncMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
            client._embedder = mock_embedder

            # Mock IngestPipeline and _create_document_embedder
            mock_result = MagicMock()
            mock_result.inserted = 5
            mock_result.failed = 0
            mock_result.elapsed_ms = 100.0

            with patch(
                "semantic_vector_router.client.IngestPipeline"
            ) as MockPipeline, patch.object(
                client, "_create_document_embedder", return_value=mock_embedder,
            ):
                mock_pipeline = AsyncMock()
                mock_pipeline.ingest = AsyncMock(return_value=mock_result)
                MockPipeline.return_value = mock_pipeline

                result = await client.ingest(
                    documents=[{"text": "hello"}],
                    partition="electronics",
                )

            assert result.inserted == 5

            ingest_events = [
                e for e in bus.events
                if e.event_type == SVREventType.INGEST_COMPLETED
            ]
            assert len(ingest_events) == 1
            assert ingest_events[0].partition == "electronics"
            assert ingest_events[0].details["inserted"] == 5

    @pytest.mark.asyncio
    async def test_connect_initializes_event_bus_when_enabled(self):
        """connect() creates an EventBus when config.events.enabled=True."""
        from semantic_vector_router.models import (
            DatabaseConfig,
            EmbeddingConfig,
            EmbeddingMode,
            EmbeddingProvider,
            PartitioningConfig,
            SVRConfig,
            VectorSearchConfig,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=1536),
            embedding=EmbeddingConfig(
                mode=EmbeddingMode.BYOM,
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
                api_key_env="OPENAI_API_KEY",
                dimensions=1536,
            ),
        )
        config.events.enabled = True

        with patch(
            "semantic_vector_router.client.validate_config",
            return_value=[],
        ):
            from semantic_vector_router.client import SVRClient

            client = SVRClient(config=config, auto_connect=False)

            with patch(
                "semantic_vector_router.client.create_backend"
            ) as MockBackend, patch(
                "semantic_vector_router.backends.metadata.MetadataStore"
            ) as MockMeta, patch(
                "semantic_vector_router.factories.get_api_key",
                return_value="sk-test",
            ), patch(
                "semantic_vector_router.factories.OpenAIEmbedder"
            ):
                mock_backend = AsyncMock()
                mock_backend._db = MagicMock()
                MockBackend.return_value = mock_backend

                mock_meta = AsyncMock()
                mock_meta.migrate_from_config = AsyncMock(return_value=0)
                MockMeta.return_value = mock_meta

                await client.connect()

            from semantic_vector_router.events.bus import EventBus

            assert client._event_bus is not None
            assert isinstance(client._event_bus, EventBus)

            await client.disconnect()

    @pytest.mark.asyncio
    async def test_connect_creates_scheduler_when_enabled(self):
        """connect() creates a JobScheduler when config.scheduler.enabled=True."""
        from semantic_vector_router.models import (
            DatabaseConfig,
            EmbeddingConfig,
            EmbeddingMode,
            EmbeddingProvider,
            PartitioningConfig,
            SVRConfig,
            VectorSearchConfig,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=1536),
            embedding=EmbeddingConfig(
                mode=EmbeddingMode.BYOM,
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
                api_key_env="OPENAI_API_KEY",
                dimensions=1536,
            ),
        )
        config.scheduler.enabled = True
        config.events.enabled = True

        with patch(
            "semantic_vector_router.client.validate_config",
            return_value=[],
        ):
            from semantic_vector_router.client import SVRClient

            client = SVRClient(config=config, auto_connect=False)

            with patch(
                "semantic_vector_router.client.create_backend"
            ) as MockBackend, patch(
                "semantic_vector_router.backends.metadata.MetadataStore"
            ) as MockMeta, patch(
                "semantic_vector_router.factories.get_api_key",
                return_value="sk-test",
            ), patch(
                "semantic_vector_router.factories.OpenAIEmbedder"
            ), patch(
                "semantic_vector_router.scheduler.engine.JobScheduler"
            ) as MockScheduler:
                mock_backend = AsyncMock()
                mock_backend._db = MagicMock()
                MockBackend.return_value = mock_backend

                mock_meta = AsyncMock()
                mock_meta.migrate_from_config = AsyncMock(return_value=0)
                MockMeta.return_value = mock_meta

                mock_scheduler = AsyncMock()
                mock_scheduler.start = AsyncMock()
                mock_scheduler.stop = AsyncMock()
                mock_scheduler.register_job = MagicMock()
                mock_scheduler.set_job_handler = MagicMock()
                MockScheduler.return_value = mock_scheduler

                await client.connect()

                assert client._scheduler is not None
                mock_scheduler.start.assert_awaited_once()

                await client.disconnect()

                mock_scheduler.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_stops_scheduler(self):
        """disconnect() calls scheduler.stop() and clears the reference."""
        from semantic_vector_router.models import (
            DatabaseConfig,
            EmbeddingConfig,
            EmbeddingMode,
            EmbeddingProvider,
            PartitioningConfig,
            SVRConfig,
            VectorSearchConfig,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=1536),
            embedding=EmbeddingConfig(
                mode=EmbeddingMode.BYOM,
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
                api_key_env="OPENAI_API_KEY",
                dimensions=1536,
            ),
        )

        with patch(
            "semantic_vector_router.client.validate_config",
            return_value=[],
        ):
            from semantic_vector_router.client import SVRClient

            client = SVRClient(config=config, auto_connect=False)
            client._connected = True
            client._backend = AsyncMock()
            client._metadata = None

            mock_scheduler = AsyncMock()
            mock_scheduler.stop = AsyncMock()
            client._scheduler = mock_scheduler

            await client.disconnect()

            mock_scheduler.stop.assert_awaited_once()
            assert client._scheduler is None

    @pytest.mark.asyncio
    async def test_disconnect_clears_event_bus(self):
        """disconnect() sets _event_bus to None."""
        from semantic_vector_router.models import (
            DatabaseConfig,
            EmbeddingConfig,
            EmbeddingMode,
            EmbeddingProvider,
            PartitioningConfig,
            SVRConfig,
            VectorSearchConfig,
        )

        config = SVRConfig(
            database=DatabaseConfig(
                connection_string_env="MONGODB_URI",
                database="test_db",
                source_collection="products",
            ),
            partitioning=PartitioningConfig(field="category"),
            vector_search=VectorSearchConfig(dimensions=1536),
            embedding=EmbeddingConfig(
                mode=EmbeddingMode.BYOM,
                provider=EmbeddingProvider.OPENAI,
                model="text-embedding-3-small",
                api_key_env="OPENAI_API_KEY",
                dimensions=1536,
            ),
        )

        with patch(
            "semantic_vector_router.client.validate_config",
            return_value=[],
        ):
            from semantic_vector_router.client import SVRClient

            client = SVRClient(config=config, auto_connect=False)
            client._connected = True
            client._backend = AsyncMock()
            client._metadata = None
            client._event_bus = MockEventBus()

            await client.disconnect()

            assert client._event_bus is None

    @pytest.mark.asyncio
    async def test_ingest_no_event_bus_no_error(self):
        """ingest() succeeds when no event bus is configured."""
        with patch(
            "semantic_vector_router.client.validate_config",
            return_value=[],
        ):
            from semantic_vector_router.models import (
                DatabaseConfig,
                EmbeddingConfig,
                EmbeddingMode,
                EmbeddingProvider,
                PartitioningConfig,
                SVRConfig,
                VectorSearchConfig,
            )

            config = SVRConfig(
                database=DatabaseConfig(
                    connection_string_env="MONGODB_URI",
                    database="test_db",
                    source_collection="products",
                ),
                partitioning=PartitioningConfig(field="category"),
                vector_search=VectorSearchConfig(dimensions=1536),
                embedding=EmbeddingConfig(
                    mode=EmbeddingMode.BYOM,
                    provider=EmbeddingProvider.OPENAI,
                    model="text-embedding-3-small",
                    api_key_env="OPENAI_API_KEY",
                    dimensions=1536,
                ),
            )

            from semantic_vector_router.client import SVRClient

            client = SVRClient(config=config, auto_connect=False)
            client._connected = True
            client._event_bus = None  # No event bus
            client._backend = MagicMock()
            client._metadata = None

            mock_embedder = AsyncMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
            client._embedder = mock_embedder

            mock_result = MagicMock()
            mock_result.inserted = 1
            mock_result.failed = 0
            mock_result.elapsed_ms = 50.0

            with patch(
                "semantic_vector_router.client.IngestPipeline"
            ) as MockPipeline, patch.object(
                client, "_create_document_embedder", return_value=mock_embedder,
            ):
                mock_pipeline = AsyncMock()
                mock_pipeline.ingest = AsyncMock(return_value=mock_result)
                MockPipeline.return_value = mock_pipeline

                result = await client.ingest(
                    documents=[{"text": "hello"}],
                    partition="electronics",
                )

            assert result.inserted == 1
