"""Partition resolution for query routing."""

import time
from typing import TYPE_CHECKING, Any, Optional, Union

from semantic_vector_router.exceptions import PartitionNotFoundError
from semantic_vector_router.models import (
    PartitionInfo,
    PartitionStatus,
    SVRConfig,
)
from semantic_vector_router.routing.centroid import CentroidRouter
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.metrics import MetricsCollector, MetricType

if TYPE_CHECKING:
    from semantic_vector_router.backends.metadata import MetadataStore

logger = get_logger(__name__)


class PartitionResolver:
    """Resolves partition specifications to actual partition objects.

    Handles:
    - Explicit partition lists: ["electronics", "furniture"]
    - "all" keyword to search all partitions
    - Default partition behavior
    - Split partition expansion (parent -> children)

    Supports dual-source mode:
    - If a MetadataStore is provided, reads from the metadata collection
    - Otherwise falls back to config file registry
    """

    def __init__(
        self,
        config: SVRConfig,
        metadata: Optional["MetadataStore"] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize the resolver.

        Args:
            config: SVR configuration with partition registry.
            metadata: Optional metadata store for partition state.
            metrics: Optional metrics collector for observability.
        """
        self.config = config
        self._metadata = metadata
        self._metrics = metrics
        self._filter_map: dict[str, list[str]] = {}
        self._filter_map_version: int = 0
        self._registry_version: int = 0
        self._centroid_router: Optional[CentroidRouter] = None

    @property
    def registry(self) -> dict[str, PartitionInfo]:
        """Get the partition registry from config (sync fallback)."""
        return self.config.partitions.registry

    async def _get_registry(self) -> dict[str, PartitionInfo]:
        """Get partition registry from metadata store or config file."""
        if self._metadata:
            partitions = await self._metadata.list_partitions()
            return {p.name: p for p in partitions}
        return self.config.partitions.registry

    async def resolve(
        self,
        partitions: Optional[Union[list[str], str]] = None,
        filters: Optional[dict[str, Any]] = None,
        query_embedding: Optional[list[float]] = None,
        include_disabled: bool = False,
    ) -> list[PartitionInfo]:
        """Resolve partition specification to partition list.

        Resolution cascParts Distributor (first match wins):
        1. Explicit partition names → resolve directly
        2. Filter-map routing → match filters against partition field
        3. Centroid routing → walk tree using query embedding similarity
        4. Fallback → fan-out to all partitions

        Args:
            partitions: Partition specification:
                - None: Use default behavior from config
                - "all": Return all active partitions
                - List of names: Return specified partitions
            filters: Optional query filters dict. If a key matches the
                configured partition field, filter-map routing is used
                to resolve to matching leaf partitions.
            query_embedding: Optional query embedding vector for centroid
                routing. When provided and centroid routing is enabled,
                the resolver walks the partition tree to select the most
                relevant partitions.
            include_disabled: Whether to include disabled partitions.

        Returns:
            List of resolved PartitionInfo objects.

        Raises:
            PartitionNotFoundError: If a specified partition doesn't exist.
        """
        # Handle None - use default from config
        if partitions is None:
            partitions = self.config.routing.default_partitions

        # Handle explicit list
        if isinstance(partitions, list):
            return await self._resolve_explicit(partitions, include_disabled)

        # Handle single partition name as string (not "all")
        if isinstance(partitions, str) and partitions != "all":
            return await self._resolve_explicit([partitions], include_disabled)

        # --- Filter-map routing (step 2) ---
        if partitions == "all" and filters:
            filter_result = await self._try_filter_map(filters, include_disabled)
            if filter_result is not None:
                return filter_result

        # --- Centroid routing (step 3) ---
        if (
            partitions == "all"
            and query_embedding is not None
            and self.config.routing.centroid_routing.enabled
        ):
            centroid_result = await self._try_centroid_routing(
                query_embedding, include_disabled
            )
            if centroid_result is not None:
                return centroid_result

        # Handle "all" keyword (fallback)
        if partitions == "all":
            return await self._get_all_partitions(include_disabled)

        raise ValueError(f"Invalid partitions specification: {partitions}")

    async def _get_all_partitions(
        self, include_disabled: bool = False
    ) -> list[PartitionInfo]:
        """Get all active partitions.

        Args:
            include_disabled: Whether to include disabled partitions.

        Returns:
            List of all active partitions.
        """
        registry = await self._get_registry()
        partitions = []
        for partition in registry.values():
            if not include_disabled and partition.status == PartitionStatus.DISABLED:
                continue

            # Skip retired partitions (replaced by children)
            if partition.status == PartitionStatus.RETIRED:
                children = await self._get_child_partitions(
                    partition.name, registry
                )
                partitions.extend(children)
                continue

            # Skip split parent partitions, include their children instead
            if partition.status == PartitionStatus.SPLIT:
                children = await self._get_child_partitions(
                    partition.name, registry
                )
                partitions.extend(children)
            else:
                partitions.append(partition)

        # Respect max partitions limit
        max_partitions = self.config.routing.max_partitions_per_query
        if len(partitions) > max_partitions:
            logger.warning(
                f"Limiting query to {max_partitions} partitions "
                f"(total available: {len(partitions)})"
            )
            partitions = partitions[:max_partitions]

        return partitions

    async def _resolve_explicit(
        self,
        partition_names: list[str],
        include_disabled: bool = False,
    ) -> list[PartitionInfo]:
        """Resolve explicit partition names.

        Args:
            partition_names: List of partition names.
            include_disabled: Whether to include disabled partitions.

        Returns:
            List of resolved partitions.

        Raises:
            PartitionNotFoundError: If any partition doesn't exist.
        """
        registry = await self._get_registry()
        resolved = []

        for name in partition_names:
            partition = registry.get(name)

            if partition is None:
                raise PartitionNotFoundError(
                    f"Partition '{name}' not found",
                    details={"available": list(registry.keys())},
                )

            if not include_disabled and partition.status == PartitionStatus.DISABLED:
                logger.warning(f"Skipping disabled partition: {name}")
                continue

            # Handle split/retired partitions - expand to children
            if partition.status in (
                PartitionStatus.SPLIT,
                PartitionStatus.RETIRED,
            ):
                children = await self._get_child_partitions(name, registry)
                if children:
                    resolved.extend(children)
                    logger.debug(
                        f"Expanded split partition {name} to "
                        f"{len(children)} child partitions"
                    )
                else:
                    logger.warning(
                        f"Split partition {name} has no children, skipping"
                    )
            else:
                resolved.append(partition)

        # Respect max partitions limit
        max_partitions = self.config.routing.max_partitions_per_query
        if len(resolved) > max_partitions:
            logger.warning(
                f"Limiting query to {max_partitions} partitions "
                f"(requested: {len(resolved)})"
            )
            resolved = resolved[:max_partitions]

        return resolved

    async def _get_child_partitions(
        self,
        parent_name: str,
        registry: Optional[dict[str, PartitionInfo]] = None,
    ) -> list[PartitionInfo]:
        """Get child partitions of a split parent.

        Args:
            parent_name: Name of the parent partition.
            registry: Pre-fetched registry (avoids re-fetching).

        Returns:
            List of child partitions.
        """
        if registry is None:
            registry = await self._get_registry()

        parent = registry.get(parent_name)
        if parent is None:
            return []

        children = []
        for child_name in parent.child_partitions:
            child = registry.get(child_name)
            if child and child.status != PartitionStatus.DISABLED:
                children.append(child)

        return children

    async def get_partition(self, name: str) -> PartitionInfo:
        """Get a single partition by name.

        Args:
            name: Partition name.

        Returns:
            PartitionInfo for the partition.

        Raises:
            PartitionNotFoundError: If partition doesn't exist.
        """
        registry = await self._get_registry()
        partition = registry.get(name)
        if partition is None:
            raise PartitionNotFoundError(
                f"Partition '{name}' not found",
                details={"available": list(registry.keys())},
            )
        return partition

    async def list_partitions(
        self,
        status: Optional[PartitionStatus] = None,
    ) -> list[PartitionInfo]:
        """List partitions, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            List of partitions.
        """
        registry = await self._get_registry()
        partitions = list(registry.values())
        if status is not None:
            partitions = [p for p in partitions if p.status == status]
        return partitions

    async def partition_exists(self, name: str) -> bool:
        """Check if a partition exists.

        Args:
            name: Partition name.

        Returns:
            True if partition exists.
        """
        registry = await self._get_registry()
        return name in registry

    # --- Filter-map routing ---

    async def _try_filter_map(
        self,
        filters: dict[str, Any],
        include_disabled: bool = False,
    ) -> Optional[list[PartitionInfo]]:
        """Try to resolve partitions via filter-map lookup.

        If one of the filter keys matches the configured partition field,
        look up the filter value in the filter map to find matching
        leaf partitions. This is O(1) dict lookup.

        Args:
            filters: Query filters dict.
            include_disabled: Whether to include disabled partitions.

        Returns:
            List of resolved partitions if filter matched, None otherwise.
        """
        partition_field = self.config.partitioning.field
        if partition_field not in filters:
            return None

        filter_value = filters[partition_field]

        # Build/refresh filter map
        registry = await self._get_registry()
        filter_map = self._build_filter_map(registry)

        # Handle both single value and list of values
        if isinstance(filter_value, list):
            all_names: list[str] = []
            for v in filter_value:
                names = filter_map.get(str(v), [])
                all_names.extend(names)
        else:
            all_names = filter_map.get(str(filter_value), [])

        if not all_names:
            return None  # No match — fall through to next cascParts Distributor step

        # Resolve the matched partition names
        return await self._resolve_explicit(
            list(dict.fromkeys(all_names)),  # dedupe preserving order
            include_disabled,
        )

    def _build_filter_map(
        self, registry: dict[str, PartitionInfo]
    ) -> dict[str, list[str]]:
        """Build filter value -> leaf partition names mapping.

        Split-aware: when a partition is SPLIT, the filter map maps
        its filter value directly to its leaf descendants.

        The map is rebuilt when the registry changes (version check).

        Args:
            registry: Current partition registry.

        Returns:
            Dict mapping filter value strings to lists of leaf partition names.
        """
        # Simple version tracking: rebuild if registry size changed
        current_version = len(registry)
        if self._filter_map and self._registry_version == current_version:
            return self._filter_map

        filter_map: dict[str, list[str]] = {}

        for name, p in registry.items():
            if p.status == PartitionStatus.DISABLED:
                continue

            filter_val = str(p.filter_value) if p.filter_value is not None else name

            if p.status in (PartitionStatus.SPLIT, PartitionStatus.RETIRED):
                # Map to leaf descendants
                leaves = self._collect_leaves(p, registry)
                if leaves:
                    leaf_names = [leaf.name for leaf in leaves]
                    if filter_val in filter_map:
                        filter_map[filter_val].extend(leaf_names)
                    else:
                        filter_map[filter_val] = leaf_names
            elif p.status == PartitionStatus.ACTIVE:
                if filter_val in filter_map:
                    filter_map[filter_val].append(name)
                else:
                    filter_map[filter_val] = [name]

        self._filter_map = filter_map
        self._registry_version = current_version

        return filter_map

    def _collect_leaves(
        self,
        partition: PartitionInfo,
        registry: dict[str, PartitionInfo],
    ) -> list[PartitionInfo]:
        """Recursively collect ACTIVE leaf descendants of a partition."""
        if partition.status == PartitionStatus.ACTIVE:
            return [partition]

        leaves: list[PartitionInfo] = []
        for child_name in partition.child_partitions:
            child = registry.get(child_name)
            if child and child.status != PartitionStatus.DISABLED:
                leaves.extend(self._collect_leaves(child, registry))
        return leaves

    # --- Centroid routing ---

    async def _try_centroid_routing(
        self,
        query_embedding: list[float],
        include_disabled: bool = False,
    ) -> Optional[list[PartitionInfo]]:
        """Try to resolve partitions via centroid-based tree walk.

        Walks the partition hierarchy top-down, scoring partition centroids
        against the query embedding. At each level, prunes branches below
        a dynamic threshold (max_score * relative_threshold).

        Args:
            query_embedding: The query's embedding vector.
            include_disabled: Whether to include disabled partitions.

        Returns:
            List of resolved partitions if centroid routing succeeded,
            None if routing could not determine partitions (fall through).
        """
        start = time.perf_counter()
        registry = await self._get_registry()

        if self._centroid_router is None:
            self._centroid_router = CentroidRouter(
                self.config.routing.centroid_routing
            )

        result = await self._centroid_router.route_by_centroid(
            query_embedding=query_embedding,
            registry=registry,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if result:
            if self._metrics:
                self._metrics.emit_timing(
                    MetricType.CENTROID_ROUTE_LATENCY, elapsed_ms
                )
                self._metrics.emit_count(
                    MetricType.CENTROID_ROUTE_PARTITIONS, len(result)
                )
            logger.debug(
                f"Centroid routing resolved to {len(result)} partitions "
                f"in {elapsed_ms:.2f}ms"
            )
            return result

        return None  # Fall through to fan-out

    def invalidate_caches(self) -> None:
        """Invalidate all routing caches.

        Call this when partitions are created, deleted, or split.
        Invalidates both the filter map and centroid router state.
        """
        self._filter_map = {}
        self._registry_version = 0
        self._centroid_router = None

    def invalidate_filter_map(self) -> None:
        """Invalidate the filter map cache.

        Call this when partitions are created, deleted, or split.
        """
        self._filter_map = {}
        self._registry_version = 0
