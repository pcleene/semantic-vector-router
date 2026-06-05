"""Partition value scanner for discovering existing partitions."""

from typing import Any, Optional, cast

from pymongo.errors import (
    ConnectionFailure,
    OperationFailure,
    ServerSelectionTimeoutError,
)

from semantic_vector_router.backends.base import BaseBackend
from semantic_vector_router.exceptions import ScanError
from semantic_vector_router.models import SVRConfig
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class PartitionScanner:
    """Scans collections to discover partition field values.

    Used during setup to find existing partition values and their
    document counts, and during runtime to detect new values.
    """

    def __init__(self, backend: BaseBackend, config: SVRConfig):
        """Initialize the scanner.

        Args:
            backend: Database backend.
            config: SVR configuration.
        """
        self.backend = backend
        self.config = config

    async def scan_partition_values(
        self,
        limit: Optional[int] = None,
    ) -> dict[str, int]:
        """Scan for all unique partition field values and their counts.

        Args:
            limit: Maximum number of distinct values to return.

        Returns:
            Dictionary mapping partition values to document counts,
            sorted by count descending.

        Raises:
            ScanError: If the backend call fails.
        """
        partition_field = self.config.partitioning.field

        logger.info(f"Scanning for partition values on field: {partition_field}")

        try:
            # Get partition counts from backend
            counts = await self.backend.get_partition_document_counts(partition_field)
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise ScanError(
                "Failed to scan partition values: database connection error",
                details={"field": partition_field, "error": str(e)},
            )
        except OperationFailure as e:
            raise ScanError(
                "Failed to scan partition values: operation failed",
                details={"field": partition_field, "error": str(e)},
            )

        # Sort by count descending
        sorted_counts = dict(
            sorted(counts.items(), key=lambda x: x[1], reverse=True)
        )

        # Apply limit if specified
        if limit and len(sorted_counts) > limit:
            sorted_counts = dict(list(sorted_counts.items())[:limit])

        logger.info(
            f"Found {len(sorted_counts)} unique values for '{partition_field}'"
        )

        return sorted_counts

    async def get_new_partition_values(self) -> list[Any]:
        """Find partition values that don't have corresponding partitions.

        Returns:
            List of partition values without existing partitions.

        Raises:
            ScanError: If the backend call fails.
        """
        partition_field = self.config.partitioning.field

        try:
            # Get all distinct values
            all_values = await self.backend.get_distinct_values(partition_field)
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise ScanError(
                "Failed to get distinct values: database connection error",
                details={"field": partition_field, "error": str(e)},
            )
        except OperationFailure as e:
            raise ScanError(
                "Failed to get distinct values: operation failed",
                details={"field": partition_field, "error": str(e)},
            )

        # Get known partitions
        known_partitions = set(self.config.partitions.registry.keys())

        # Find new values
        new_values = [v for v in all_values if str(v) not in known_partitions]

        if new_values:
            logger.info(f"Found {len(new_values)} new partition values")

        return new_values

    async def get_partition_stats(self) -> list[dict[str, Any]]:
        """Get detailed stats for all partition values.

        Returns:
            List of partition statistics including value, count, and registry status.

        Raises:
            ScanError: If the backend call fails.
        """
        partition_field = self.config.partitioning.field

        try:
            # Get counts
            counts = await self.backend.get_partition_document_counts(partition_field)
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise ScanError(
                "Failed to get partition stats: database connection error",
                details={"field": partition_field, "error": str(e)},
            )
        except OperationFailure as e:
            raise ScanError(
                "Failed to get partition stats: operation failed",
                details={"field": partition_field, "error": str(e)},
            )

        # Build stats list
        stats = []
        for value, count in counts.items():
            partition_name = str(value)
            partition_info = self.config.partitions.registry.get(partition_name)

            stat = {
                "value": value,
                "name": partition_name,
                "document_count": count,
                "has_partition": partition_info is not None,
                "view_name": partition_info.view_name if partition_info else None,
                "index_name": partition_info.index_name if partition_info else None,
                "status": partition_info.status.value if partition_info else "not_provisioned",
            }
            stats.append(stat)

        # Sort by document count descending
        stats.sort(key=lambda x: cast(int, x["document_count"]), reverse=True)

        return stats

    async def validate_partitions(self) -> dict[str, list[str]]:
        """Validate existing partitions against actual data.

        Returns:
            Dictionary with 'missing' (registered but no data),
            'orphaned' (data but not registered), and 'valid' partition names.

        Raises:
            ScanError: If the backend call fails.
        """
        partition_field = self.config.partitioning.field

        try:
            # Get all distinct values from data
            data_values = set(
                str(v) for v in await self.backend.get_distinct_values(partition_field)
            )
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise ScanError(
                "Failed to validate partitions: database connection error",
                details={"field": partition_field, "error": str(e)},
            )
        except OperationFailure as e:
            raise ScanError(
                "Failed to validate partitions: operation failed",
                details={"field": partition_field, "error": str(e)},
            )

        # Get registered partitions
        registered = set(self.config.partitions.registry.keys())

        # Calculate differences
        missing = registered - data_values  # Registered but no data
        orphaned = data_values - registered  # Data but not registered
        valid = registered & data_values  # Both registered and has data

        result = {
            "missing": list(missing),
            "orphaned": list(orphaned),
            "valid": list(valid),
        }

        if missing:
            logger.warning(f"Partitions with no data: {missing}")
        if orphaned:
            logger.info(f"Data values without partitions: {orphaned}")

        return result
