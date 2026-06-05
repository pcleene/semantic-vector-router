"""Metadata store for partition and operation state management.

Manages an svr_metadata collection in MongoDB with three document types:
- partition: Partition state (replaces config file registry)
- operation: Repartition operations (pending/in-progress/done)
- lock: Distributed locks for multi-worker detection
"""

import os
from datetime import datetime, timedelta
from typing import Any, Optional

from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from semantic_vector_router.exceptions import MetadataError
from semantic_vector_router.models import (
    PartitionInfo,
    SVRConfig,
)
from semantic_vector_router.utils.logging import get_logger
from semantic_vector_router.utils.retry import with_retry

logger = get_logger(__name__)


# Retryable MongoDB exceptions (lazy loaded)
def _retryable_mongo():
    try:
        from pymongo.errors import (
            AutoReconnect,
            NetworkTimeout,
            ServerSelectionTimeoutError,
        )
        return (AutoReconnect, NetworkTimeout, ServerSelectionTimeoutError)
    except ImportError:
        return (Exception,)


class MetadataStore:
    """Manages partition and operation metadata in MongoDB.

    Uses an svr_metadata collection with three document types:
    - partition documents (partition:<name>)
    - operation documents (op:<type>-<partition>-<timestamp>)
    - lock documents (lock:<id>)

    Can either share the main backend's database or use a separate connection.
    """

    def __init__(self, config: SVRConfig):
        self.config = config
        self._client: Optional[AsyncMongoClient] = None
        self._db: Optional[AsyncDatabase[Any]] = None
        self._collection: Optional[AsyncCollection[Any]] = None
        self._use_separate_client = bool(
            config.lifecycle.metadata.connection_string_env
        )

    @property
    def _coll(self) -> AsyncCollection[Any]:
        """Return collection, asserting it is connected."""
        assert self._collection is not None, "MetadataStore not connected"
        return self._collection

    async def connect(self) -> None:
        """Connect to metadata MongoDB. Create collection reference."""
        if self._use_separate_client:
            conn_str_env = self.config.lifecycle.metadata.connection_string_env
            assert conn_str_env is not None, "connection_string_env must be set"
            conn_str = os.getenv(conn_str_env)
            if not conn_str:
                raise MetadataError(
                    f"Metadata connection string env var not set: {conn_str_env}"
                )
            self._client = AsyncMongoClient(
                conn_str,
                connectTimeoutMS=self.config.resilience.connection_timeout_ms,
                serverSelectionTimeoutMS=self.config.resilience.server_selection_timeout_ms,
            )
            db_name = (
                self.config.lifecycle.metadata.database
                or self.config.database.database
            )
            self._db = self._client[db_name]

        if self._db is None:
            raise MetadataError(
                "Metadata store not connected. Call _set_shared_db() or configure "
                "lifecycle.metadata.connection_string_env."
            )

        collection_name = self.config.lifecycle.metadata.collection
        self._collection = self._db[collection_name]
        logger.info(f"Metadata store connected: {collection_name}")

    async def disconnect(self) -> None:
        """Close metadata connection if using separate client."""
        if self._use_separate_client and self._client:
            await self._client.close()
            self._client = None
        self._db = None
        self._collection = None

    def _set_shared_db(self, db) -> None:
        """Set shared database reference from main backend.

        Called by SVRClient when no separate metadata connection is configured.
        """
        if not self._use_separate_client:
            self._db = db

    # --- Partition CRUD ---

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def get_partition(self, name: str) -> Optional[PartitionInfo]:
        """Get partition by name."""
        doc = await self._coll.find_one(
            {"_id": f"partition:{name}", "type": "partition"}
        )
        if not doc:
            return None
        return self._doc_to_partition(doc)

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def list_partitions(
        self, status: Optional[str] = None
    ) -> list[PartitionInfo]:
        """List all partitions, optionally filtered by status."""
        query: dict[str, Any] = {"type": "partition"}
        if status:
            query["status"] = status
        cursor = self._coll.find(query)
        docs = await cursor.to_list(length=None)
        return [self._doc_to_partition(doc) for doc in docs]

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def save_partition(self, partition: PartitionInfo) -> None:
        """Save or update partition (upsert)."""
        doc = self._partition_to_doc(partition)
        await self._coll.replace_one(
            {"_id": f"partition:{partition.name}"},
            doc,
            upsert=True,
        )

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def delete_partition(self, name: str) -> bool:
        """Delete partition. Returns True if deleted."""
        result = await self._coll.delete_one(
            {"_id": f"partition:{name}", "type": "partition"}
        )
        return result.deleted_count > 0

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def update_partition_count(self, name: str, count: int) -> None:
        """Update document count and last_count_update timestamp."""
        await self._coll.update_one(
            {"_id": f"partition:{name}", "type": "partition"},
            {
                "$set": {
                    "document_count": count,
                    "last_count_update": datetime.utcnow(),
                }
            },
        )

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def append_health_history(self, name: str, count: int) -> None:
        """Append to health_history, capped at 30 data points."""
        await self._coll.update_one(
            {"_id": f"partition:{name}", "type": "partition"},
            {
                "$push": {
                    "health_history": {
                        "$each": [{"ts": datetime.utcnow(), "count": count}],
                        "$slice": -30,
                    }
                }
            },
        )

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def get_health_history(self, name: str) -> list[dict[str, Any]]:
        """Get health_history array for a partition."""
        doc = await self._coll.find_one(
            {"_id": f"partition:{name}", "type": "partition"},
            {"health_history": 1},
        )
        if not doc:
            return []
        return doc.get("health_history", [])

    # --- Operation CRUD ---

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def create_operation(self, operation: dict[str, Any]) -> str:
        """Create operation document. Returns operation ID."""
        op_id = operation["_id"]
        operation.setdefault("type", "operation")
        await self._coll.insert_one(operation)
        return op_id

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def get_operation(self, op_id: str) -> Optional[dict[str, Any]]:
        """Get operation by ID."""
        return await self._coll.find_one(
            {"_id": op_id, "type": "operation"}
        )

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def list_operations(
        self, status: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """List operations, optionally filtered by status."""
        query: dict[str, Any] = {"type": "operation"}
        if status:
            query["status"] = status
        cursor = self._coll.find(query).sort("created_at", -1)
        return await cursor.to_list(length=None)

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def update_operation_status(
        self, op_id: str, status: str, **kwargs: Any
    ) -> None:
        """Update operation status and optional fields."""
        update: dict[str, Any] = {"$set": {"status": status, **kwargs}}
        await self._coll.update_one(
            {"_id": op_id, "type": "operation"}, update
        )

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def update_operation_step(
        self,
        op_id: str,
        step_action: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update a specific step within an operation."""
        update_fields: dict[str, Any] = {
            "steps.$[elem].status": status,
        }
        if status == "in_progress":
            update_fields["steps.$[elem].started_at"] = datetime.utcnow()
        elif status in ("done", "failed"):
            update_fields["steps.$[elem].completed_at"] = datetime.utcnow()
        if error:
            update_fields["steps.$[elem].error"] = error

        await self._coll.update_one(
            {"_id": op_id, "type": "operation"},
            {"$set": update_fields},
            array_filters=[{"elem.action": step_action}],
        )

    # --- Distributed Lock ---

    async def acquire_lock(
        self, lock_id: str, holder: str, ttl_seconds: int = 300
    ) -> bool:
        """Attempt to acquire a distributed lock.

        Uses find_one_and_update with upsert to atomically create or
        re-acquire an expired lock. Returns True if acquired.
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds)

        try:
            result = await self._coll.find_one_and_update(
                {
                    "_id": f"lock:{lock_id}",
                    "$or": [
                        {"expires_at": {"$lt": now}},
                        {"holder": holder},
                    ],
                },
                {
                    "$set": {
                        "type": "lock",
                        "holder": holder,
                        "acquired_at": now,
                        "expires_at": expires_at,
                    }
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            return result is not None and result.get("holder") == holder
        except DuplicateKeyError:
            # Another worker grabbed it via upsert race
            return False

    async def release_lock(self, lock_id: str, holder: str) -> bool:
        """Release lock only if holder matches. Returns True if released."""
        result = await self._coll.delete_one(
            {"_id": f"lock:{lock_id}", "type": "lock", "holder": holder}
        )
        return result.deleted_count > 0

    async def is_lock_held(self, lock_id: str) -> bool:
        """Check if lock is currently held (not expired)."""
        doc = await self._coll.find_one(
            {"_id": f"lock:{lock_id}", "type": "lock"}
        )
        if not doc:
            return False
        return doc.get("expires_at", datetime.min) > datetime.utcnow()

    # --- Migration ---

    async def migrate_from_config(self, config: SVRConfig) -> int:
        """One-time migration: copy partitions from config to metadata.

        Idempotent — skips partitions that already exist in metadata.
        Returns number of partitions migrated.
        """
        migrated = 0
        for name, partition in config.partitions.registry.items():
            existing = await self.get_partition(name)
            if existing is None:
                await self.save_partition(partition)
                migrated += 1
                logger.info(f"Migrated partition to metadata: {name}")
        return migrated

    # --- Centroid Management ---

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def update_centroid(
        self, name: str, centroid: list[float], updated_at: Optional[datetime] = None
    ) -> None:
        """Update partition centroid vector and timestamp.

        Args:
            name: Partition name.
            centroid: Normalized centroid embedding vector.
            updated_at: Timestamp of computation. Defaults to now.
        """
        if updated_at is None:
            updated_at = datetime.utcnow()
        await self._coll.update_one(
            {"_id": f"partition:{name}", "type": "partition"},
            {
                "$set": {
                    "centroid": centroid,
                    "centroid_updated_at": updated_at,
                }
            },
        )

    @with_retry(max_attempts=3, retryable_exceptions=_retryable_mongo())
    async def get_centroids(self) -> dict[str, list[float]]:
        """Get all partition centroids as a name -> vector mapping.

        Only returns partitions that have a non-null centroid.

        Returns:
            Dict of partition_name -> centroid_vector.
        """
        cursor = self._coll.find(
            {"type": "partition", "centroid": {"$ne": None}},
            {"name": 1, "centroid": 1},
        )
        docs = await cursor.to_list(length=None)
        return {doc["name"]: doc["centroid"] for doc in docs}

    # --- Conversion Helpers ---

    def _partition_to_doc(self, partition: PartitionInfo) -> dict[str, Any]:
        """Convert PartitionInfo to metadata document."""
        doc = partition.model_dump(mode="json", exclude_none=True)
        doc["_id"] = f"partition:{partition.name}"
        doc["type"] = "partition"
        doc.setdefault("health_history", [])
        return doc

    def _doc_to_partition(self, doc: dict[str, Any]) -> PartitionInfo:
        """Convert metadata document to PartitionInfo."""
        data = dict(doc)
        data.pop("_id", None)
        data.pop("type", None)
        data.pop("health_history", None)
        return PartitionInfo.model_validate(data)
