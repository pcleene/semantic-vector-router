"""MongoDB Atlas Search index management."""

from typing import Any, Optional

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import OperationFailure, PyMongoError

from semantic_vector_router.exceptions import IndexCreationError, SVRException
from semantic_vector_router.models import MongoDBIndexQuantization
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class MongoDBIndexOps:
    """Index creation, deletion, and status management."""

    def __init__(self) -> None:
        self._db: Optional[AsyncDatabase] = None

    def set_db(self, db: AsyncDatabase) -> None:
        """Set the database reference after connection."""
        self._db = db

    @property
    def db(self) -> AsyncDatabase:
        if self._db is None:
            from semantic_vector_router.exceptions import ConnectionError
            raise ConnectionError("Not connected to MongoDB.")
        return self._db

    async def create_vector_search_index(
        self,
        collection_name: str,
        index_name: str,
        embedding_field: str,
        dimensions: int,
        similarity: str,
        filter_fields: Optional[list[str]] = None,
        quantization: Optional[MongoDBIndexQuantization] = None,
        retry_func: Optional[Any] = None,
    ) -> None:
        """Create a vector search index."""
        vector_field: dict[str, Any] = {
            "type": "vector",
            "path": embedding_field,
            "numDimensions": dimensions,
            "similarity": similarity,
        }

        if quantization and quantization != MongoDBIndexQuantization.NONE:
            vector_field["quantization"] = quantization.value

        fields = [vector_field]

        if filter_fields:
            for field in filter_fields:
                fields.append({
                    "type": "filter",
                    "path": field
                })

        index_definition = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": fields
            }
        }

        async def _do_create_index() -> None:
            collection = self.db[collection_name]
            await collection.create_search_index(index_definition)
            logger.info(
                f"Created vector search index {index_name} on {collection_name}"
            )

        try:
            if retry_func:
                await retry_func(_do_create_index)
            else:
                await _do_create_index()
        except OperationFailure as e:
            if "already exists" in str(e).lower():
                logger.info(f"Index {index_name} already exists on {collection_name}")
            else:
                raise IndexCreationError(
                    f"Failed to create index {index_name}: {e}",
                    details={
                        "collection": collection_name,
                        "error_code": getattr(e, "code", None)
                    }
                )

    async def delete_vector_search_index(
        self, collection_name: str, index_name: str
    ) -> None:
        """Delete a vector search index."""
        try:
            collection = self.db[collection_name]
            await collection.drop_search_index(index_name)
            logger.info(f"Deleted index {index_name} from {collection_name}")
        except PyMongoError as e:
            raise SVRException(f"Failed to delete index {index_name}: {e}")

    async def index_exists(self, collection_name: str, index_name: str) -> bool:
        """Check if an index exists."""
        try:
            collection = self.db[collection_name]
            cursor = await collection.list_search_indexes()
            indexes = await cursor.to_list()
            return any(idx.get("name") == index_name for idx in indexes)
        except PyMongoError:
            return False

    async def get_index_status(
        self, collection_name: str, index_name: str
    ) -> dict[str, Any]:
        """Get the status of an index."""
        try:
            collection = self.db[collection_name]
            cursor = await collection.list_search_indexes()
            indexes = await cursor.to_list()

            for idx in indexes:
                if idx.get("name") == index_name:
                    return {
                        "name": idx.get("name"),
                        "status": idx.get("status", "unknown"),
                        "type": idx.get("type"),
                        "queryable": idx.get("queryable", False),
                    }

            return {"name": index_name, "status": "not_found"}
        except PyMongoError as e:
            return {"name": index_name, "status": "error", "error": str(e)}
