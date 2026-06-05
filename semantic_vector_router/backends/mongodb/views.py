"""MongoDB view pipeline building and management."""

import re
from typing import Any, Optional

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import OperationFailure, PyMongoError

from semantic_vector_router.exceptions import SVRException, ViewCreationError
from semantic_vector_router.models import (
    EmbeddingMode,
    SVRConfig,
    VectorStorageMode,
)
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class MongoDBViewOps:
    """View pipeline building and lifecycle management."""

    def __init__(self, config: SVRConfig):
        self.config = config
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

    def _build_partition_view_pipeline(
        self,
        partition_name: str,
        filter_value: Any,
        filter_expression: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Build the aggregation pipeline for a partition view."""
        pipeline: list[dict[str, Any]] = []

        if filter_expression:
            match_stage = {"$match": filter_expression}
        else:
            partition_field = self.config.partitioning.field
            match_stage = {"$match": {partition_field: filter_value}}

        pipeline.append(match_stage)

        if self.config.embedding.source_fields:
            embedding_expr = self._build_embedding_field_expression()
            pipeline.append({
                "$addFields": {
                    self.config.embedding.computed_field: embedding_expr
                }
            })

        if self.config.vector_storage.mode == VectorStorageMode.SEPARATE:
            embeddings_collection = self.config.vector_storage.embeddings_collection
            reference_field = self.config.vector_storage.reference_field

            pipeline.extend([
                {
                    "$lookup": {
                        "from": embeddings_collection,
                        "localField": "_id",
                        "foreignField": reference_field,
                        "as": "_svr_embedding_data"
                    }
                },
                {
                    "$unwind": {
                        "path": "$_svr_embedding_data",
                        "preserveNullAndEmptyArrays": False
                    }
                },
                {
                    "$addFields": {
                        self.config.vector_search.embedding_field: (
                            f"$_svr_embedding_data.{self.config.vector_search.embedding_field}"
                        )
                    }
                },
                {
                    "$project": {
                        "_svr_embedding_data": 0
                    }
                }
            ])

        return pipeline

    def _build_embedding_field_expression(self) -> Any:
        """Build embedding text expression for a MongoDB view's $addFields stage.

        Behavior depends on embedding mode and template configuration:

        - **Template set** (any mode): ``$concat`` using template placeholders
          (unchanged legacy behavior).
        - **BYOM, no template**: Object projection preserving field types.
          Arrays stay arrays, nested objects stay nested. The embedder
          layer will call ``serialize_for_embedding()`` at embed time.
        - **AUTO, no template**: Field-labeled ``$concat`` producing a single
          string that Atlas can embed directly.

        Returns:
            A MongoDB expression — either a dict (object projection for BYOM)
            or a ``$concat`` expression (string for AUTO / template).
        """
        source_fields = self.config.embedding.source_fields
        if not source_fields:
            return {"$literal": ""}

        # Template mode: unchanged behavior for both BYOM and AUTO
        if self.config.embedding.template:
            return self._build_template_expression(
                self.config.embedding.template,
                source_fields,
            )

        if self.config.embedding.mode == EmbeddingMode.AUTO:
            # AUTO mode: field-labeled $concat producing a string
            # Atlas auto-embedding needs a string, not an object
            return self._build_labeled_concat_expression(source_fields)
        else:
            # BYOM mode: structured object projection
            # Preserves arrays, nested objects, etc. natively
            return self._build_object_projection(source_fields)

    def _build_object_projection(
        self, source_fields: list[str]
    ) -> dict[str, Any]:
        """Build an object projection that preserves field types.

        Each source field is projected with ``$ifNull`` to provide
        safe defaults: empty string for scalars, empty array for
        fields that might be arrays (handled at serialization time).
        """
        projection: dict[str, Any] = {}
        for field in source_fields:
            projection[field] = {"$ifNull": [f"${field}", None]}
        return projection

    def _build_labeled_concat_expression(
        self, source_fields: list[str]
    ) -> dict[str, Any]:
        """Build a field-labeled $concat for Atlas AUTO embedding mode.

        Produces: ``"title: <value>\\ndescription: <value>\\ntags: <value>"``
        """
        concat_parts: list[Any] = []
        for i, field in enumerate(source_fields):
            if i > 0:
                concat_parts.append("\n")
            concat_parts.append(f"{field}: ")
            concat_parts.append(
                {"$toString": {"$ifNull": [f"${field}", ""]}}
            )
        return {"$concat": concat_parts}

    # Keep legacy name as alias for backward compatibility
    def _build_concat_expression(self) -> Any:
        """Legacy alias — delegates to _build_embedding_field_expression()."""
        return self._build_embedding_field_expression()

    def _build_template_expression(
        self, template: str, fields: list[str]
    ) -> dict[str, Any]:
        """Build $concat expression from template."""
        parts: list[Any] = []
        last_end = 0

        for match in re.finditer(r"\{(\w+)\}", template):
            if match.start() > last_end:
                parts.append(template[last_end:match.start()])
            field_name = match.group(1)
            parts.append({"$ifNull": [f"${field_name}", ""]})
            last_end = match.end()

        if last_end < len(template):
            parts.append(template[last_end:])

        return {"$concat": parts}

    async def create_partition_view(
        self,
        partition_name: str,
        filter_value: Any,
        filter_expression: Optional[dict[str, Any]] = None,
        retry_func: Optional[Any] = None,
    ) -> str:
        """Create a view for a partition."""
        view_name = f"{self.config.partitioning.view_prefix}{partition_name}"
        source_collection = self.config.database.source_collection

        pipeline = self._build_partition_view_pipeline(
            partition_name, filter_value, filter_expression
        )

        async def _do_create_view() -> str:
            existing = await self.db.list_collection_names(
                filter={"name": view_name, "type": "view"}
            )
            if view_name in existing:
                logger.info(f"View {view_name} already exists, dropping and recreating")
                await self.db.drop_collection(view_name)

            await self.db.command({
                "create": view_name,
                "viewOn": source_collection,
                "pipeline": pipeline
            })

            logger.info(f"Created partition view: {view_name}")
            return view_name

        try:
            if retry_func:
                return await retry_func(_do_create_view)
            return await _do_create_view()
        except OperationFailure as e:
            raise ViewCreationError(
                f"Failed to create view {view_name}: {e}",
                details={"partition": partition_name, "error_code": e.code}
            )

    async def delete_partition_view(self, view_name: str) -> None:
        """Delete a partition view."""
        try:
            await self.db.drop_collection(view_name)
            logger.info(f"Deleted partition view: {view_name}")
        except PyMongoError as e:
            raise SVRException(f"Failed to delete view {view_name}: {e}")

    async def view_exists(self, view_name: str) -> bool:
        """Check if a view exists."""
        try:
            collections = await self.db.list_collection_names(
                filter={"name": view_name}
            )
            return view_name in collections
        except PyMongoError:
            return False

    async def list_views(self) -> list[str]:
        """List all views in the database."""
        try:
            collections = await self.db.list_collection_names(
                filter={"type": "view"}
            )
            return collections
        except PyMongoError as e:
            raise SVRException(f"Failed to list views: {e}")

    async def list_partition_views(self) -> list[str]:
        """List all SVR partition views."""
        all_views = await self.list_views()
        prefix = self.config.partitioning.view_prefix
        return [v for v in all_views if v.startswith(prefix)]
