"""Interactive setup wizard for Semantic Vector Router."""

import asyncio
import os
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from semantic_vector_router.backends.factory import create_backend
from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.backends.postgres.config import (
    PgDistanceMetric,
    PgIndexType,
    PostgresBackendConfig,
)
from semantic_vector_router.config import save_config, validate_config
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.lifecycle.scanner import PartitionScanner
from semantic_vector_router.models import (
    BackendType,
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    IndexLocation,
    PartitioningConfig,
    RerankerProvider,
    SVRConfig,
    VectorSearchConfig,
    VectorStorageConfig,
)

console = Console()


@click.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(),
    default=None,
    help="Path to save config file (default: .svr/config.json)",
)
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Run in non-interactive mode with defaults",
)
@click.option(
    "--backend",
    type=click.Choice(["mongodb", "postgres"]),
    default="mongodb",
    help="Database backend to use",
)
@click.option(
    "--connection-string-env", default=None,
    help="Env var for connection string (default: MONGODB_URI or POSTGRES_URI)",
)
@click.option("--database", default=None, help="Database name")
@click.option("--collection", default=None, help="Source collection name")
@click.option("--partition-field", default=None, help="Field to partition by")
@click.option(
    "--embedding-provider", default=None,
    type=click.Choice(["openai", "voyage", "cohere", "huggingface"]),
    help="Embedding provider",
)
@click.option("--embedding-model", default=None, help="Embedding model name")
@click.option("--dimensions", type=int, default=None, help="Embedding dimensions")
@click.option(
    "--index-location", default=None,
    type=click.Choice(["source", "views", "fields"]),
    help="Index location mode (MongoDB only)",
)
@click.option("--provision", is_flag=True, help="Auto-provision discovered partitions")
@click.option("--schema", default="public", help="PostgreSQL schema (postgres only)")
@click.option("--table-prefix", default="svr_", help="Table prefix (postgres only)")
@click.option(
    "--index-type", default="hnsw",
    type=click.Choice(["hnsw", "ivfflat"]),
    help="Vector index type (postgres only)",
)
@click.option(
    "--distance-metric", default="cosine",
    type=click.Choice(["cosine", "l2", "ip"]),
    help="Distance metric (postgres only)",
)
def init_command(
    config_path: Optional[str],
    non_interactive: bool,
    backend: str,
    connection_string_env: Optional[str],
    database: Optional[str],
    collection: Optional[str],
    partition_field: Optional[str],
    embedding_provider: Optional[str],
    embedding_model: Optional[str],
    dimensions: Optional[int],
    index_location: Optional[str],
    provision: bool,
    schema: str,
    table_prefix: str,
    index_type: str,
    distance_metric: str,
) -> None:
    """Initialize Semantic Vector Router with interactive setup."""
    # Resolve default connection_string_env based on backend
    if connection_string_env is None:
        connection_string_env = "POSTGRES_URI" if backend == "postgres" else "MONGODB_URI"

    console.print(
        Panel.fit(
            "[bold blue]Semantic Vector Router[/bold blue] - Setup Wizard",
            border_style="blue",
        )
    )

    if non_interactive:
        # Validate required fields
        missing = []
        if not database:
            missing.append("--database")
        if not collection:
            missing.append("--collection")
        if not partition_field:
            missing.append("--partition-field")
        if missing:
            msg = f"Missing required options for non-interactive mode: {', '.join(missing)}"
            console.print(f"[red]{msg}[/red]")
            raise click.Abort()

        try:
            assert database is not None
            assert collection is not None
            assert partition_field is not None
            asyncio.run(_non_interactive_setup(
                config_path=config_path,
                backend=backend,
                connection_string_env=connection_string_env,
                database=database,
                collection=collection,
                partition_field=partition_field,
                embedding_provider=embedding_provider or "openai",
                embedding_model=embedding_model,
                dimensions=dimensions or 1536,
                index_location=index_location or "views",
                provision=provision,
                schema=schema,
                table_prefix=table_prefix,
                index_type=index_type,
                distance_metric=distance_metric,
            ))
        except Exception as e:
            console.print(f"\n[red]Error during setup: {e}[/red]")
            raise click.Abort()
        return

    try:
        asyncio.run(_interactive_setup(config_path))
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during setup: {e}[/red]")
        raise click.Abort()


async def _non_interactive_setup(
    config_path: Optional[str],
    backend: str,
    connection_string_env: str,
    database: str,
    collection: str,
    partition_field: str,
    embedding_provider: str,
    embedding_model: Optional[str],
    dimensions: int,
    index_location: str,
    provision: bool,
    schema: str = "public",
    table_prefix: str = "svr_",
    index_type: str = "hnsw",
    distance_metric: str = "cosine",
) -> None:
    """Run non-interactive setup with provided options."""
    provider_map = {
        "openai": EmbeddingProvider.OPENAI,
        "voyage": EmbeddingProvider.VOYAGE,
        "cohere": EmbeddingProvider.COHERE,
        "huggingface": EmbeddingProvider.HUGGINGFACE,
    }
    model_defaults = {
        "openai": "text-embedding-3-small",
        "voyage": "voyage-4-lite",
        "cohere": "embed-english-v3.0",
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    }
    api_key_defaults = {
        "openai": "OPENAI_API_KEY",
        "voyage": "VOYAGE_API_KEY",
        "cohere": "COHERE_API_KEY",
    }
    location_map = {
        "source": IndexLocation.SOURCE,
        "views": IndexLocation.VIEWS,
        "fields": IndexLocation.FIELDS,
    }

    is_postgres = backend == "postgres"

    config = SVRConfig(
        database=DatabaseConfig(
            backend=BackendType.POSTGRES if is_postgres else BackendType.MONGODB,
            connection_string_env=connection_string_env,
            database=database,
            source_collection=collection,
        ),
        partitioning=PartitioningConfig(field=partition_field),
        vector_search=VectorSearchConfig(dimensions=dimensions),
        vector_storage=VectorStorageConfig(
            index_on=IndexLocation.SOURCE if is_postgres else location_map[index_location],
        ),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=provider_map[embedding_provider],
            model=embedding_model or model_defaults[embedding_provider],
            api_key_env=api_key_defaults.get(embedding_provider),
            dimensions=dimensions,
        ),
    )

    # Attach PostgresBackendConfig when using postgres backend
    if is_postgres:
        index_type_map = {
            "hnsw": PgIndexType.HNSW,
            "ivfflat": PgIndexType.IVFFLAT,
        }
        distance_metric_map = {
            "cosine": PgDistanceMetric.COSINE,
            "l2": PgDistanceMetric.L2,
            "ip": PgDistanceMetric.INNER_PRODUCT,
        }
        config.postgres = PostgresBackendConfig(
            connection_string_env=connection_string_env,
            schema=schema,
            table_prefix=table_prefix,
            index_type=index_type_map[index_type],
            distance_metric=distance_metric_map[distance_metric],
            vector_dimensions=dimensions,
        )

    # Validate
    warnings = validate_config(config)
    for w in warnings:
        console.print(f"[yellow]Warning: {w}[/yellow]")

    # Try to connect and discover partitions
    backend_instance = None
    connection_string = os.environ.get(connection_string_env)
    if connection_string:
        try:
            if is_postgres:
                backend_instance = create_backend(config)
            else:
                backend_instance = MongoDBBackend(config)
            await backend_instance.connect()
            backend_label = "PostgreSQL" if is_postgres else "MongoDB"
            console.print(f"[green]Connected to {backend_label}.[/green]")

            if not is_postgres:
                scanner = PartitionScanner(backend_instance, config)
                partition_counts = await scanner.scan_partition_values(limit=50)
                console.print(f"Found {len(partition_counts)} partition values.")

                if provision and partition_counts:
                    provisioner = PartitionProvisioner(backend_instance, config)
                    keys = list(partition_counts.keys())
                    created = await provisioner.create_partitions_batch(keys)
                    console.print(f"[green]Provisioned {len(created)} partition(s).[/green]")
            else:
                console.print("[dim]Partition provisioning not yet supported for PostgreSQL.[/dim]")

        except Exception as e:
            console.print(f"[yellow]Connection/provisioning warning: {e}[/yellow]")
        finally:
            if backend_instance:
                await backend_instance.disconnect()

    # Save config
    saved_path = save_config(config, config_path)
    console.print(f"[green]Configuration saved to {saved_path}[/green]")


async def _interactive_setup(config_path: Optional[str]) -> None:
    """Run the interactive setup wizard."""
    config = SVRConfig(
        database=DatabaseConfig(
            database="",
            source_collection="",
        ),
        partitioning=PartitioningConfig(field=""),
    )

    # Step 0: Backend Selection
    console.print("\n[bold]Step 0: Database Backend[/bold]")
    console.print("Which database backend?")
    console.print("  1. [bold]MongoDB Atlas[/bold]")
    console.print("  2. [bold]PostgreSQL (pgvector)[/bold]")

    backend_choice = Prompt.ask("Select backend", choices=["1", "2"], default="1")
    is_postgres = backend_choice == "2"

    if is_postgres:
        config.database.backend = BackendType.POSTGRES

    # Step 1: Database Configuration
    console.print("\n[bold]Step 1: Database Configuration[/bold]")

    if is_postgres:
        connection_env = Prompt.ask(
            "PostgreSQL connection string env var",
            default="POSTGRES_URI",
        )
    else:
        connection_env = Prompt.ask(
            "MongoDB connection string (or env var name)",
            default="MONGODB_URI",
        )
    config.database.connection_string_env = connection_env

    # Check if we can connect
    connection_string = os.environ.get(connection_env)
    if not connection_string:
        console.print(
            f"[yellow]Note: Environment variable {connection_env} not set. "
            "Set it before using the SDK.[/yellow]"
        )

    database_name = Prompt.ask("Database name")
    config.database.database = database_name

    collection_name = Prompt.ask("Source collection" if not is_postgres else "Source table name")
    config.database.source_collection = collection_name

    # Postgres-specific configuration
    if is_postgres:
        console.print("\n[bold]PostgreSQL Settings[/bold]")

        pg_schema = Prompt.ask("Schema", default="public")
        pg_table_prefix = Prompt.ask("Table prefix", default="svr_")

        console.print("\nVector index type:")
        console.print("  1. [bold]HNSW[/bold] - Approximate nearest neighbor (recommended)")
        console.print("  2. [bold]IVFFlat[/bold] - Inverted file index")
        pg_idx_choice = Prompt.ask("Select index type", choices=["1", "2"], default="1")
        pg_index_type = PgIndexType.HNSW if pg_idx_choice == "1" else PgIndexType.IVFFLAT

        console.print("\nDistance metric:")
        console.print("  1. [bold]cosine[/bold] - Cosine similarity (recommended)")
        console.print("  2. [bold]L2[/bold] - Euclidean distance")
        console.print("  3. [bold]inner product[/bold] - Dot product")
        pg_dist_choice = Prompt.ask("Select distance metric", choices=["1", "2", "3"], default="1")
        pg_dist_map = {
            "1": PgDistanceMetric.COSINE,
            "2": PgDistanceMetric.L2,
            "3": PgDistanceMetric.INNER_PRODUCT,
        }
        pg_distance_metric = pg_dist_map[pg_dist_choice]

        config.postgres = PostgresBackendConfig(
            connection_string_env=connection_env,
            schema=pg_schema,
            table_prefix=pg_table_prefix,
            index_type=pg_index_type,
            distance_metric=pg_distance_metric,
        )

    # Try to connect and discover fields
    fields: list[str] = []
    partition_counts: dict[str, int] = {}
    backend = None

    if connection_string:
        backend_label = "PostgreSQL" if is_postgres else "MongoDB"
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Connecting to {backend_label}...", total=None)

            try:
                if is_postgres:
                    backend = create_backend(config)
                else:
                    backend = MongoDBBackend(config)
                await backend.connect()
                progress.update(task, description="Connected! Discovering fields...")

                if hasattr(backend, "get_field_names"):
                    fields = await backend.get_field_names()
                else:
                    fields = []
                progress.update(task, description="Fields discovered!")

            except Exception as e:
                console.print(f"[yellow]Could not connect: {e}[/yellow]")
                backend = None

    # Step 2: Partitioning Configuration
    console.print("\n[bold]Step 2: Partitioning Configuration[/bold]")

    if fields:
        console.print("Available fields: " + ", ".join(fields[:20]))
        if len(fields) > 20:
            console.print(f"  ... and {len(fields) - 20} more")

    partition_field = Prompt.ask("Field to partition by")
    config.partitioning.field = partition_field

    # Scan for partition values if connected
    if connection_string and backend:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning partition values...", total=None)

            try:
                scanner = PartitionScanner(backend, config)
                partition_counts = await scanner.scan_partition_values(limit=20)

                console.print(
                    f"\n[green]Found {len(partition_counts)} unique values "
                    f"for '{partition_field}':[/green]"
                )

                table = Table(show_header=True, header_style="bold")
                table.add_column("Value")
                table.add_column("Documents", justify="right")

                for value, count in list(partition_counts.items())[:12]:
                    table.add_row(str(value), f"{count:,}")

                if len(partition_counts) > 12:
                    table.add_row(
                        f"... and {len(partition_counts) - 12} more",
                        "",
                    )

                console.print(table)

            except Exception as e:
                console.print(f"[yellow]Could not scan: {e}[/yellow]")

    # Step 3: Vector Search Configuration
    console.print("\n[bold]Step 3: Vector Search Configuration[/bold]")

    embedding_field = Prompt.ask("Embedding field name", default="embedding")
    config.vector_search.embedding_field = embedding_field

    dimensions = int(Prompt.ask("Embedding dimensions", default="1536"))
    config.vector_search.dimensions = dimensions
    config.embedding.dimensions = dimensions

    # Update postgres vector_dimensions if applicable
    if is_postgres and config.postgres is not None:
        config.postgres.vector_dimensions = dimensions

    # Index location choice (MongoDB only)
    if not is_postgres:
        console.print("\n[bold]Index Location[/bold]")
        console.print("Where should vector search indexes be created?")
        console.print("  1. [bold]VIEWS[/bold] - One index per partition view")
        console.print("     [dim]Best for: Fewer partitions, complete isolation between partitions[/dim]")
        console.print("  2. [bold]SOURCE[/bold] - Single index on source collection with pre-filtering")
        console.print("     [dim]Best for: Many partitions, simpler index management[/dim]")

        index_choice = Prompt.ask("Select index location", choices=["1", "2"], default="1")

        if index_choice == "1":
            config.vector_storage.index_on = IndexLocation.VIEWS
            console.print("[dim]Each partition will have its own vector search index.[/dim]")
        else:
            config.vector_storage.index_on = IndexLocation.SOURCE
            console.print(
                f"[dim]A single index will be created on '{collection_name}' with "
                f"'{partition_field}' as a filter field.[/dim]"
            )
    else:
        config.vector_storage.index_on = IndexLocation.SOURCE
        console.print("[dim]PostgreSQL uses table-based vector indexes.[/dim]")

    # Step 4: Embedding Configuration
    console.print("\n[bold]Step 4: Embedding Configuration[/bold]")

    if is_postgres:
        # PostgreSQL only supports BYOM — no auto-embedding mode
        console.print("[dim]PostgreSQL uses BYOM (Bring Your Own Model) embedding mode.[/dim]")
        config.embedding.mode = EmbeddingMode.BYOM
        mode_choice = "1"
    else:
        console.print("Embedding mode:")
        console.print("  1. BYOM (Bring Your Own Model) - SDK calls embedding API")
        console.print("  2. Auto-embedding (MongoDB Atlas) - MongoDB embeds automatically")

        mode_choice = Prompt.ask("Select mode", choices=["1", "2"], default="1")

    if mode_choice == "1":
        config.embedding.mode = EmbeddingMode.BYOM

        console.print("\nEmbedding provider:")
        console.print("  1. OpenAI")
        console.print("  2. Voyage AI")
        console.print("  3. Cohere")
        console.print("  4. HuggingFace (local)")

        provider_choice = Prompt.ask(
            "Select provider", choices=["1", "2", "3", "4"], default="1"
        )

        if provider_choice == "1":
            config.embedding.provider = EmbeddingProvider.OPENAI
            config.embedding.model = Prompt.ask(
                "Model", default="text-embedding-3-small"
            )
            config.embedding.api_key_env = Prompt.ask(
                "API key env var", default="OPENAI_API_KEY"
            )
        elif provider_choice == "2":
            config.embedding.provider = EmbeddingProvider.VOYAGE
            config.embedding.api_key_env = Prompt.ask(
                "API key env var", default="VOYAGE_API_KEY"
            )

            # Voyage model selection
            console.print("\n[bold]Voyage Model Selection[/bold]")
            console.print("Voyage 4 models feature a [cyan]shared embedding space[/cyan]:")
            console.print("  1. [bold]voyage-4-large[/bold] - Best accuracy (recommended for documents)")
            console.print("  2. [bold]voyage-4[/bold] - Balanced accuracy/speed")
            console.print("  3. [bold]voyage-4-lite[/bold] - Fastest (recommended for queries)")
            console.print("  4. voyage-3-large - Legacy model")

            voyage_choice = Prompt.ask(
                "Select model for queries", choices=["1", "2", "3", "4"], default="3"
            )

            voyage_models = {
                "1": "voyage-4-large",
                "2": "voyage-4",
                "3": "voyage-4-lite",
                "4": "voyage-3-large",
            }
            config.embedding.model = voyage_models[voyage_choice]

            # Asymmetric embeddings for Voyage 4
            if voyage_choice in ["1", "2", "3"]:
                console.print("\n[bold]Asymmetric Embeddings[/bold]")
                console.print(
                    "[dim]Voyage 4 models share an embedding space. You can use a faster model\n"
                    "for queries and a more accurate model for document indexing.[/dim]"
                )

                use_asymmetric = Confirm.ask(
                    "Use asymmetric embeddings (different model for documents)?",
                    default=voyage_choice == "3",  # Default yes if using lite for queries
                )

                if use_asymmetric:
                    console.print("Document model:")
                    console.print("  1. [bold]voyage-4-large[/bold] - Best accuracy (recommended)")
                    console.print("  2. voyage-4 - Balanced")
                    console.print("  3. voyage-4-lite - Same as query model")

                    doc_choice = Prompt.ask(
                        "Select model for documents", choices=["1", "2", "3"], default="1"
                    )
                    doc_models = {
                        "1": "voyage-4-large",
                        "2": "voyage-4",
                        "3": "voyage-4-lite",
                    }
                    config.embedding.document_model = doc_models[doc_choice]

                    console.print(
                        f"[green]Asymmetric config: queries={config.embedding.model}, "
                        f"documents={config.embedding.document_model}[/green]"
                    )

                # Voyage 4 dimension options
                console.print("\n[bold]Embedding Dimensions[/bold]")
                console.print("Voyage 4 supports flexible dimensions via Matryoshka learning:")
                console.print("  1. 1024 (default, best accuracy)")
                console.print("  2. 512 (balanced)")
                console.print("  3. 256 (smallest, fastest)")
                console.print("  4. 2048 (highest, if needed)")

                dim_choice = Prompt.ask(
                    "Select dimensions", choices=["1", "2", "3", "4"], default="1"
                )
                dim_map = {"1": 1024, "2": 512, "3": 256, "4": 2048}
                config.embedding.voyage_output_dimension = dim_map[dim_choice]
                config.embedding.dimensions = dim_map[dim_choice]
                config.vector_search.dimensions = dim_map[dim_choice]
        elif provider_choice == "3":
            config.embedding.provider = EmbeddingProvider.COHERE
            config.embedding.model = Prompt.ask(
                "Model", default="embed-english-v3.0"
            )
            config.embedding.api_key_env = Prompt.ask(
                "API key env var", default="COHERE_API_KEY"
            )
        else:
            config.embedding.provider = EmbeddingProvider.HUGGINGFACE
            config.embedding.model = Prompt.ask(
                "Model", default="sentence-transformers/all-MiniLM-L6-v2"
            )
            config.embedding.local = True
            config.embedding.device = Prompt.ask("Device", default="cpu")

    else:
        config.embedding.mode = EmbeddingMode.AUTO
        config.embedding.provider = EmbeddingProvider.ATLAS_VOYAGE
        config.embedding.model = Prompt.ask("Model", default="voyage-3-large")
        console.print(
            "[yellow]Note: Auto-embedding is in Public Preview. "
            "Ensure it's enabled for your Atlas cluster.[/yellow]"
        )

    # Step 5: Reranking Configuration
    console.print("\n[bold]Step 5: Reranking Configuration[/bold]")

    enable_reranking = Confirm.ask(
        "Enable reranking for multi-partition queries?", default=True
    )
    config.reranking.enabled = enable_reranking

    if enable_reranking:
        console.print("Reranking provider:")
        console.print("  1. Voyage AI")
        console.print("  2. Cohere")

        rerank_choice = Prompt.ask("Select provider", choices=["1", "2"], default="1")

        if rerank_choice == "1":
            config.reranking.provider = RerankerProvider.VOYAGE
            config.reranking.model = Prompt.ask("Model", default="rerank-2")
            config.reranking.api_key_env = Prompt.ask(
                "API key env var", default="VOYAGE_API_KEY"
            )
        else:
            config.reranking.provider = RerankerProvider.COHERE
            config.reranking.model = Prompt.ask(
                "Model", default="rerank-english-v3.0"
            )
            config.reranking.api_key_env = Prompt.ask(
                "API key env var", default="COHERE_API_KEY"
            )

        config.reranking.top_k_per_partition = int(
            Prompt.ask("Candidates per partition", default="20")
        )
        config.reranking.final_top_k = int(
            Prompt.ask("Final results to return", default="10")
        )

    # Step 6: Lifecycle Configuration
    console.print("\n[bold]Step 6: Lifecycle Configuration[/bold]")

    auto_provision = Confirm.ask(
        "Auto-provision partitions for new values?", default=True
    )
    config.lifecycle.auto_provision = auto_provision

    if auto_provision:
        confirm_required = Confirm.ask(
            "Require confirmation before creating?", default=False
        )
        config.lifecycle.confirmation_required = confirm_required

    # Validate configuration
    console.print("\n[bold]Validating configuration...[/bold]")
    try:
        warnings = validate_config(config)
        for warning in warnings:
            console.print(f"[yellow]Warning: {warning}[/yellow]")
        console.print("[green]Configuration valid![/green]")
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()

    # Save configuration
    console.print("\n[bold]Saving configuration...[/bold]")
    saved_path = save_config(config, config_path)
    console.print(f"[green]Created {saved_path}[/green]")

    # Provision partitions if connected (MongoDB only — not yet supported for postgres)
    if connection_string and backend and partition_counts and not is_postgres:
        provision = Confirm.ask(
            f"\nProvision {len(partition_counts)} partitions now?", default=True
        )

        if provision:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                provisioner = PartitionProvisioner(backend, config)

                for i, (value, count) in enumerate(partition_counts.items()):
                    task = progress.add_task(
                        f"Creating partition: {value}...", total=None
                    )
                    try:
                        await provisioner.create_partition(
                            name=str(value),
                            filter_value=value,
                            skip_if_exists=True,
                        )
                        progress.update(
                            task, description=f"[green]Created: {value}[/green]"
                        )
                    except Exception as e:
                        progress.update(
                            task, description=f"[red]Failed: {value} - {e}[/red]"
                        )
                    progress.remove_task(task)

            # Save updated config with partitions
            save_config(config, config_path)
            console.print(
                f"[green]Provisioned {len(partition_counts)} partitions[/green]"
            )
    elif is_postgres and connection_string and backend:
        console.print("[dim]Partition provisioning not yet supported for PostgreSQL.[/dim]")

    # Cleanup
    if backend:
        await backend.disconnect()

    # Print quick start
    console.print("\n" + "=" * 50)
    console.print("[bold green]Setup complete![/bold green]")
    console.print("\n[bold]Quick start:[/bold]")
    console.print(
        """
[dim]from semantic_vector_router import SVRClient

async def main():
    svr = SVRClient()
    await svr.connect()

    results = await svr.search(
        query="your search query",
        partitions=["partition_name"],
        limit=10
    )

    for hit in results.hits:
        print(f"{hit.score:.3f} - {hit.document}")
[/dim]
"""
    )


if __name__ == "__main__":
    init_command()
