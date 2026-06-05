"""Partition management CLI commands."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error
from semantic_vector_router.config import load_config
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.lifecycle.scanner import PartitionScanner

console = Console()


@click.group()
def partitions_group():
    """Manage partitions (list, create, delete, scan, provision)."""
    pass


@partitions_group.command("list")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def partitions_list(config_path):
    """List all registered partitions."""
    config = load_config(config_path=config_path)
    registry = config.partitions.registry

    if not registry:
        console.print("[yellow]No partitions registered.[/yellow]")
        console.print("[dim]Run 'svr partitions scan' to discover partition values.[/dim]")
        return

    table = Table(title="Partitions", show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Documents", justify="right")
    table.add_column("Index Location")
    table.add_column("Index Name")

    for name, p in registry.items():
        status_color = {
            "active": "green",
            "pending_split": "yellow",
            "split": "blue",
            "disabled": "red",
        }.get(p.status.value, "white")

        table.add_row(
            name,
            f"[{status_color}]{p.status.value}[/{status_color}]",
            f"{p.document_count:,}",
            p.index_location.value,
            p.index_name or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(registry)} partitions[/dim]")


@partitions_group.command("status")
@click.argument("name", required=False)
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def partitions_status(name, config_path):
    """Show partition health status (all or one)."""
    config = load_config(config_path=config_path)
    registry = config.partitions.registry

    if name:
        p = registry.get(name)
        if not p:
            console.print(f"[red]Partition '{name}' not found.[/red]")
            return

        panel_content = (
            f"[bold]Name:[/bold] {p.name}\n"
            f"[bold]Status:[/bold] {p.status.value}\n"
            f"[bold]Documents:[/bold] {p.document_count:,}\n"
            f"[bold]Index Location:[/bold] {p.index_location.value}\n"
            f"[bold]Index Name:[/bold] {p.index_name}\n"
            f"[bold]View Name:[/bold] {p.view_name or 'N/A'}\n"
            f"[bold]Search Collection:[/bold] {p.search_collection or p.view_name or 'N/A'}\n"
            f"[bold]Filter Value:[/bold] {p.filter_value}\n"
            f"[bold]Created:[/bold] {p.created_at}\n"
            f"[bold]Last Count Update:[/bold] {p.last_count_update or 'Never'}"
        )
        if p.embedding_field:
            panel_content += f"\n[bold]Embedding Field:[/bold] {p.embedding_field}"

        console.print(Panel(panel_content, title=f"Partition: {name}", border_style="cyan"))
    else:
        if not registry:
            console.print("[yellow]No partitions registered.[/yellow]")
            return

        table = Table(title="Partition Status", show_header=True, header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("Documents", justify="right")
        table.add_column("Last Updated")

        for pname, p in registry.items():
            status_color = {
                "active": "green",
                "pending_split": "yellow",
                "split": "blue",
                "disabled": "red",
            }.get(p.status.value, "white")

            table.add_row(
                pname,
                f"[{status_color}]{p.status.value}[/{status_color}]",
                f"{p.document_count:,}",
                str(p.last_count_update or "Never"),
            )

        console.print(table)


@partitions_group.command("create")
@click.argument("name")
@click.option("--filter-value", default=None, help="Filter value (defaults to name)")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def partitions_create(name, filter_value, config_path):
    """Create a new partition."""

    async def _create():
        config, backend = await _get_backend(config_path)
        try:
            provisioner = PartitionProvisioner(backend, config)
            partition = await provisioner.create_partition(
                name=name,
                filter_value=filter_value if filter_value else name,
            )
            console.print(f"[green]Created partition '{name}' with {partition.document_count:,} documents.[/green]")
        finally:
            await backend.disconnect()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(f"Creating partition '{name}'...", total=None)
        _run_async(_create())


@partitions_group.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def partitions_delete(name, yes, config_path):
    """Delete a partition (with confirmation)."""
    config = load_config(config_path=config_path)
    if name not in config.partitions.registry:
        console.print(f"[red]Partition '{name}' not found.[/red]")
        return

    if not yes:
        if not click.confirm(f"Delete partition '{name}'? This will remove the view and index."):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    async def _delete():
        config_inner, backend = await _get_backend(config_path)
        try:
            provisioner = PartitionProvisioner(backend, config_inner)
            await provisioner.delete_partition(name)
            console.print(f"[green]Deleted partition '{name}'.[/green]")
        finally:
            await backend.disconnect()

    _run_async(_delete())


@partitions_group.command("refresh")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def partitions_refresh(config_path):
    """Refresh document counts for all partitions."""

    async def _refresh():
        config, backend = await _get_backend(config_path)
        try:
            provisioner = PartitionProvisioner(backend, config)
            counts = await provisioner.update_all_partition_counts()

            table = Table(title="Updated Counts", show_header=True, header_style="bold cyan")
            table.add_column("Partition")
            table.add_column("Documents", justify="right")

            for name, count in counts.items():
                table.add_row(name, f"{count:,}")

            console.print(table)
        finally:
            await backend.disconnect()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Refreshing partition counts...", total=None)
        _run_async(_refresh())


@partitions_group.command("scan")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def partitions_scan(config_path):
    """Scan for new partition values in the collection."""

    async def _scan():
        config, backend = await _get_backend(config_path)
        try:
            scanner = PartitionScanner(backend, config)
            new_values = await scanner.get_new_partition_values()

            if not new_values:
                console.print("[green]No new partition values found.[/green]")
                return

            console.print(f"[yellow]Found {len(new_values)} new partition value(s):[/yellow]")
            for v in new_values:
                console.print(f"  - {v}")
            console.print("\n[dim]Run 'svr partitions provision' to create partitions for these values.[/dim]")
        finally:
            await backend.disconnect()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Scanning for new partition values...", total=None)
        _run_async(_scan())


@partitions_group.command("provision")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def partitions_provision(config_path):
    """Auto-provision partitions for discovered values."""

    async def _provision():
        config, backend = await _get_backend(config_path)
        try:
            scanner = PartitionScanner(backend, config)
            new_values = await scanner.get_new_partition_values()

            if not new_values:
                console.print("[green]No new partition values to provision.[/green]")
                return

            console.print(f"Provisioning {len(new_values)} new partition(s)...")
            provisioner = PartitionProvisioner(backend, config)
            created = await provisioner.create_partitions_batch(new_values)

            for name, p in created.items():
                console.print(f"  [green]Created:[/green] {name} ({p.document_count:,} docs)")

            console.print(f"\n[green]Provisioned {len(created)} partition(s).[/green]")
        finally:
            await backend.disconnect()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Provisioning partitions...", total=None)
        _run_async(_provision())


@partitions_group.command("compute-centroids")
@click.option("--partition", "-p", default=None, help="Compute for specific partition only")
@click.option("--sample-size", "-s", default=500, type=int, help="Documents to sample per partition")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def partitions_compute_centroids(partition, sample_size, config_path):
    """Compute centroid embeddings for partitions.

    Samples documents, reads their stored embedding vectors, computes
    the element-wise mean, and stores as the partition centroid.
    Zero API calls — reads already-stored embeddings.
    """
    from semantic_vector_router.backends.metadata import MetadataStore
    from semantic_vector_router.utils.vector_math import mean_vector, normalize

    async def _compute():
        config, backend = await _get_backend(config_path)
        metadata = MetadataStore(config)
        metadata._set_shared_db(backend._db)
        await metadata.connect()

        try:
            # Determine which partitions to compute
            partitions_list = await metadata.list_partitions()

            if partition:
                partitions_list = [p for p in partitions_list if p.name == partition]
                if not partitions_list:
                    console.print(f"[red]Partition '{partition}' not found.[/red]")
                    return

            # Filter to only ACTIVE partitions
            active = [p for p in partitions_list if p.status.value == "active"]
            if not active:
                console.print("[yellow]No active partitions found.[/yellow]")
                return

            console.print(f"Computing centroids for {len(active)} partition(s) (sample_size={sample_size})...")

            source_collection = config.database.source_collection
            collection = backend._db[source_collection]
            embedding_field = config.vector_search.embedding_field
            partition_field = config.partitioning.field
            index_on = config.vector_storage.index_on

            computed = 0
            skipped = 0

            for p in active:
                # Determine the embedding field for this partition
                if index_on.value == "fields" and p.embedding_field:
                    field_path = p.embedding_field
                else:
                    field_path = embedding_field

                # Build filter for this partition
                query_filter = {}
                if p.filter_value is not None:
                    query_filter[partition_field] = p.filter_value

                # Sample documents that have the embedding field
                pipeline = [
                    {"$match": {**query_filter, field_path: {"$exists": True, "$ne": None}}},
                    {"$sample": {"size": sample_size}},
                    {"$project": {field_path: 1}},
                ]
                cursor = await collection.aggregate(pipeline)
                docs = await cursor.to_list(length=None)

                if not docs:
                    console.print(f"  [yellow]Skipping {p.name}: no documents with embeddings[/yellow]")
                    skipped += 1
                    continue

                # Extract vectors
                vectors = []
                for doc in docs:
                    vec = doc.get(field_path)
                    if vec is not None and isinstance(vec, list):
                        vectors.append(vec)

                if not vectors:
                    console.print(f"  [yellow]Skipping {p.name}: no valid embedding vectors[/yellow]")
                    skipped += 1
                    continue

                # Compute mean and normalize
                centroid = normalize(mean_vector(vectors))
                await metadata.update_centroid(p.name, centroid)

                console.print(
                    f"  [green]Computed centroid for {p.name}[/green] "
                    f"({len(vectors)} vectors sampled, {len(centroid)} dimensions)"
                )
                computed += 1

            console.print(f"\n[green]Done:[/green] {computed} centroids computed, {skipped} skipped.")

        finally:
            await metadata.disconnect()
            await backend.disconnect()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Computing centroids...", total=None)
        _run_async(_compute())
