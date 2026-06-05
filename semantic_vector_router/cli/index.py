"""Index management CLI commands."""

import asyncio
import time

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error

console = Console()


@click.group()
def index_group():
    """Manage vector search indexes."""
    pass


@index_group.command("status")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def index_status(config_path):
    """Show status of all vector search indexes."""

    async def _status():
        config, backend = await _get_backend(config_path)
        try:
            registry = config.partitions.registry

            if not registry:
                console.print("[yellow]No partitions registered.[/yellow]")
                return

            table = Table(title="Index Status", show_header=True, header_style="bold cyan")
            table.add_column("Partition")
            table.add_column("Collection")
            table.add_column("Index Name")
            table.add_column("Status")
            table.add_column("Queryable")

            for name, p in registry.items():
                collection = p.search_collection or p.view_name or config.database.source_collection
                idx_status = await backend.get_index_status(collection, p.index_name)

                status = idx_status.get("status", "unknown")
                queryable = idx_status.get("queryable", False)

                status_color = "green" if status == "READY" else ("yellow" if status == "PENDING" else "red")
                queryable_str = "[green]Yes[/green]" if queryable else "[red]No[/red]"

                table.add_row(
                    name,
                    collection,
                    p.index_name,
                    f"[{status_color}]{status}[/{status_color}]",
                    queryable_str,
                )

            console.print(table)
        finally:
            await backend.disconnect()

    _run_async(_status())


@index_group.command("rebuild")
@click.argument("partition")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def index_rebuild(partition, yes, config_path):
    """Rebuild the vector search index for a partition (delete + recreate)."""

    async def _rebuild():
        config, backend = await _get_backend(config_path)
        try:
            p = config.partitions.registry.get(partition)
            if not p:
                console.print(f"[red]Partition '{partition}' not found.[/red]")
                return

            if not yes:
                if not click.confirm(f"Rebuild index '{p.index_name}' for partition '{partition}'?"):
                    console.print("[yellow]Cancelled.[/yellow]")
                    return

            collection = p.search_collection or p.view_name or config.database.source_collection

            console.print(f"Deleting index '{p.index_name}'...")
            try:
                await backend.delete_vector_search_index(collection, p.index_name)
            except Exception as e:
                console.print(f"[yellow]Delete warning: {e}[/yellow]")

            console.print(f"Recreating index '{p.index_name}'...")
            embedding_field = p.embedding_field or config.vector_search.embedding_field
            await backend.create_vector_search_index(
                collection_name=collection,
                index_name=p.index_name,
                embedding_field=embedding_field,
                dimensions=config.vector_search.dimensions,
                similarity=config.vector_search.similarity.value,
                quantization=config.vector_storage.index_quantization,
            )

            console.print(f"[green]Index '{p.index_name}' rebuild initiated.[/green]")
            console.print("[dim]Run 'svr index wait <partition>' to wait for queryable state.[/dim]")
        finally:
            await backend.disconnect()

    _run_async(_rebuild())


@index_group.command("wait")
@click.argument("partition")
@click.option("--timeout", default=300, help="Timeout in seconds (default 300)")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def index_wait(partition, timeout, config_path):
    """Wait for a partition's index to become queryable."""

    async def _wait():
        config, backend = await _get_backend(config_path)
        try:
            p = config.partitions.registry.get(partition)
            if not p:
                console.print(f"[red]Partition '{partition}' not found.[/red]")
                return

            collection = p.search_collection or p.view_name or config.database.source_collection
            start = time.time()

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task(f"Waiting for index '{p.index_name}' to be queryable...", total=None)

                while time.time() - start < timeout:
                    idx_status = await backend.get_index_status(collection, p.index_name)
                    if idx_status.get("queryable", False):
                        progress.update(task, description=f"[green]Index '{p.index_name}' is queryable![/green]")
                        console.print(f"\n[green]Index is queryable (took {time.time() - start:.1f}s).[/green]")
                        return
                    await asyncio.sleep(5)

                progress.update(task, description="[red]Timeout waiting for index.[/red]")
                console.print(f"\n[red]Timeout after {timeout}s. Index may still be building.[/red]")
        finally:
            await backend.disconnect()

    _run_async(_wait())
