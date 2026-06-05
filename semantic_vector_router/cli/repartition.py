"""Repartition CLI commands."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from semantic_vector_router.backends.metadata import MetadataStore
from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error
from semantic_vector_router.lifecycle.repartition import RepartitionEngine

console = Console()


@click.group()
def repartition_group():
    """Manage repartition operations."""
    pass


@repartition_group.command("pending")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def repartition_pending(config_path):
    """List pending repartition operations."""

    async def _pending():
        config, backend = await _get_backend(config_path)
        metadata = None
        try:
            metadata = MetadataStore(config)
            if not config.lifecycle.metadata.connection_string_env:
                metadata._set_shared_db(backend._db)
            await metadata.connect()

            ops = await metadata.list_operations(status="pending")

            if not ops:
                console.print("[green]No pending operations.[/green]")
                return

            table = Table(
                title="Pending Operations",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("ID")
            table.add_column("Action")
            table.add_column("Partition")
            table.add_column("Signal")
            table.add_column("Created")

            for op in ops:
                table.add_row(
                    op.get("_id", "?"),
                    op.get("action", "?"),
                    op.get("target_partition", op.get("partition", "?")),
                    op.get("signal", "?"),
                    str(op.get("created_at", "?")),
                )

            console.print(table)
            console.print(f"\n[dim]{len(ops)} pending operation(s)[/dim]")
        finally:
            if metadata:
                await metadata.disconnect()
            await backend.disconnect()

    _run_async(_pending())


@repartition_group.command("execute")
@click.argument("op_id")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def repartition_execute(op_id, config_path):
    """Execute a repartition operation."""

    async def _execute():
        config, backend = await _get_backend(config_path)
        metadata = None
        try:
            metadata = MetadataStore(config)
            if not config.lifecycle.metadata.connection_string_env:
                metadata._set_shared_db(backend._db)
            await metadata.connect()

            engine = RepartitionEngine(backend, metadata, config)

            console.print(f"Executing operation [bold]{op_id}[/bold]...")
            success = await engine.execute_operation(op_id)

            if success:
                console.print("[green]Operation completed successfully.[/green]")
            else:
                console.print("[red]Operation failed. Check 'svr repartition status' for details.[/red]")
        finally:
            if metadata:
                await metadata.disconnect()
            await backend.disconnect()

    _run_async(_execute())


@repartition_group.command("status")
@click.argument("op_id")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def repartition_status(op_id, config_path):
    """Show status of a repartition operation."""

    async def _status():
        config, backend = await _get_backend(config_path)
        metadata = None
        try:
            metadata = MetadataStore(config)
            if not config.lifecycle.metadata.connection_string_env:
                metadata._set_shared_db(backend._db)
            await metadata.connect()

            op = await metadata.get_operation(op_id)
            if not op:
                console.print(f"[red]Operation '{op_id}' not found.[/red]")
                return

            status_color = {
                "pending": "yellow",
                "done": "green",
                "failed": "red",
            }.get(op.get("status", ""), "white")

            info = (
                f"[bold]ID:[/bold] {op.get('_id')}\n"
                f"[bold]Status:[/bold] [{status_color}]{op.get('status')}[/{status_color}]\n"
                f"[bold]Action:[/bold] {op.get('action')}\n"
                f"[bold]Partition:[/bold] {op.get('target_partition', op.get('partition'))}\n"
                f"[bold]Created:[/bold] {op.get('created_at')}"
            )
            if op.get("error"):
                info += f"\n[bold]Error:[/bold] [red]{op.get('error')}[/red]"

            console.print(Panel(info, title="Operation Details", border_style="cyan"))

            # Show steps
            steps = op.get("steps", [])
            if steps:
                table = Table(
                    title="Steps",
                    show_header=True,
                    header_style="bold cyan",
                )
                table.add_column("#", justify="right", width=4)
                table.add_column("Action")
                table.add_column("Status")
                table.add_column("Started")
                table.add_column("Completed")

                for step in steps:
                    step_status = step.get("status", "pending")
                    step_color = {
                        "pending": "dim",
                        "in_progress": "yellow",
                        "done": "green",
                        "failed": "red",
                    }.get(step_status, "white")

                    table.add_row(
                        str(step.get("step", "?")),
                        step.get("action", "?"),
                        f"[{step_color}]{step_status}[/{step_color}]",
                        str(step.get("started_at") or "-"),
                        str(step.get("completed_at") or "-"),
                    )

                console.print(table)
        finally:
            if metadata:
                await metadata.disconnect()
            await backend.disconnect()

    _run_async(_status())


@repartition_group.command("rollback")
@click.argument("op_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def repartition_rollback(op_id, yes, config_path):
    """Rollback a failed repartition operation."""
    if not yes:
        if not click.confirm(f"Rollback operation '{op_id}'? This will delete child partitions."):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    async def _rollback():
        config, backend = await _get_backend(config_path)
        metadata = None
        try:
            metadata = MetadataStore(config)
            if not config.lifecycle.metadata.connection_string_env:
                metadata._set_shared_db(backend._db)
            await metadata.connect()

            engine = RepartitionEngine(backend, metadata, config)
            await engine.rollback_operation(op_id)

            console.print(f"[green]Operation '{op_id}' rolled back successfully.[/green]")
        finally:
            if metadata:
                await metadata.disconnect()
            await backend.disconnect()

    _run_async(_rollback())
