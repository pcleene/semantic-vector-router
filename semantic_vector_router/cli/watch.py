"""Watch CLI commands for change stream monitoring."""

import asyncio
import signal

import click
from rich.console import Console
from rich.panel import Panel

from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error
from semantic_vector_router.config import load_config
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.lifecycle.watcher import PartitionWatcher

console = Console()


@click.group()
def watch_group():
    """Monitor collection changes and manage pending partitions."""
    pass


@watch_group.command("start")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def watch_start(config_path):
    """Start the change stream watcher (foreground, Ctrl+C to stop)."""

    async def _watch():
        config, backend = await _get_backend(config_path)
        try:
            provisioner = PartitionProvisioner(backend, config)
            watcher = PartitionWatcher(backend, config, provisioner=provisioner)

            console.print("[bold blue]Starting change stream watcher...[/bold blue]")
            console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

            # Set up signal handling for clean shutdown
            stop_event = asyncio.Event()

            def _signal_handler():
                console.print("\n[yellow]Stopping watcher...[/yellow]")
                stop_event.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _signal_handler)

            await watcher.start()

            # Wait for stop signal
            await stop_event.wait()
            await watcher.stop()

            status = watcher.get_status()
            console.print(f"[green]Watcher stopped.[/green] Created {status.partitions_created} partition(s).")
        finally:
            await backend.disconnect()

    _run_async(_watch())


@watch_group.command("status")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def watch_status(config_path):
    """Show watcher status and pending partitions."""
    config = load_config(config_path=config_path)
    pending = config.lifecycle.pending_partitions

    console.print(Panel(
        f"[bold]Auto-provision:[/bold] {'Enabled' if config.lifecycle.auto_provision else 'Disabled'}\n"
        f"[bold]Confirmation required:[/bold] {'Yes' if config.lifecycle.confirmation_required else 'No'}\n"
        f"[bold]Pending partitions:[/bold] {len(pending)}",
        title="Watcher Configuration",
        border_style="blue",
    ))

    if pending:
        console.print("\n[bold]Pending partitions:[/bold]")
        for p in pending:
            console.print(f"  - {p}")
        console.print("\n[dim]Use 'svr watch confirm <name>' or 'svr watch confirm-all' to provision.[/dim]")


@watch_group.command("confirm")
@click.argument("name")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def watch_confirm(name, config_path):
    """Confirm and provision a pending partition."""

    async def _confirm():
        config, backend = await _get_backend(config_path)
        try:
            provisioner = PartitionProvisioner(backend, config)
            watcher = PartitionWatcher(backend, config, provisioner=provisioner)

            result = await watcher.confirm_partition(name)
            if result:
                console.print(f"[green]Confirmed and provisioned partition '{name}'.[/green]")
            else:
                console.print(f"[red]Partition '{name}' not found in pending list.[/red]")
        finally:
            await backend.disconnect()

    _run_async(_confirm())


@watch_group.command("reject")
@click.argument("name")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def watch_reject(name, config_path):
    """Reject a pending partition (don't provision)."""
    config = load_config(config_path=config_path)
    watcher = PartitionWatcher(None, config)  # type: ignore[arg-type]

    result = watcher.reject_partition(name)
    if result:
        console.print(f"[green]Rejected partition '{name}'.[/green]")
    else:
        console.print(f"[red]Partition '{name}' not found in pending list.[/red]")


@watch_group.command("confirm-all")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def watch_confirm_all(config_path):
    """Confirm and provision all pending partitions."""

    async def _confirm_all():
        config, backend = await _get_backend(config_path)
        try:
            provisioner = PartitionProvisioner(backend, config)
            watcher = PartitionWatcher(backend, config, provisioner=provisioner)

            count = await watcher.confirm_all_pending()
            console.print(f"[green]Confirmed and provisioned {count} partition(s).[/green]")
        finally:
            await backend.disconnect()

    _run_async(_confirm_all())
