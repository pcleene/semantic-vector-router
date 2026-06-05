"""Monitor CLI commands for detection pipeline."""

import click
from rich.console import Console
from rich.table import Table

from semantic_vector_router.backends.metadata import MetadataStore
from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error
from semantic_vector_router.lifecycle.detector import PartitionDetector

console = Console()


@click.group()
def monitor_group():
    """Monitor partition health and run detection."""
    pass


@monitor_group.command("check")
@click.option("--auto-execute", is_flag=True, help="Execute auto-approved operations")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def monitor_check(auto_execute, config_path):
    """Run detection pipeline and display results."""

    async def _check():
        config, backend = await _get_backend(config_path)
        metadata = None
        try:
            metadata = MetadataStore(config)
            if not config.lifecycle.metadata.connection_string_env:
                metadata._set_shared_db(backend._db)
            await metadata.connect()

            detector = PartitionDetector(backend, metadata, config)
            results = await detector.run_detection()

            if not results:
                console.print("[green]No issues detected.[/green]")
                return

            table = Table(
                title="Detection Results",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Signal")
            table.add_column("Partition")
            table.add_column("Details")
            table.add_column("Auto", justify="center")
            table.add_column("Action")

            for r in results:
                signal_color = {
                    "threshold_breach": "red",
                    "approaching_threshold": "yellow",
                    "severe_skew": "magenta",
                    "underpopulated": "blue",
                    "stale": "dim",
                }.get(r.signal.value, "white")

                detail_str = ", ".join(
                    f"{k}={v}" for k, v in r.details.items()
                )
                if len(detail_str) > 60:
                    detail_str = detail_str[:60] + "..."

                table.add_row(
                    f"[{signal_color}]{r.signal.value}[/{signal_color}]",
                    r.partition,
                    detail_str,
                    "[green]Y[/green]" if r.auto_executable else "[dim]N[/dim]",
                    r.suggested_action,
                )

            console.print(table)
            console.print(f"\n[dim]Total: {len(results)} issue(s) detected[/dim]")

            if auto_execute:
                auto_results = [r for r in results if r.auto_executable]
                if auto_results:
                    console.print(
                        f"\n[yellow]Auto-executing {len(auto_results)} operation(s)...[/yellow]"
                    )
                else:
                    console.print("[dim]No auto-executable operations.[/dim]")
        finally:
            if metadata:
                await metadata.disconnect()
            await backend.disconnect()

    _run_async(_check())


@monitor_group.command("history")
@click.argument("partition")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def monitor_history(partition, config_path):
    """Show health history for a partition."""

    async def _history():
        config, backend = await _get_backend(config_path)
        metadata = None
        try:
            metadata = MetadataStore(config)
            if not config.lifecycle.metadata.connection_string_env:
                metadata._set_shared_db(backend._db)
            await metadata.connect()

            history = await metadata.get_health_history(partition)

            if not history:
                console.print(f"[yellow]No health history for partition '{partition}'.[/yellow]")
                return

            table = Table(
                title=f"Health History: {partition}",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Timestamp")
            table.add_column("Count", justify="right")

            for entry in history:
                table.add_row(
                    str(entry.get("ts", "?")),
                    f"{entry.get('count', 0):,}",
                )

            console.print(table)
            console.print(f"\n[dim]{len(history)} data point(s)[/dim]")
        finally:
            if metadata:
                await metadata.disconnect()
            await backend.disconnect()

    _run_async(_history())
