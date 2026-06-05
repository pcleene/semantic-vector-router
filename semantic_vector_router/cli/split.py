"""Split CLI commands for partition splitting."""

import click
from rich.console import Console
from rich.table import Table

from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error

console = Console()


@click.group()
def split_group():
    """Check and execute partition splits."""
    pass


@split_group.command("check")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def split_check(config_path):
    """Check which partitions need splitting."""

    async def _check():
        config, backend = await _get_backend(config_path)
        try:
            if not config.lifecycle.auto_split or not config.lifecycle.auto_split.enabled:
                console.print("[yellow]Auto-split is not enabled in configuration.[/yellow]")
                return

            threshold = config.lifecycle.auto_split.threshold_vectors
            from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
            from semantic_vector_router.lifecycle.splitter import PartitionSplitter

            provisioner = PartitionProvisioner(backend, config)
            splitter = PartitionSplitter(backend, config, provisioner)

            table = Table(title="Split Check", show_header=True, header_style="bold cyan")
            table.add_column("Partition")
            table.add_column("Documents", justify="right")
            table.add_column("Threshold", justify="right")
            table.add_column("Status")

            needs_split = False
            for name, p in config.partitions.registry.items():
                if p.status.value in ("split", "disabled"):
                    continue

                count = p.document_count
                if count > threshold:
                    table.add_row(
                        name,
                        f"{count:,}",
                        f"{threshold:,}",
                        "[red]Needs split[/red]",
                    )
                    needs_split = True
                else:
                    table.add_row(
                        name,
                        f"{count:,}",
                        f"{threshold:,}",
                        "[green]OK[/green]",
                    )

            console.print(table)

            if not needs_split:
                console.print("\n[green]No partitions need splitting.[/green]")
            else:
                console.print("\n[yellow]Run 'svr split execute <name>' to split a partition.[/yellow]")
        finally:
            await backend.disconnect()

    _run_async(_check())


@split_group.command("execute")
@click.argument("name", required=False)
@click.option("--all", "split_all", is_flag=True, help="Execute all pending splits")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def split_execute(name, split_all, config_path):
    """Execute a partition split."""
    if not name and not split_all:
        console.print("[red]Provide a partition name or use --all.[/red]")
        return

    async def _execute():
        config, backend = await _get_backend(config_path)
        try:
            from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
            from semantic_vector_router.lifecycle.splitter import PartitionSplitter

            provisioner = PartitionProvisioner(backend, config)
            splitter = PartitionSplitter(backend, config, provisioner)

            if split_all:
                results = await splitter.execute_pending_splits()
                if results:
                    console.print(f"[green]Split {len(results)} partition(s): {', '.join(results)}[/green]")
                else:
                    console.print("[yellow]No pending splits to execute.[/yellow]")
            else:
                children = await splitter.execute_split(name)
                if children:
                    console.print(f"[green]Split '{name}' into {len(children)} child partitions:[/green]")
                    for child in children:
                        console.print(f"  - {child}")
                else:
                    console.print(f"[yellow]No children created for '{name}'.[/yellow]")
        finally:
            await backend.disconnect()

    _run_async(_execute())
