"""Analyze CLI command for collection field analysis."""

import click
from rich.console import Console
from rich.table import Table

from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error
from semantic_vector_router.utils.field_analyzer import (
    analyze_fields,
    get_recommended_filter_fields,
)

console = Console()


@click.command()
@click.option("--field", default=None, help="Analyze a specific field")
@click.option("--filters", is_flag=True, help="Focus on filter field detection")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def analyze_command(field, filters, config_path):
    """Analyze collection fields for partitioning and filtering."""

    async def _analyze():
        config, backend = await _get_backend(config_path)
        try:
            analyses = await analyze_fields(backend, config)

            if not analyses:
                console.print("[yellow]No fields found to analyze.[/yellow]")
                return

            if field:
                # Show detailed analysis for a single field
                match = [a for a in analyses if a.name == field]
                if not match:
                    console.print(f"[red]Field '{field}' not found in analysis.[/red]")
                    return

                a = match[0]
                console.print(f"\n[bold]Field Analysis: {a.name}[/bold]")
                console.print(f"  Distinct values: {a.distinct_count:,}")
                console.print(f"  Total documents: {a.total_documents:,}")
                console.print(f"  Coverage: {a.coverage:.1%}")
                console.print(f"  Cardinality ratio: {a.cardinality_ratio:.2%}")
                console.print(f"  Suitable for filtering: {'[green]Yes[/green]' if a.is_suitable else '[red]No[/red]'}")
                console.print(f"  Reason: {a.reason}")
                if a.sample_values:
                    console.print(f"  Sample values: {a.sample_values}")

            elif filters:
                # Focus on filter recommendations
                recommended = get_recommended_filter_fields(analyses)

                table = Table(title="Filter Field Analysis", show_header=True, header_style="bold cyan")
                table.add_column("Field")
                table.add_column("Suitable")
                table.add_column("Distinct", justify="right")
                table.add_column("Coverage", justify="right")
                table.add_column("Cardinality", justify="right")
                table.add_column("Reason")

                for a in analyses:
                    suitable_str = "[green]Yes[/green]" if a.is_suitable else "[red]No[/red]"
                    table.add_row(
                        a.name,
                        suitable_str,
                        f"{a.distinct_count:,}" if a.distinct_count else "-",
                        f"{a.coverage:.0%}",
                        f"{a.cardinality_ratio:.2%}" if a.distinct_count else "-",
                        a.reason,
                    )

                console.print(table)

                if recommended:
                    console.print(f"\n[green]Recommended filter fields:[/green] {', '.join(recommended)}")
                else:
                    console.print("\n[yellow]No fields recommended for filtering.[/yellow]")

            else:
                # Full analysis
                table = Table(title="Field Analysis", show_header=True, header_style="bold cyan")
                table.add_column("Field")
                table.add_column("Distinct", justify="right")
                table.add_column("Coverage", justify="right")
                table.add_column("Suitable")
                table.add_column("Reason")

                for a in analyses:
                    suitable_str = "[green]Yes[/green]" if a.is_suitable else "[red]No[/red]"
                    table.add_row(
                        a.name,
                        f"{a.distinct_count:,}" if a.distinct_count else "-",
                        f"{a.coverage:.0%}",
                        suitable_str,
                        a.reason,
                    )

                console.print(table)

        finally:
            await backend.disconnect()

    _run_async(_analyze())
