"""Quick-try search command — zero config file needed."""

import asyncio
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command("try")
@click.argument("query")
@click.option(
    "--database", "-d", default=None, help="Database name (or SVR_DATABASE env var)"
)
@click.option(
    "--collection",
    "-c",
    default=None,
    help="Collection/table name (or SVR_COLLECTION)",
)
@click.option(
    "--partition-field",
    "-p",
    default=None,
    help="Partition field (or SVR_PARTITION_FIELD)",
)
@click.option(
    "--partitions",
    default=None,
    help="Comma-separated partition names (default: all)",
)
@click.option("--limit", "-l", default=5, help="Max results (default: 5)")
@click.option(
    "--backend",
    "-b",
    default="mongodb",
    type=click.Choice(["mongodb", "postgres"]),
)
@click.option(
    "--embedding-provider", default=None, help="Embedding provider (auto-detected)"
)
def try_command(
    query: str,
    database: Optional[str],
    collection: Optional[str],
    partition_field: Optional[str],
    partitions: Optional[str],
    limit: int,
    backend: str,
    embedding_provider: Optional[str],
) -> None:
    """Run a quick search without a config file.

    Example:
        svr try "wireless headphones" -d my_store -c products -p category
    """
    asyncio.run(
        _try_search(
            query=query,
            database=database,
            collection=collection,
            partition_field=partition_field,
            partitions=partitions.split(",") if partitions else None,
            limit=limit,
            backend=backend,
            embedding_provider=embedding_provider,
        )
    )


async def _try_search(
    query: str,
    database: Optional[str],
    collection: Optional[str],
    partition_field: Optional[str],
    partitions: Optional[list[str]],
    limit: int,
    backend: str,
    embedding_provider: Optional[str],
) -> None:
    """Execute the try search."""
    from semantic_vector_router import SVRClient

    try:
        with console.status("[bold blue]Connecting and searching..."):
            svr = await SVRClient.quickstart(
                database=database,
                collection=collection,
                partition_field=partition_field,
                backend=backend,
                embedding_provider=embedding_provider,
            )

            results = await svr.search(
                query=query,
                partitions=partitions,
                limit=limit,
            )

        # Display results
        if not results.hits:
            console.print("[yellow]No results found.[/yellow]")
            console.print(
                "[dim]Try a different query or check that documents "
                "exist in your collection.[/dim]"
            )
            await svr.disconnect()
            return

        table = Table(
            title=f'Search results for: "{query}"',
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Partition", width=15)
        table.add_column("Content", max_width=80)

        for i, hit in enumerate(results.hits, 1):
            doc = hit.document
            preview = _doc_preview(doc)
            partition = doc.get(partition_field or "", "—")
            table.add_row(
                str(i),
                f"{hit.score:.4f}",
                str(partition),
                preview,
            )

        console.print(table)
        console.print(
            f"\n[dim]{len(results.hits)} results in "
            f"{results.latency_ms:.0f}ms | "
            f"Backend: {backend} | "
            f"Partitions searched: {len(results.partitions_searched)}[/dim]"
        )

        await svr.disconnect()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


def _doc_preview(doc: dict, max_len: int = 75) -> str:
    """Create a human-readable preview of a document."""
    # Try common content fields
    for field in ["title", "name", "text", "content", "description", "summary"]:
        if field in doc and doc[field]:
            text = str(doc[field])
            return text[:max_len] + "..." if len(text) > max_len else text

    # Fallback: show first few key-value pairs
    preview_parts = []
    for key, value in doc.items():
        if key.startswith("_") or key == "embedding":
            continue
        preview_parts.append(f"{key}: {str(value)[:30]}")
        if len(preview_parts) >= 3:
            break
    return " | ".join(preview_parts) if preview_parts else "(no preview)"
