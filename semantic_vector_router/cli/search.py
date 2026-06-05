"""Search CLI command."""

import json
from typing import Optional, Union

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error
from semantic_vector_router.client import SVRClient
from semantic_vector_router.models import EmbeddingMode
from semantic_vector_router.routing.merger import ResultMerger
from semantic_vector_router.routing.resolver import PartitionResolver

console = Console()


@click.command()
@click.argument("query")
@click.option("--partitions", "-p", default=None, help="Comma-separated partition names or 'all'")
@click.option("--limit", "-l", default=10, help="Max results")
@click.option("--rerank/--no-rerank", default=None, help="Enable/disable reranking")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def search_command(query, partitions, limit, rerank, output_format, config_path):
    """Execute a vector search query."""

    async def _search():
        config, backend = await _get_backend(config_path)
        try:
            client = SVRClient(config=config, auto_connect=False)
            client._backend = backend
            client._connected = True

            # Initialize embedder if BYOM mode
            if config.embedding.mode == EmbeddingMode.BYOM:
                try:
                    client._embedder = client._create_embedder()
                except Exception as e:
                    console.print(f"[red]Embedder initialization failed:[/red] {e}")
                    console.print("[dim]Ensure your embedding API key is set in the environment.[/dim]")
                    return

            # Initialize reranker if enabled
            if config.reranking.enabled:
                try:
                    client._reranker = client._create_reranker()
                except Exception:
                    pass  # Non-critical, search will work without reranking

            # Initialize metadata store (for resolver dual-source mode)
            from semantic_vector_router.backends.metadata import MetadataStore

            metadata: Optional[MetadataStore] = None
            try:
                metadata = MetadataStore(config)
                if not config.lifecycle.metadata.connection_string_env:
                    metadata._set_shared_db(backend._db)
                await metadata.connect()
            except Exception:
                metadata = None  # Fall back to config

            # Initialize routing components
            client._resolver = PartitionResolver(config, metadata=metadata)
            client._merger = ResultMerger()
            client._metadata = metadata

            # Parse partitions
            partition_list: Optional[Union[str, list[str]]] = None
            if partitions:
                if partitions.lower() == "all":
                    partition_list = "all"
                else:
                    partition_list = [p.strip() for p in partitions.split(",")]

            result = await client.search(
                query=query,
                partitions=partition_list,
                limit=limit,
                rerank=rerank,
            )

            if output_format == "json":
                output = {
                    "query": result.query,
                    "partitions_searched": result.partitions_searched,
                    "total_candidates": result.total_candidates,
                    "reranked": result.reranked,
                    "latency_ms": round(result.latency_ms, 2),
                    "hits": [
                        {
                            "id": hit.id,
                            "score": round(hit.score, 4),
                            "rerank_score": round(hit.rerank_score, 4) if hit.rerank_score else None,
                            "partition": hit.partition,
                            "document": hit.document,
                        }
                        for hit in result.hits
                    ],
                }
                console.print(json.dumps(output, indent=2, default=str))
            else:
                # Diagnostic info
                console.print(Panel(
                    f"[bold]Query:[/bold] {query}\n"
                    f"[bold]Partitions:[/bold] {', '.join(result.partitions_searched)}\n"
                    f"[bold]Total candidates:[/bold] {result.total_candidates}\n"
                    f"[bold]Reranked:[/bold] {'Yes' if result.reranked else 'No'}\n"
                    f"[bold]Latency:[/bold] {result.latency_ms:.0f}ms",
                    title="Search Diagnostics",
                    border_style="blue",
                ))

                if not result.hits:
                    console.print("[yellow]No results found.[/yellow]")
                    return

                table = Table(title="Results", show_header=True, header_style="bold cyan")
                table.add_column("#", justify="right", width=4)
                table.add_column("Score", justify="right", width=8)
                if result.reranked:
                    table.add_column("Rerank", justify="right", width=8)
                table.add_column("Partition", width=15)
                table.add_column("Document", overflow="ellipsis")

                for i, hit in enumerate(result.hits, 1):
                    doc_str = str(hit.document)
                    if len(doc_str) > 100:
                        doc_str = doc_str[:100] + "..."

                    row = [
                        str(i),
                        f"{hit.score:.4f}",
                    ]
                    if result.reranked:
                        row.append(f"{hit.rerank_score:.4f}" if hit.rerank_score else "-")
                    row.extend([hit.partition, doc_str])
                    table.add_row(*row)

                console.print(table)
        finally:
            await backend.disconnect()

    _run_async(_search())
