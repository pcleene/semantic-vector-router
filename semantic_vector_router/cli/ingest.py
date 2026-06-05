"""CLI command for document ingestion."""

import json
import sys

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from semantic_vector_router.cli.helpers import _run_async, handle_config_error

# Module-level imports for mockability
from semantic_vector_router.client import SVRClient
from semantic_vector_router.config import load_config
from semantic_vector_router.models import IngestMode

console = Console()


def _load_documents(file_path: str) -> list[dict]:
    """Load documents from a JSON or JSONL file.

    Args:
        file_path: Path to file, or "-" for stdin.

    Returns:
        List of document dictionaries.

    Raises:
        click.ClickException: If file cannot be read or parsed.
    """
    if file_path == "-":
        content = sys.stdin.read()
    else:
        try:
            with open(file_path) as f:
                content = f.read()
        except FileNotFoundError:
            raise click.ClickException(f"File not found: {file_path}")
        except OSError as e:
            raise click.ClickException(f"Error reading file: {e}")

    if not content.strip():
        raise click.ClickException("Empty input")

    # Try JSON array first, then JSONL
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise click.ClickException(
                "JSON must be an array of objects or a single object"
            )
    except json.JSONDecodeError:
        pass

    # Try JSONL (one JSON object per line)
    documents = []
    for i, line in enumerate(content.strip().split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
            if not isinstance(doc, dict):
                raise click.ClickException(
                    f"Line {i}: Expected JSON object, got {type(doc).__name__}"
                )
            documents.append(doc)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Line {i}: Invalid JSON: {e}")

    if not documents:
        raise click.ClickException("No valid documents found in input")
    return documents


@click.command("ingest")
@click.argument("file", type=click.Path(exists=False), default="-")
@click.option("-c", "--config", "config_path", required=True, help="Config file path.")
@click.option("--partition", default=None, help="Route all documents to this partition.")
@click.option(
    "--mode",
    type=click.Choice(["insert", "upsert"]),
    default=None,
    help="Ingestion mode (default: from config).",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Override embedding batch size.",
)
@handle_config_error
def ingest_command(
    file: str,
    config_path: str,
    partition: str,
    mode: str,
    batch_size: int,
) -> None:
    """Ingest documents from a JSON or JSONL file.

    FILE can be a path to a .json (array of objects) or .jsonl (one object per
    line) file, or "-" to read from stdin.
    """
    # Load documents
    documents = _load_documents(file)
    console.print(f"Loaded [bold]{len(documents)}[/bold] documents")

    # Load config
    config = load_config(config_path=config_path)

    # Override batch size if specified
    if batch_size is not None:
        config.ingestion.batch_size = batch_size

    # Parse mode
    ingest_mode = IngestMode(mode) if mode else None

    # Progress tracking state shared between callback and main code
    progress_state: dict = {"progress_bar": None, "task_id": None}

    def progress_callback(prog):
        """Update Rich progress bar from IngestProgress."""
        p = progress_state.get("progress_bar")
        tid = progress_state.get("task_id")
        if p is not None and tid is not None:
            completed = prog.embedded + prog.written
            p.update(tid, completed=completed, description=f"[cyan]{prog.phase}[/cyan]")

    async def _do_ingest():
        client = SVRClient(config=config, auto_connect=False)
        try:
            await client.connect()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    "[cyan]embedding[/cyan]",
                    total=len(documents) * 2,
                )
                progress_state["progress_bar"] = progress
                progress_state["task_id"] = task_id

                result = await client.ingest(
                    documents=documents,
                    partition=partition,
                    mode=ingest_mode,
                    progress_callback=progress_callback,
                )

                progress.update(task_id, completed=len(documents) * 2)

            # Print summary
            console.print()
            console.print("[bold green]Ingestion complete[/bold green]")
            console.print(f"  Inserted: [green]{result.inserted}[/green]")
            if result.failed > 0:
                console.print(f"  Failed:   [red]{result.failed}[/red]")
                for idx, err in result.errors[:10]:
                    console.print(f"    Doc {idx}: {err}")
                if len(result.errors) > 10:
                    console.print(
                        f"    ... and {len(result.errors) - 10} more errors"
                    )
            console.print(f"  Total time: {result.elapsed_ms:.0f}ms")
            console.print(f"    Embedding: {result.embed_ms:.0f}ms")
            console.print(f"    Writing:   {result.write_ms:.0f}ms")

        finally:
            await client.disconnect()

    _run_async(_do_ingest())
