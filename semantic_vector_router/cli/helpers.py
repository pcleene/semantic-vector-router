"""Shared helpers for CLI commands."""

import asyncio

from rich.console import Console

from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.config import load_config
from semantic_vector_router.exceptions import ConfigurationError

console = Console()


def _run_async(coro):
    """Run async code from Click (sync) commands."""
    return asyncio.run(coro)


async def _get_backend(config_path=None):
    """Load config and connect to MongoDB. Caller must disconnect."""
    config = load_config(config_path=config_path)
    backend = MongoDBBackend(config)
    await backend.connect()
    return config, backend


def handle_config_error(func):
    """Decorator to handle ConfigurationError with a user-friendly message."""
    import functools

    import click

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConfigurationError as e:
            console.print(f"[red]Configuration error:[/red] {e}")
            console.print("[dim]Run 'svr init' to create a configuration file.[/dim]")
            raise click.Abort()

    return wrapper
