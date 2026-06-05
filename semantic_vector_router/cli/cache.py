"""CLI commands for embedding cache management."""

import click

from semantic_vector_router.cli.helpers import _run_async, handle_config_error


@click.group()
def cache_group() -> None:
    """Manage the embedding cache."""
    pass


@cache_group.command("stats")
@click.option("-c", "--config", "config_path", default=None, help="Path to config file.")
@handle_config_error
def cache_stats(config_path: str) -> None:
    """Show embedding cache statistics."""

    async def _stats():
        from semantic_vector_router.client import SVRClient

        client = SVRClient(config_path=config_path, auto_connect=False)
        cache = client._embedding_cache
        stats = cache.stats()

        click.echo("Embedding Cache")
        click.echo(f"  Entries: {stats['size']:,} / {stats['max_size']:,}")
        hit_rate_pct = stats["hit_rate"] * 100
        click.echo(f"  Hit rate: {hit_rate_pct:.1f}%")
        click.echo(f"  Hits: {stats['hits']:,}")
        click.echo(f"  Misses: {stats['misses']:,}")
        click.echo(f"  Evictions: {stats['evictions']:,}")
        click.echo(f"  TTL: {client.config.cache.ttl_seconds}s")
        if not client.config.cache.enabled:
            click.echo("  Status: DISABLED")

    _run_async(_stats())
