"""Webhook management CLI commands."""

import click
from rich.console import Console
from rich.table import Table

from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error
from semantic_vector_router.config import load_config

console = Console()


@click.group()
def webhooks_group():
    """Manage webhook endpoints."""
    pass


@webhooks_group.command("list")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def webhooks_list(config_path):
    """Show configured webhooks."""
    config = load_config(config_path=config_path)
    webhooks = config.events.webhooks

    if not webhooks:
        console.print("[yellow]No webhooks configured.[/yellow]")
        console.print("[dim]Add webhooks under events.webhooks in your config.[/dim]")
        return

    table = Table(title="Webhooks", show_header=True, header_style="bold cyan")
    table.add_column("URL")
    table.add_column("Events")
    table.add_column("Status")
    table.add_column("Secret")

    for wh in webhooks:
        events_str = ", ".join(wh.events) if wh.events else "[dim]all[/dim]"
        status = "[green]enabled[/green]" if wh.enabled else "[red]disabled[/red]"
        secret="<redacted>" if wh.secret else "[dim]none[/dim]"

        # Truncate long URLs
        url = wh.url
        if len(url) > 50:
            url = url[:47] + "..."

        table.add_row(url, events_str, status, secret)

    console.print(table)


@webhooks_group.command("test")
@click.argument("url")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def webhooks_test(url, config_path):
    """Send a test event to verify endpoint connectivity."""

    async def _test():
        from semantic_vector_router.events.webhook import WebhookConfig, WebhookDispatcher

        webhook = WebhookConfig(url=url)
        dispatcher = WebhookDispatcher([webhook])

        try:
            console.print(f"[dim]Sending test event to {url}...[/dim]")
            result = await dispatcher.test_webhook(webhook)

            if result.success:
                console.print(f"[green]Success![/green] Status: {result.status_code}, "
                              f"Response time: {result.response_time_ms:.0f}ms")
            else:
                console.print(f"[red]Failed![/red] "
                              f"Status: {result.status_code or 'N/A'}, "
                              f"Error: {result.error or 'Unknown'}")
        finally:
            await dispatcher.close()

    _run_async(_test())


@webhooks_group.command("history")
@click.option("--url", "-u", default=None, help="Filter by webhook URL")
@click.option("--limit", "-n", default=20, help="Number of records")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def webhooks_history(url, limit, config_path):
    """Show recent webhook delivery log."""

    async def _history():
        config, backend = await _get_backend(config_path)
        try:
            from semantic_vector_router.backends.metadata import MetadataStore
            metadata = MetadataStore(config)
            metadata._set_shared_db(backend._db)
            await metadata.connect()

            query = {"type": "webhook_delivery"}
            if url:
                query["webhook_url"] = url

            cursor = metadata._coll.find(query).sort("delivered_at", -1).limit(limit)
            docs = await cursor.to_list(length=limit)

            if not docs:
                console.print("[yellow]No webhook delivery history found.[/yellow]")
                return

            table = Table(title="Webhook Delivery Log", show_header=True, header_style="bold cyan")
            table.add_column("Event")
            table.add_column("URL")
            table.add_column("Status")
            table.add_column("HTTP")
            table.add_column("Attempts")

            for doc in docs:
                status = doc.get("status", "unknown")
                color = {"delivered": "green", "failed": "red", "pending": "yellow"}.get(status, "white")
                wh_url = doc.get("webhook_url", "?")
                if len(wh_url) > 35:
                    wh_url = wh_url[:32] + "..."

                table.add_row(
                    doc.get("event_type", "?"),
                    wh_url,
                    f"[{color}]{status}[/{color}]",
                    str(doc.get("response_status", "-")),
                    str(doc.get("attempts", 1)),
                )

            console.print(table)
        finally:
            await backend.disconnect()

    _run_async(_history())
