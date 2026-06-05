"""Configuration CLI commands."""

import json
import re
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from semantic_vector_router.cli.helpers import handle_config_error
from semantic_vector_router.config import (
    find_config_file,
    load_config,
    save_config,
    validate_config,
)
from semantic_vector_router.models import SVRConfig

console = Console()

# Keys whose values should be redacted
SENSITIVE_PATTERNS = re.compile(r"(key|secret|password|token)", re.IGNORECASE)


def _redact_dict(d):
    """Recursively redact sensitive values in a dictionary."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _redact_dict(v)
        elif SENSITIVE_PATTERNS.search(k) and isinstance(v, str) and v:
            result[k] = "****"
        else:
            result[k] = v
    return result


@click.group()
def config_group():
    """View and manage SVR configuration."""
    pass


@config_group.command("show")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def config_show(config_path):
    """Display current configuration (API keys redacted)."""
    config = load_config(config_path=config_path)
    config_dict = config.model_dump(mode="json", exclude_none=True)
    redacted = _redact_dict(config_dict)
    config_json = json.dumps(redacted, indent=2, default=str)

    syntax = Syntax(config_json, "json", theme="monokai", line_numbers=False)
    console.print(Panel(syntax, title="SVR Configuration", border_style="blue"))


@config_group.command("validate")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def config_validate(config_path):
    """Validate configuration and show warnings."""
    config = load_config(config_path=config_path)
    warnings = validate_config(config)

    if warnings:
        for w in warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")
    else:
        console.print("[green]Configuration is valid. No warnings.[/green]")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def config_set(key, value, config_path):
    """Update a configuration value (supports dot-notation keys)."""
    config = load_config(config_path=config_path)
    config_dict = config.model_dump(mode="json", exclude_none=True)

    # Navigate dot-notation path
    parts = key.split(".")
    current = config_dict
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            console.print(f"[red]Invalid key path: {key}[/red]")
            return
        current = current[part]

    last_key = parts[-1]
    if last_key not in current:
        console.print(f"[yellow]Warning: Creating new key '{key}'[/yellow]")

    # Try to parse value as JSON for type preservation
    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        parsed_value = value

    current[last_key] = parsed_value

    # Rebuild and save config
    new_config = SVRConfig.model_validate(config_dict)

    # Determine save path
    if config_path:
        save_path: Optional[Path] = Path(config_path)
    else:
        save_path = find_config_file()

    save_config(new_config, save_path)
    console.print(f"[green]Set {key} = {parsed_value}[/green]")


@config_group.command("path")
@click.option("--config-path", "-c", default=None, help="Path to config file")
def config_path_cmd(config_path):
    """Show the resolved config file path."""
    if config_path:
        console.print(config_path)
    else:
        found = find_config_file()
        if found:
            console.print(str(found))
        else:
            console.print("[yellow]No config file found.[/yellow]")
            console.print("[dim]Run 'svr init' to create a configuration file.[/dim]")
