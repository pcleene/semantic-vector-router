"""Scheduler CLI commands."""

import click
from rich.console import Console
from rich.table import Table

from semantic_vector_router.cli.helpers import _get_backend, _run_async, handle_config_error
from semantic_vector_router.config import load_config

console = Console()


@click.group()
def schedule_group():
    """Manage scheduled jobs and maintenance windows."""
    pass


@schedule_group.command("list")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def schedule_list(config_path):
    """Show all registered jobs with next run time."""
    config = load_config(config_path=config_path)

    if not config.scheduler.enabled:
        console.print("[yellow]Scheduler is disabled.[/yellow]")
        console.print("[dim]Set scheduler.enabled=true in config to enable.[/dim]")
        return

    table = Table(title="Scheduled Jobs", show_header=True, header_style="bold cyan")
    table.add_column("Job ID")
    table.add_column("Type")
    table.add_column("Interval")
    table.add_column("Status")

    sc = config.scheduler
    jobs = [
        ("detection", "detection", sc.detection_interval),
        ("centroid_refresh", "centroid_refresh", sc.centroid_refresh_interval),
        ("count_update", "partition_count_update", sc.count_update_interval),
        ("repartition_check", "repartition", sc.repartition_check_interval),
        ("index_health", "index_health_check", sc.index_health_interval),
    ]

    for job_id, job_type, interval in jobs:
        if interval:
            table.add_row(
                job_id,
                job_type,
                interval,
                "[green]enabled[/green]",
            )
        else:
            table.add_row(
                job_id,
                job_type,
                "-",
                "[dim]disabled[/dim]",
            )

    console.print(table)


@schedule_group.command("status")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def schedule_status(config_path):
    """Show scheduler status and configuration."""
    config = load_config(config_path=config_path)
    sc = config.scheduler

    console.print("[bold]Scheduler Status[/bold]")
    console.print(f"  Enabled: {'[green]yes[/green]' if sc.enabled else '[red]no[/red]'}")
    console.print(f"  Tick interval: {sc.tick_interval_seconds}s")
    console.print(f"  Worker ID: {sc.worker_id or '[dim]auto-generated[/dim]'}")

    if sc.maintenance_window:
        mw = sc.maintenance_window
        console.print("\n[bold]Maintenance Window[/bold]")
        console.print(f"  Days: {', '.join(mw.allowed_days)}")
        console.print(f"  Hours: {mw.allowed_hours.get('start', 0):02d}:00 - {mw.allowed_hours.get('end', 24):02d}:00")
        console.print(f"  Timezone: {mw.timezone}")


@schedule_group.command("history")
@click.option("--job", "-j", default=None, help="Filter by job ID")
@click.option("--limit", "-n", default=20, help="Number of records")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def schedule_history(job, limit, config_path):
    """Show job execution history."""

    async def _history():
        config, backend = await _get_backend(config_path)
        try:
            from semantic_vector_router.backends.metadata import MetadataStore
            metadata = MetadataStore(config)
            metadata._set_shared_db(backend._db)
            await metadata.connect()

            query = {"type": "job_run"}
            if job:
                query["job_id"] = job

            cursor = metadata._coll.find(query).sort("scheduled_at", -1).limit(limit)
            docs = await cursor.to_list(length=limit)

            if not docs:
                console.print("[yellow]No job history found.[/yellow]")
                return

            table = Table(title="Job History", show_header=True, header_style="bold cyan")
            table.add_column("Job ID")
            table.add_column("Status")
            table.add_column("Duration")
            table.add_column("Worker")
            table.add_column("Time")

            for doc in docs:
                status = doc.get("status", "unknown")
                color = {"completed": "green", "failed": "red", "skipped": "yellow"}.get(status, "white")
                duration = doc.get("duration_ms")
                duration_str = f"{duration:.0f}ms" if duration else "-"
                scheduled = doc.get("scheduled_at")
                time_str = scheduled.strftime("%Y-%m-%d %H:%M:%S") if scheduled else "-"

                table.add_row(
                    doc.get("job_id", "?"),
                    f"[{color}]{status}[/{color}]",
                    duration_str,
                    doc.get("worker_id", "-"),
                    time_str,
                )

            console.print(table)
        finally:
            await backend.disconnect()

    _run_async(_history())


@schedule_group.command("run")
@click.argument("job_id")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def schedule_run(job_id, config_path):
    """Force-execute a job NOW (bypasses maintenance window)."""
    console.print(f"[yellow]Force-running job '{job_id}'...[/yellow]")
    console.print("[dim]Note: Force-run requires an active SVRClient with scheduler enabled.[/dim]")
    console.print(f"[dim]Use the SDK: await svr._scheduler.run_now('{job_id}')[/dim]")


@schedule_group.command("pause")
@click.argument("job_id")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def schedule_pause(job_id, config_path):
    """Disable a job without removing it."""
    console.print(f"[yellow]Pausing job '{job_id}'...[/yellow]")
    console.print("[dim]Note: Job pausing requires an active SVRClient with scheduler enabled.[/dim]")


@schedule_group.command("resume")
@click.argument("job_id")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def schedule_resume(job_id, config_path):
    """Re-enable a paused job."""
    console.print(f"[green]Resuming job '{job_id}'...[/green]")
    console.print("[dim]Note: Job resuming requires an active SVRClient with scheduler enabled.[/dim]")


@schedule_group.command("window")
@click.option("--check", is_flag=True, help="Check if window is currently open")
@click.option("--config-path", "-c", default=None, help="Path to config file")
@handle_config_error
def schedule_window(check, config_path):
    """Show or check maintenance window configuration."""
    config = load_config(config_path=config_path)
    mw = config.scheduler.maintenance_window

    if mw is None:
        console.print("[dim]No maintenance window configured (jobs run anytime).[/dim]")
        return

    if check:
        from semantic_vector_router.scheduler.models import MaintenanceWindow
        from semantic_vector_router.scheduler.window import is_within_window

        window = MaintenanceWindow(
            allowed_days=mw.allowed_days,
            allowed_hours=mw.allowed_hours,
            timezone=mw.timezone,
        )
        is_open = is_within_window(window)
        if is_open:
            console.print("[green]Maintenance window is OPEN.[/green]")
        else:
            console.print("[red]Maintenance window is CLOSED.[/red]")
    else:
        console.print("[bold]Maintenance Window[/bold]")
        console.print(f"  Days: {', '.join(mw.allowed_days)}")
        console.print(f"  Hours: {mw.allowed_hours.get('start', 0):02d}:00 - {mw.allowed_hours.get('end', 24):02d}:00")
        console.print(f"  Timezone: {mw.timezone}")
