"""CLI entry point for Semantic Vector Router."""

import click

from semantic_vector_router.cli.analyze import analyze_command
from semantic_vector_router.cli.cache import cache_group
from semantic_vector_router.cli.config_cmd import config_group
from semantic_vector_router.cli.index import index_group
from semantic_vector_router.cli.ingest import ingest_command
from semantic_vector_router.cli.init import init_command
from semantic_vector_router.cli.monitor import monitor_group
from semantic_vector_router.cli.partitions import partitions_group
from semantic_vector_router.cli.repartition import repartition_group
from semantic_vector_router.cli.schedule import schedule_group
from semantic_vector_router.cli.search import search_command
from semantic_vector_router.cli.split import split_group
from semantic_vector_router.cli.try_search import try_command
from semantic_vector_router.cli.watch import watch_group
from semantic_vector_router.cli.webhooks import webhooks_group


@click.group()
@click.version_option(version="0.1.0", prog_name="svr")
def main() -> None:
    """Semantic Vector Router - Automatic vector index partitioning and query routing."""
    pass


# Register commands
main.add_command(init_command, name="init")
main.add_command(partitions_group, name="partitions")
main.add_command(search_command, name="search")
main.add_command(analyze_command, name="analyze")
main.add_command(watch_group, name="watch")
main.add_command(split_group, name="split")
main.add_command(config_group, name="config")
main.add_command(index_group, name="index")
main.add_command(monitor_group, name="monitor")
main.add_command(repartition_group, name="repartition")
main.add_command(cache_group, name="cache")
main.add_command(ingest_command, name="ingest")
main.add_command(schedule_group, name="schedule")
main.add_command(webhooks_group, name="webhooks")
main.add_command(try_command, name="try")


if __name__ == "__main__":
    main()
