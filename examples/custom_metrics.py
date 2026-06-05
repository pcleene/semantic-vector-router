"""Custom metrics handler: capture and display SVR telemetry.

Demonstrates the MetricsHandler protocol for integrating with
Prometheus, Datadog, StatsD, or any monitoring backend.

SVR emits 18 metric types covering search latency, embedding latency,
cache hit rates, ingestion throughput, rate limiter waits, and errors.
"""
import asyncio
from collections import defaultdict

from semantic_vector_router import SVRClient
from semantic_vector_router.utils.metrics import MetricEvent


class InMemoryMetrics:
    """Simple in-memory metrics collector for demonstration."""

    def __init__(self):
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)

    def handle(self, event: MetricEvent) -> None:
        """Receive metric events from SVR."""
        if event.value is not None:
            self.timings[event.metric_type.value].append(event.value)
        self.counts[event.metric_type.value] += 1

    def report(self):
        """Print a summary of collected metrics."""
        print("\n--- Metrics Report ---")
        for metric, values in sorted(self.timings.items()):
            avg = sum(values) / len(values) if values else 0
            print(f"  {metric}: count={len(values)}, avg={avg:.1f}ms")
        for metric, count in sorted(self.counts.items()):
            if metric not in self.timings:
                print(f"  {metric}: count={count}")


async def main():
    metrics = InMemoryMetrics()
    svr = SVRClient(auto_connect=False, metrics_handler=metrics)
    await svr.connect()

    # Run searches to generate metrics
    queries = ["headphones", "desk chair", "laptop", "mechanical keyboard"]
    for query in queries:
        await svr.search(query, partitions=["electronics"], limit=5)

    metrics.report()
    await svr.disconnect()


asyncio.run(main())
