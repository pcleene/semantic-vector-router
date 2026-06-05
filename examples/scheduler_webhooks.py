"""Example: Job Scheduler with Maintenance Windows and Webhook Events.

Demonstrates configuring SVR with:
- A job scheduler that runs detection and centroid refresh on intervals
- Maintenance windows that restrict operations to off-peak hours
- Webhook endpoints that receive event notifications with HMAC signing

Usage:
    python examples/scheduler_webhooks.py
"""

import asyncio

from semantic_vector_router import SVRClient
from semantic_vector_router.models import SVRConfig


async def main():
    config = SVRConfig.model_validate({
        "database": {
            "connection_string_env": "MONGODB_URI",
            "database": "my_database",
            "source_collection": "documents",
        },
        "partitioning": {
            "field": "category",
        },
        "embedding": {
            "provider": "voyage",
            "model": "voyage-3-large",
            "dimensions": 1024,
        },
        # --- Scheduler configuration ---
        "scheduler": {
            "enabled": True,
            "tick_interval_seconds": 60,
            # Maintenance window: only run jobs 2-5am UTC on weekends
            "maintenance_window": {
                "allowed_days": ["saturday", "sunday"],
                "allowed_hours": {"start": 2, "end": 5},
                "timezone": "UTC",
            },
            # Built-in job intervals
            "detection_interval": "1h",
            "centroid_refresh_interval": "6h",
            "count_update_interval": "30m",
        },
        # --- Events configuration ---
        "events": {
            "enabled": True,
            "log_events": True,
            "webhooks": [
                {
                    "url": "https://hooks.slack.com/services/T00/B00/xxx",
                    "secret": "my-webhook-secret",
                    "events": [
                        "partition.created",
                        "partition.deleted",
                        "repartition.completed",
                        "health.threshold_breach",
                    ],
                    "timeout_seconds": 10,
                    "retry_count": 3,
                },
                {
                    "url": "https://my-app.example.com/svr-events",
                    "secret": "another-secret",
                    # Empty events list = receive ALL events
                    "events": [],
                },
            ],
        },
    })

    client = SVRClient(config)

    # connect() initializes the event bus and starts the scheduler
    await client.connect()

    print("Scheduler is running. Jobs will execute within maintenance windows.")
    print("Webhooks will fire on configured events.")
    print("Press Ctrl+C to stop.")

    try:
        # Keep the process alive — scheduler runs in background
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        # disconnect() stops the scheduler and cleans up
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
