"""In-process event bus for SVR lifecycle events."""

import asyncio
import logging

from semantic_vector_router.events.models import EventHandler, SVREvent, SVREventType

logger = logging.getLogger("semantic_vector_router.events")


class EventBus:
    """In-process event dispatch with async handlers.

    Handlers are called fire-and-forget — a slow handler doesn't block
    the emitter. Handler exceptions are logged, never propagated.
    """

    def __init__(self) -> None:
        self._handlers: dict[SVREventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []

    def subscribe(self, event_type: SVREventType, handler: EventHandler) -> None:
        """Subscribe handler to a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe handler to ALL event types."""
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)

    def unsubscribe(self, event_type: SVREventType, handler: EventHandler) -> None:
        """Unsubscribe handler from a specific event type."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def unsubscribe_all(self, handler: EventHandler) -> None:
        """Unsubscribe handler from all event types."""
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)
        for handlers in self._handlers.values():
            if handler in handlers:
                handlers.remove(handler)

    async def emit(self, event: SVREvent) -> None:
        """Emit event to all matching subscribers. Non-blocking.

        Dispatches to:
        1. Handlers subscribed to the specific event type
        2. Global handlers (subscribed via subscribe_all)

        Each handler is called as a fire-and-forget task.
        Exceptions are caught and logged.
        """
        handlers: list[EventHandler] = []
        handlers.extend(self._handlers.get(event.event_type, []))
        handlers.extend(self._global_handlers)

        for handler in handlers:
            try:
                # Fire-and-forget: create task so we don't block
                asyncio.ensure_future(self._safe_handle(handler, event))
            except Exception as e:
                logger.warning(
                    f"Failed to dispatch event {event.event_type.value} "
                    f"to {type(handler).__name__}: {e}"
                )

    async def _safe_handle(self, handler: EventHandler, event: SVREvent) -> None:
        """Call handler with exception catching."""
        try:
            await handler.handle_event(event)
        except Exception as e:
            logger.warning(
                f"Event handler {type(handler).__name__} failed for "
                f"{event.event_type.value}: {e}",
                exc_info=True,
            )
