"""Minimal LLM observability SDK.

Usage (global singleton):

    from aiobs import observer
    observer.observe()  # enable instrumentation
    # ... make LLM calls ...
    observer.end()      # end current session
    observer.flush()    # write a single JSON file to disk

Extensible provider model with OpenAI support out of the box.

Function tracing with @observe decorator:

    from aiobs import observe

    @observe
    def my_function():
        ...

    @observe(name="custom_name")
    async def my_async_function():
        ...
"""

from .collector import Collector
from .providers.base import BaseProvider
from .observe import observe

# Global collector singleton, intentionally simple API
observer = Collector()

__all__ = ["observer", "observe", "Collector", "BaseProvider"]
