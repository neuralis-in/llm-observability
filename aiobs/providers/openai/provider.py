"""OpenAI provider using OpenTelemetry instrumentation.

This provider uses the official OpenTelemetry OpenAI instrumentation
(opentelemetry-instrumentation-openai-v2) to automatically trace OpenAI API calls.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI provider using OTel instrumentation.
    
    This provider delegates to the official OpenTelemetry OpenAI instrumentor.
    The actual instrumentation is installed by the Collector, but this class
    provides backward compatibility for users who explicitly register providers.
    """
    
    name = "openai"

    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenAI OTel instrumentation is available."""
        try:
            from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor  # noqa: F401
            return True
        except ImportError:
            return False

    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        """Install OpenAI OTel instrumentation.
        
        Note: The Collector already installs this automatically.
        This method is kept for backward compatibility.
        """
        try:
            from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
            
            instrumentor = OpenAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                return lambda: OpenAIInstrumentor().uninstrument()
        except ImportError:
            pass
        except Exception:
            pass
        
        return None
