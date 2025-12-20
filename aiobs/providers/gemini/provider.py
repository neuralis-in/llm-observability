"""Gemini provider using OpenTelemetry instrumentation.

This provider uses the OpenTelemetry Google GenAI instrumentation
(opentelemetry-instrumentation-google-genai) to automatically trace Gemini API calls.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..base import BaseProvider


class GeminiProvider(BaseProvider):
    """Gemini provider using OTel instrumentation.
    
    This provider delegates to the OpenTelemetry Google GenAI instrumentor.
    The actual instrumentation is installed by the Collector, but this class
    provides backward compatibility for users who explicitly register providers.
    """
    
    name = "gemini"

    @classmethod
    def is_available(cls) -> bool:
        """Check if Gemini OTel instrumentation is available."""
        try:
            from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor  # noqa: F401
            return True
        except ImportError:
            # Try Vertex AI instrumentor as fallback
            try:
                from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor  # noqa: F401
                return True
            except ImportError:
                return False

    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        """Install Gemini OTel instrumentation.
        
        Note: The Collector already installs this automatically.
        This method is kept for backward compatibility.
        """
        unpatchers = []
        
        # Try Google GenAI instrumentor
        try:
            from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
            
            instrumentor = GoogleGenAiSdkInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                unpatchers.append(lambda: GoogleGenAiSdkInstrumentor().uninstrument())
        except ImportError:
            pass
        except Exception:
            pass
        
        # Also try Vertex AI instrumentor
        try:
            from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor
            
            instrumentor = VertexAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                unpatchers.append(lambda: VertexAIInstrumentor().uninstrument())
        except ImportError:
            pass
        except Exception:
            pass
        
        if unpatchers:
            def unpatch_all():
                for up in unpatchers:
                    try:
                        up()
                    except Exception:
                        pass
            return unpatch_all
        
        return None
