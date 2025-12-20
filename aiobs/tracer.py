"""OpenTelemetry tracer configuration for aiobs.

This module provides centralized OTel tracer initialization with in-memory
exporters for collecting spans and logs that will be converted to aiobs format on flush.
"""

from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING

# Enable GenAI message content capture by default
# This allows OTel instrumentors to capture full prompt/completion content
if "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT" not in os.environ:
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk._logs import LogData

_provider = None
_exporter = None
_log_provider = None
_log_exporter = None
_initialized = False


def init_tracer() -> None:
    """Initialize the OTel tracer and logger with in-memory exporters.
    
    This should be called once when observer.observe() is invoked.
    Subsequent calls are no-ops. If OTel providers are already set globally
    (e.g., in tests), we only create our exporters.
    """
    global _provider, _exporter, _log_provider, _log_exporter, _initialized
    
    if _initialized:
        return
    
    # Set up tracing
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    
    _exporter = InMemorySpanExporter()
    _provider = TracerProvider()
    _provider.add_span_processor(SimpleSpanProcessor(_exporter))
    
    # Try to set as global provider, but don't fail if already set
    current_provider = trace.get_tracer_provider()
    if not isinstance(current_provider, TracerProvider):
        # No SDK provider set yet, set ours
        trace.set_tracer_provider(_provider)
    else:
        # SDK provider already set - add our exporter to existing provider
        try:
            current_provider.add_span_processor(SimpleSpanProcessor(_exporter))
        except Exception:
            # If that fails, just use the provider as-is
            pass
    
    # Set up logging for GenAI message content capture
    try:
        from opentelemetry import _logs
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, InMemoryLogExporter
        
        _log_exporter = InMemoryLogExporter()
        _log_provider = LoggerProvider()
        _log_provider.add_log_record_processor(SimpleLogRecordProcessor(_log_exporter))
        
        # Try to set as global log provider, but don't fail if already set
        try:
            _logs.set_logger_provider(_log_provider)
        except Exception:
            # Already set, try to add processor to existing provider
            try:
                current_log_provider = _logs.get_logger_provider()
                if hasattr(current_log_provider, 'add_log_record_processor'):
                    current_log_provider.add_log_record_processor(SimpleLogRecordProcessor(_log_exporter))
            except Exception:
                pass
    except ImportError:
        # Logging SDK not available - message content won't be captured
        pass
    
    _initialized = True


def get_tracer():
    """Get the aiobs tracer.
    
    Returns:
        The OpenTelemetry Tracer for aiobs instrumentation.
    """
    from opentelemetry import trace
    return trace.get_tracer("aiobs", "0.1.0")


def get_finished_spans() -> List["ReadableSpan"]:
    """Get all finished spans from the in-memory exporter.
    
    Returns:
        List of finished ReadableSpan objects.
    """
    if _exporter is None:
        return []
    return list(_exporter.get_finished_spans())


def clear_spans() -> None:
    """Clear all collected spans from the exporter."""
    if _exporter is not None:
        _exporter.clear()


def get_finished_logs() -> List["LogData"]:
    """Get all finished logs from the in-memory log exporter.
    
    Returns:
        List of finished LogData objects containing message content.
    """
    if _log_exporter is None:
        return []
    return list(_log_exporter.get_finished_logs())


def clear_logs() -> None:
    """Clear all collected logs from the log exporter."""
    if _log_exporter is not None:
        _log_exporter.clear()


def reset_tracer() -> None:
    """Reset tracer state completely.
    
    This is primarily for testing - clears spans and resets initialization state.
    """
    global _provider, _exporter, _log_provider, _log_exporter, _initialized
    
    if _exporter is not None:
        _exporter.clear()
    if _log_exporter is not None:
        _log_exporter.clear()
    
    _provider = None
    _exporter = None
    _log_provider = None
    _log_exporter = None
    _initialized = False


def is_initialized() -> bool:
    """Check if the tracer has been initialized.
    
    Returns:
        True if init_tracer() has been called, False otherwise.
    """
    return _initialized

