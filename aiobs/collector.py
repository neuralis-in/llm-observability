from __future__ import annotations

import contextvars
import json
import logging
import os
import platform
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from .models import (
    Session as ObsSession,
    SessionMeta as ObsSessionMeta,
    Event as ObsEvent,
    FunctionEvent as ObsFunctionEvent,
    ObservedEvent,
    ObservedFunctionEvent,
    ObservabilityExport,
)

if TYPE_CHECKING:
    from .exporters.base import BaseExporter, ExportResult
    from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)

# Default shepherd server URL for usage tracking
SHEPHERD_SERVER_URL = "https://shepherd-api-48963996968.us-central1.run.app"

# Default flush server URL for trace storage
AIOBS_FLUSH_SERVER_URL = "https://aiobs-flush-server-48963996968.us-central1.run.app"

# Context variable to track current span for nested tracing (fallback for non-OTel spans)
_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_span_id", default=None
)

# Label validation constants
LABEL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
LABEL_VALUE_MAX_LENGTH = 256
LABEL_MAX_COUNT = 64
LABEL_RESERVED_PREFIX = "aiobs_"
LABEL_ENV_PREFIX = "AIOBS_LABEL_"

# SDK version for system labels
SDK_VERSION = "0.1.0"


def _validate_label_key(key: str) -> None:
    """Validate a label key format.
    
    Args:
        key: The label key to validate.
        
    Raises:
        ValueError: If the key is invalid.
    """
    if not isinstance(key, str):
        raise ValueError(f"Label key must be a string, got {type(key).__name__}")
    if key.startswith(LABEL_RESERVED_PREFIX):
        raise ValueError(f"Label key '{key}' uses reserved prefix '{LABEL_RESERVED_PREFIX}'")
    if not LABEL_KEY_PATTERN.match(key):
        raise ValueError(
            f"Label key '{key}' is invalid. Keys must match pattern ^[a-z][a-z0-9_]{{0,62}}$"
        )


def _validate_label_value(value: str, key: str = "") -> None:
    """Validate a label value.
    
    Args:
        value: The label value to validate.
        key: The associated key (for error messages).
        
    Raises:
        ValueError: If the value is invalid.
    """
    if not isinstance(value, str):
        raise ValueError(f"Label value for '{key}' must be a string, got {type(value).__name__}")
    if len(value) > LABEL_VALUE_MAX_LENGTH:
        raise ValueError(
            f"Label value for '{key}' exceeds maximum length of {LABEL_VALUE_MAX_LENGTH} characters"
        )


def _validate_labels(labels: Dict[str, str]) -> None:
    """Validate a dictionary of labels.
    
    Args:
        labels: The labels dictionary to validate.
        
    Raises:
        ValueError: If any label is invalid or count exceeds limit.
    """
    if not isinstance(labels, dict):
        raise ValueError(f"Labels must be a dictionary, got {type(labels).__name__}")
    if len(labels) > LABEL_MAX_COUNT:
        raise ValueError(f"Too many labels ({len(labels)}). Maximum allowed is {LABEL_MAX_COUNT}.")
    for key, value in labels.items():
        _validate_label_key(key)
        _validate_label_value(value, key)


def _get_env_labels() -> Dict[str, str]:
    """Get labels from environment variables.
    
    Looks for variables prefixed with AIOBS_LABEL_ and converts them to labels.
    E.g., AIOBS_LABEL_ENVIRONMENT=production -> {"environment": "production"}
    
    Returns:
        Dictionary of labels from environment variables.
    """
    labels = {}
    for key, value in os.environ.items():
        if key.startswith(LABEL_ENV_PREFIX):
            label_key = key[len(LABEL_ENV_PREFIX):].lower()
            if label_key and LABEL_KEY_PATTERN.match(label_key):
                labels[label_key] = value[:LABEL_VALUE_MAX_LENGTH]
    return labels


def _get_system_labels() -> Dict[str, str]:
    """Get system-generated labels.
    
    Returns:
        Dictionary of system labels (prefixed with aiobs_).
    """
    import socket
    
    return {
        "aiobs_sdk_version": SDK_VERSION,
        "aiobs_python_version": platform.python_version(),
        "aiobs_hostname": socket.gethostname()[:LABEL_VALUE_MAX_LENGTH],
        "aiobs_os": platform.system().lower(),
    }


class Collector:
    """Simple, global-style collector with OpenTelemetry-based instrumentation.

    API:
      - observe(): enable instrumentation and start a session
      - end(): finish current session
      - flush(): write captured data to JSON (default: ./<session-id>.json)
      
    This collector uses OpenTelemetry underneath for trace context propagation
    and provider instrumentation, while maintaining the same output format.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ObsSession] = {}
        self._events: Dict[str, List[ObsEvent]] = {}
        self._active_session: Optional[str] = None
        self._lock = threading.RLock()
        self._instrumented = False
        self._unpatchers: List[Callable[[], None]] = []
        self._api_key: Optional[str] = None

    # Public API
    def observe(
        self,
        session_name: Optional[str] = None,
        api_key: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """Enable instrumentation (once) and start a new session.

        Args:
            session_name: Optional name for the session.
            api_key: API key (aiobs_sk_...) for usage tracking with shepherd-server.
                     Can also be set via AIOBS_API_KEY environment variable.
            labels: Optional dictionary of key-value labels for filtering and
                    categorization. Keys must be lowercase alphanumeric with
                    underscores (matching ^[a-z][a-z0-9_]{0,62}$). Values are
                    UTF-8 strings (max 256 chars). Labels from AIOBS_LABEL_*
                    environment variables are automatically merged.

        Returns a session id.

        Raises:
            ValueError: If no API key is provided, the API key is invalid,
                        or labels contain invalid keys/values.
            RuntimeError: If unable to connect to shepherd server.
        """
        with self._lock:
            # Store API key (parameter takes precedence over env var)
            self._api_key = api_key or os.getenv("AIOBS_API_KEY")

            if not self._api_key:
                raise ValueError(
                    "API key is required. Provide api_key parameter or set AIOBS_API_KEY environment variable."
                )

            # Validate API key with shepherd server
            self._validate_api_key()

            if not self._instrumented:
                self._instrumented = True
                self._install_instrumentation()

            # Build merged labels: system < env vars < explicit
            merged_labels: Dict[str, str] = {}
            merged_labels.update(_get_system_labels())
            merged_labels.update(_get_env_labels())
            if labels:
                _validate_labels(labels)
                merged_labels.update(labels)

            session_id = str(uuid.uuid4())
            now = _now()
            self._sessions[session_id] = ObsSession(
                id=session_id,
                name=session_name or session_id,
                started_at=now,
                ended_at=None,
                meta=ObsSessionMeta(pid=os.getpid(), cwd=os.getcwd()),
                labels=merged_labels if merged_labels else None,
            )
            self._events[session_id] = []
            self._active_session = session_id
            return session_id

    def end(self) -> None:
        with self._lock:
            if not self._active_session:
                return
            sess = self._sessions[self._active_session]
            self._sessions[self._active_session] = sess.model_copy(update={"ended_at": _now()})
            self._active_session = None

    def flush(
        self,
        path: Optional[str] = None,
        include_trace_tree: bool = True,
        exporter: Optional["BaseExporter"] = None,
        persist: bool = True,
        **exporter_kwargs: Any,
    ) -> Union[str, "ExportResult", None]:
        """Flush all sessions and events to a file or custom exporter.

        Args:
            path: Output file path. Defaults to LLM_OBS_OUT env var or '<session-id>.json'.
                  Ignored if exporter is provided or persist is False.
            include_trace_tree: Whether to include the nested trace_tree structure. Defaults to True.
            exporter: Optional exporter instance (e.g., GCSExporter, CustomExporter).
                      If provided, data is exported using this exporter instead of writing to a local file.
            persist: If True, dump observations to <session-id>.json file. If False, skip JSON file creation.
                     Defaults to True. Ignored if exporter is provided.
            **exporter_kwargs: Additional keyword arguments passed to the exporter's export() method.

        Returns:
            If exporter is provided: ExportResult from the exporter.
            If persist is True: The output file path used.
            If persist is False: None.
        """
        with self._lock:
            # Collect OTel spans and convert to events
            self._collect_otel_spans()
            
            # Separate standard events from function events
            standard_events = []
            function_events = []
            for sid, evs in self._events.items():
                for ev in evs:
                    if isinstance(ev, ObsFunctionEvent):
                        function_events.append(
                            ObservedFunctionEvent(session_id=sid, **ev.model_dump())
                        )
                    else:
                        standard_events.append(
                            ObservedEvent(session_id=sid, **ev.model_dump())
                        )

            # Count total traces for usage tracking (events + function_events)
            trace_count = len(standard_events) + len(function_events)

            # Build trace tree from all events (if enabled)
            all_events_for_tree = standard_events + function_events
            trace_tree = _build_trace_tree(all_events_for_tree) if include_trace_tree else []

            # Build enh_prompt_traces by extracting nodes with enh_prompt=True
            enh_prompt_traces = _extract_enh_prompt_traces(trace_tree) if include_trace_tree else None

            # Build a single JSON payload via pydantic models
            export = ObservabilityExport(
                sessions=list(self._sessions.values()),
                events=standard_events,
                function_events=function_events,
                trace_tree=trace_tree if include_trace_tree else None,
                enh_prompt_traces=enh_prompt_traces if enh_prompt_traces else None,
                generated_at=_now(),
            )

            # Use exporter if provided
            if exporter is not None:
                result = exporter.export(export, **exporter_kwargs)
                # Flush traces to remote server
                if self._api_key:
                    self._flush_to_server(export)
                # Record usage if API key is configured
                if self._api_key and trace_count > 0:
                    self._record_usage(trace_count)
                # Clear in-memory store after successful export
                self._sessions.clear()
                self._events.clear()
                self._active_session = None
                return result

            # Default: write to local file (only if persist=True)
            out_path = None
            if persist:
                # Determine default filename based on session ID
                default_filename = "llm_observability.json"
                if self._active_session:
                    default_filename = f"{self._active_session}.json"
                elif self._sessions:
                    # Use the first session ID if no active session
                    default_filename = f"{next(iter(self._sessions.keys()))}.json"
                
                out_path = path or os.getenv("LLM_OBS_OUT", default_filename)
                # Ensure directory exists if a nested path
                out_dir = os.path.dirname(out_path)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)

                # Write/overwrite JSON file
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(export.model_dump(), f, ensure_ascii=False, indent=2)

            # Flush traces to remote server
            if self._api_key:
                self._flush_to_server(export)

            # Record usage if API key is configured
            if self._api_key and trace_count > 0:
                self._record_usage(trace_count)

            # Optionally clear in-memory store after flush
            self._sessions.clear()
            self._events.clear()
            self._active_session = None
            return out_path

    def set_labels(
        self,
        labels: Dict[str, str],
        merge: bool = True,
    ) -> None:
        """Set or update labels for the current session.

        Args:
            labels: Dictionary of labels to set.
            merge: If True, merge with existing labels. If False, replace all
                   user labels (system labels are preserved).

        Raises:
            RuntimeError: If no active session.
            ValueError: If labels contain invalid keys or values.
        """
        with self._lock:
            if not self._active_session:
                raise RuntimeError("No active session. Call observe() first.")

            _validate_labels(labels)

            session = self._sessions[self._active_session]
            current_labels = dict(session.labels) if session.labels else {}

            if merge:
                current_labels.update(labels)
            else:
                # Preserve system labels, replace user labels
                system_labels = {k: v for k, v in current_labels.items() if k.startswith(LABEL_RESERVED_PREFIX)}
                system_labels.update(labels)
                current_labels = system_labels

            # Check total count after merge
            if len(current_labels) > LABEL_MAX_COUNT:
                raise ValueError(
                    f"Too many labels ({len(current_labels)}). Maximum allowed is {LABEL_MAX_COUNT}."
                )

            self._sessions[self._active_session] = session.model_copy(
                update={"labels": current_labels}
            )

    def add_label(self, key: str, value: str) -> None:
        """Add a single label to the current session.

        Args:
            key: Label key (lowercase alphanumeric with underscores).
            value: Label value (UTF-8 string, max 256 chars).

        Raises:
            RuntimeError: If no active session.
            ValueError: If key or value is invalid.
        """
        with self._lock:
            if not self._active_session:
                raise RuntimeError("No active session. Call observe() first.")

            _validate_label_key(key)
            _validate_label_value(value, key)

            session = self._sessions[self._active_session]
            current_labels = dict(session.labels) if session.labels else {}

            # Check if adding would exceed limit
            if key not in current_labels and len(current_labels) >= LABEL_MAX_COUNT:
                raise ValueError(
                    f"Cannot add label. Maximum of {LABEL_MAX_COUNT} labels already reached."
                )

            current_labels[key] = value
            self._sessions[self._active_session] = session.model_copy(
                update={"labels": current_labels}
            )

    def remove_label(self, key: str) -> None:
        """Remove a label from the current session.

        Args:
            key: Label key to remove.

        Raises:
            RuntimeError: If no active session.
            ValueError: If trying to remove a system label.
        """
        with self._lock:
            if not self._active_session:
                raise RuntimeError("No active session. Call observe() first.")

            if key.startswith(LABEL_RESERVED_PREFIX):
                raise ValueError(f"Cannot remove system label '{key}'")

            session = self._sessions[self._active_session]
            if session.labels and key in session.labels:
                current_labels = dict(session.labels)
                del current_labels[key]
                self._sessions[self._active_session] = session.model_copy(
                    update={"labels": current_labels if current_labels else None}
                )

    def get_labels(self) -> Dict[str, str]:
        """Get all labels for the current session.

        Returns:
            Dictionary of current labels (empty dict if none).

        Raises:
            RuntimeError: If no active session.
        """
        with self._lock:
            if not self._active_session:
                raise RuntimeError("No active session. Call observe() first.")

            session = self._sessions[self._active_session]
            return dict(session.labels) if session.labels else {}

    # Internal API
    def _install_instrumentation(self) -> None:
        """Install OpenTelemetry tracer and provider instrumentors."""
        # Initialize OTel tracer
        from .tracer import init_tracer
        init_tracer()
        
        # Install OpenAI instrumentor (if available)
        try:
            from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
            instrumentor = OpenAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                self._unpatchers.append(lambda: OpenAIInstrumentor().uninstrument())
                logger.debug("OpenAI OTel instrumentation installed")
        except ImportError:
            logger.debug("OpenAI OTel instrumentation not available")
        except Exception as e:
            logger.debug(f"Failed to install OpenAI OTel instrumentation: {e}")
        
        # Install Google GenAI instrumentor (for google-genai SDK)
        try:
            from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
            instrumentor = GoogleGenAiSdkInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                self._unpatchers.append(lambda: GoogleGenAiSdkInstrumentor().uninstrument())
                logger.debug("Google GenAI OTel instrumentation installed")
        except ImportError:
            logger.debug("Google GenAI OTel instrumentation not available")
        except Exception as e:
            logger.debug(f"Failed to install Google GenAI OTel instrumentation: {e}")
        
        # Install Vertex AI instrumentor (for langchain/vertex)
        try:
            from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor
            instrumentor = VertexAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                self._unpatchers.append(lambda: VertexAIInstrumentor().uninstrument())
                logger.debug("Vertex AI OTel instrumentation installed")
        except ImportError:
            logger.debug("Vertex AI OTel instrumentation not available")
        except Exception as e:
            logger.debug(f"Failed to install Vertex AI OTel instrumentation: {e}")

    def reset(self) -> None:
        """Reset collector state and unpatch providers (for tests/dev)."""
        with self._lock:
            # End session and clear data
            self._active_session = None
            self._sessions.clear()
            self._events.clear()
            self._api_key = None

            # Unpatch instrumentors
            for up in reversed(self._unpatchers):
                try:
                    up()
                except Exception:
                    pass
            self._unpatchers.clear()
            self._instrumented = False
            
            # Reset OTel tracer
            from .tracer import reset_tracer
            reset_tracer()

    def _collect_otel_spans(self) -> None:
        """Collect finished OTel spans and logs, converting them to aiobs events.
        
        Note: Spans from the aiobs tracer (created by @observe decorator) are
        filtered out to avoid duplicates, since those are already recorded
        directly as FunctionEvents.
        
        Logs are associated with spans via their span context and used to
        extract message content (prompts and completions).
        """
        from .tracer import get_finished_spans, clear_spans, get_finished_logs, clear_logs
        
        # Get the session to add events to (prefer active, fall back to first available)
        session_id = self._active_session
        if not session_id and self._sessions:
            session_id = next(iter(self._sessions.keys()))
        
        if not session_id:
            # No sessions at all, clear spans/logs and return
            clear_spans()
            clear_logs()
            return
        
        # Ensure events list exists for this session
        if session_id not in self._events:
            self._events[session_id] = []
        
        # Collect logs and group by span_id for message content extraction
        logs = get_finished_logs()
        logs_by_span: Dict[str, List] = {}
        for log in logs:
            if log.log_record and log.log_record.span_id:
                span_id = format(log.log_record.span_id, '016x')
                if span_id not in logs_by_span:
                    logs_by_span[span_id] = []
                logs_by_span[span_id].append(log)
            
        spans = get_finished_spans()
        for span in spans:
            # Skip spans from aiobs tracer (these are @observe decorated functions,
            # which are already recorded as FunctionEvents)
            if span.instrumentation_scope and span.instrumentation_scope.name == "aiobs":
                continue
            
            # Get logs associated with this span
            span_ctx = span.get_span_context()
            span_id = format(span_ctx.span_id, '016x') if span_ctx else None
            span_logs = logs_by_span.get(span_id, []) if span_id else []
            
            event = self._convert_otel_span_to_event(span, span_logs)
            if event:
                self._events[session_id].append(event)
        
        # Clear spans and logs after collecting
        clear_spans()
        clear_logs()

    def _convert_otel_span_to_event(self, span: "ReadableSpan", logs: Optional[List] = None) -> Optional[ObsEvent]:
        """Convert an OTel span to aiobs Event model.
        
        Extracts data from OTel GenAI semantic conventions to match our output format.
        Requires OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true for full message capture.
        
        Args:
            span: An OpenTelemetry ReadableSpan object.
            logs: Optional list of LogData objects associated with this span.
            
        Returns:
            An ObsEvent if conversion is successful, None otherwise.
        """
        try:
            attrs = dict(span.attributes) if span.attributes else {}
            span_name = span.name
            
            # Determine provider from attributes (GenAI semantic conventions)
            provider = str(attrs.get("gen_ai.system", ""))
            
            # Normalize provider names
            if provider in ("vertex_ai", "google_genai", "google"):
                provider = "gemini"
            elif not provider:
                # Try to infer from span name or other attributes
                if "openai" in span_name.lower() or attrs.get("server.address", "").endswith("openai.com"):
                    provider = "openai"
                elif "gemini" in span_name.lower() or "google" in span_name.lower() or "vertex" in span_name.lower():
                    provider = "gemini"
                else:
                    provider = "unknown"
            
            # Build full API name (e.g., "chat.completions.create" for OpenAI)
            operation = str(attrs.get("gen_ai.operation.name", ""))
            if provider == "openai":
                if operation == "chat":
                    api_name = "chat.completions.create"
                elif operation == "embeddings":
                    api_name = "embeddings.create"
                else:
                    api_name = f"{operation}.create" if operation else span_name
            elif provider in ("gemini", "google", "google_genai"):
                # Gemini uses "generate_content" as the operation
                if operation == "generate_content" or "generate" in span_name.lower():
                    api_name = "models.generate_content"
                else:
                    api_name = operation or span_name
            else:
                api_name = operation or span_name
            
            # Extract timing (OTel uses nanoseconds)
            started_at = span.start_time / 1e9 if span.start_time else _now()
            ended_at = span.end_time / 1e9 if span.end_time else _now()
            duration_ms = (span.end_time - span.start_time) / 1e6 if span.start_time and span.end_time else 0.0
            
            # Extract span IDs
            span_ctx = span.get_span_context()
            span_id = format(span_ctx.span_id, '016x') if span_ctx else None
            trace_id = format(span_ctx.trace_id, '032x') if span_ctx else None
            
            parent_span_id = None
            if span.parent:
                parent_span_id = format(span.parent.span_id, '016x')
            
            # Build request object
            request_model = attrs.get("gen_ai.request.model")
            request: Dict[str, Any] = {
                "model": request_model,
            }
            
            # Add request parameters if available
            if attrs.get("gen_ai.request.max_tokens"):
                request["max_tokens"] = attrs.get("gen_ai.request.max_tokens")
            if attrs.get("gen_ai.request.temperature") is not None:
                request["temperature"] = attrs.get("gen_ai.request.temperature")
            if attrs.get("gen_ai.request.top_p") is not None:
                request["top_p"] = attrs.get("gen_ai.request.top_p")
            if attrs.get("gen_ai.request.frequency_penalty") is not None:
                request["frequency_penalty"] = attrs.get("gen_ai.request.frequency_penalty")
            if attrs.get("gen_ai.request.presence_penalty") is not None:
                request["presence_penalty"] = attrs.get("gen_ai.request.presence_penalty")
            
            # Collect other request attributes
            other_attrs = {}
            for key, value in attrs.items():
                if key.startswith("gen_ai.request.") and key not in [
                    "gen_ai.request.model", "gen_ai.request.max_tokens",
                    "gen_ai.request.temperature", "gen_ai.request.top_p",
                    "gen_ai.request.frequency_penalty", "gen_ai.request.presence_penalty"
                ]:
                    other_attrs[key.replace("gen_ai.request.", "")] = value
            if other_attrs:
                request["other"] = other_attrs
            
            # Extract messages from logs (GenAI semantic conventions use logs for message content)
            messages = []
            completion_text = None
            
            if logs:
                for log_data in logs:
                    log_record = log_data.log_record
                    if not log_record or not log_record.body:
                        continue
                    
                    body = log_record.body
                    
                    # Parse body - can be dict or JSON string
                    if isinstance(body, str):
                        try:
                            import json
                            body_data = json.loads(body)
                        except json.JSONDecodeError:
                            continue
                    elif isinstance(body, dict):
                        body_data = body
                    else:
                        continue
                    
                    # Get event name to determine message type
                    # Format: gen_ai.{role}.message or gen_ai.choice
                    event_name = getattr(log_record, 'event_name', '') or ''
                    
                    # Completion/choice logs - handle both OpenAI and Gemini formats
                    if event_name == "gen_ai.choice":
                        content = None
                        
                        # OpenAI format: message.content
                        if "message" in body_data:
                            msg = body_data.get("message", {})
                            if isinstance(msg, dict):
                                content = msg.get("content", "")
                        
                        # Gemini format: content.parts[].text (nested structure)
                        elif "content" in body_data:
                            content_obj = body_data.get("content", {})
                            if isinstance(content_obj, dict) and "parts" in content_obj:
                                parts = content_obj.get("parts", [])
                                text_parts = []
                                for part in parts:
                                    if isinstance(part, dict):
                                        # Try various field names
                                        text = part.get("text") or part.get("content") or ""
                                        if text:
                                            text_parts.append(str(text))
                                    elif hasattr(part, "text"):
                                        text_parts.append(str(part.text))
                                    elif hasattr(part, "content"):
                                        text_parts.append(str(part.content))
                                if text_parts:
                                    content = "".join(text_parts)
                            elif isinstance(content_obj, str):
                                content = content_obj
                        
                        # Fallback: parts directly in body
                        elif "parts" in body_data:
                            parts = body_data.get("parts", [])
                            text_parts = []
                            for part in parts:
                                if isinstance(part, dict):
                                    text = part.get("text") or part.get("content") or ""
                                    if text:
                                        text_parts.append(str(text))
                                elif hasattr(part, "text"):
                                    text_parts.append(str(part.text))
                                elif hasattr(part, "content"):
                                    text_parts.append(str(part.content))
                            if text_parts:
                                content = "".join(text_parts)
                        
                        if content and not completion_text:
                            completion_text = str(content)
                    
                    # Prompt logs: gen_ai.{role}.message format (Gemini uses this)
                    elif event_name.startswith("gen_ai.") and event_name.endswith(".message"):
                        # Extract role from event name (e.g., "gen_ai.system.message" -> "system")
                        role = event_name.replace("gen_ai.", "").replace(".message", "")
                        content = None
                        
                        # Direct content field
                        if "content" in body_data:
                            content = body_data.get("content", "")
                        
                        # Gemini format: parts[].content
                        elif "parts" in body_data:
                            parts = body_data.get("parts", [])
                            text_parts = []
                            for part in parts:
                                if isinstance(part, dict) and "content" in part:
                                    text_parts.append(str(part["content"]))
                                elif isinstance(part, dict) and "text" in part:
                                    text_parts.append(str(part["text"]))
                                elif hasattr(part, "content"):
                                    text_parts.append(str(part.content))
                            if text_parts:
                                content = "".join(text_parts)
                        
                        if content:
                            messages.append({"role": str(role), "content": str(content)})
                    
                    # Fallback: prompt logs with 'content' directly (OpenAI format)
                    elif "content" in body_data and "index" not in body_data:
                        role = body_data.get("role", "user")
                        content = body_data.get("content", "")
                        if content:
                            messages.append({"role": str(role), "content": str(content)})
            
            # Fallback: try span events if no logs provided message data
            if not messages:
                for event in span.events or []:
                    event_attrs = dict(event.attributes) if event.attributes else {}
                    
                    if event.name == "gen_ai.content.prompt":
                        role = event_attrs.get("gen_ai.prompt.role", "user")
                        content = event_attrs.get("gen_ai.prompt.content") or event_attrs.get("gen_ai.prompt", "")
                        if content:
                            messages.append({"role": str(role), "content": str(content)})
                    elif event.name.startswith("gen_ai.") and "prompt" in event.name.lower():
                        content = event_attrs.get("content") or event_attrs.get("gen_ai.prompt", "")
                        role = event_attrs.get("role", "user")
                        if content:
                            messages.append({"role": str(role), "content": str(content)})
            
            if not completion_text:
                for event in span.events or []:
                    event_attrs = dict(event.attributes) if event.attributes else {}
                    
                    if event.name == "gen_ai.content.completion":
                        completion_text = event_attrs.get("gen_ai.completion.content") or event_attrs.get("gen_ai.completion", "")
                        break
                    elif event.name.startswith("gen_ai.") and "completion" in event.name.lower():
                        completion_text = event_attrs.get("content") or event_attrs.get("gen_ai.completion", "")
                        break
            
            if messages:
                request["messages"] = messages
            
            # Build response object
            response: Dict[str, Any] = {}
            
            # Response ID
            response_id = attrs.get("gen_ai.response.id")
            if response_id:
                response["id"] = response_id
            
            # Response model
            response_model = attrs.get("gen_ai.response.model", request_model)
            if response_model:
                response["model"] = response_model
            
            if completion_text:
                response["text"] = str(completion_text)
            
            # Finish reasons
            finish_reasons = attrs.get("gen_ai.response.finish_reasons")
            if finish_reasons:
                if isinstance(finish_reasons, (list, tuple)):
                    response["finish_reason"] = finish_reasons[0] if len(finish_reasons) == 1 else list(finish_reasons)
                else:
                    response["finish_reason"] = str(finish_reasons)
            
            # Extract usage metrics
            usage: Dict[str, Any] = {}
            input_tokens = attrs.get("gen_ai.usage.input_tokens") or attrs.get("gen_ai.usage.prompt_tokens")
            output_tokens = attrs.get("gen_ai.usage.output_tokens") or attrs.get("gen_ai.usage.completion_tokens")
            total_tokens = attrs.get("gen_ai.usage.total_tokens")
            
            if input_tokens is not None:
                usage["prompt_tokens"] = int(input_tokens)
            if output_tokens is not None:
                usage["completion_tokens"] = int(output_tokens)
            if total_tokens is not None:
                usage["total_tokens"] = int(total_tokens)
            elif input_tokens is not None and output_tokens is not None:
                usage["total_tokens"] = int(input_tokens) + int(output_tokens)
            
            # Add detailed token info if available
            prompt_tokens_details = {}
            completion_tokens_details = {}
            for key, value in attrs.items():
                if "prompt_tokens" in key and key != "gen_ai.usage.prompt_tokens":
                    detail_key = key.split(".")[-1]
                    prompt_tokens_details[detail_key] = value
                elif "completion_tokens" in key and key != "gen_ai.usage.completion_tokens":
                    detail_key = key.split(".")[-1]
                    completion_tokens_details[detail_key] = value
            
            if prompt_tokens_details:
                usage["prompt_tokens_details"] = prompt_tokens_details
            if completion_tokens_details:
                usage["completion_tokens_details"] = completion_tokens_details
                
            if usage:
                response["usage"] = usage
            
            # Check for errors
            error = None
            if span.status and span.status.status_code:
                from opentelemetry.trace import StatusCode
                if span.status.status_code == StatusCode.ERROR:
                    error = span.status.description or "Unknown error"
            
            # Also check for error attributes
            if not error:
                error_type = attrs.get("error.type")
                if error_type:
                    error_msg = attrs.get("error.message", "")
                    error = f"{error_type}: {error_msg}" if error_msg else str(error_type)
            
            return ObsEvent(
                provider=provider,
                api=api_name,
                request=request if request.get("model") or request.get("messages") else None,
                response=response if response else None,
                error=error,
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=round(duration_ms, 3),
                span_id=span_id,
                parent_span_id=parent_span_id,
                trace_id=trace_id,
            )
        except Exception as e:
            logger.debug(f"Failed to convert OTel span to event: {e}")
            return None

    def _validate_api_key(self) -> None:
        """Validate the API key with shepherd server.

        Raises:
            ValueError: If the API key is invalid.
            RuntimeError: If unable to connect to shepherd server.
        """
        if not self._api_key:
            return

        import urllib.request
        import urllib.error

        url = f"{SHEPHERD_SERVER_URL}/v1/usage"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                if result.get("success"):
                    usage = result.get("usage", {})
                    logger.debug(
                        "API key validated: tier=%s, traces_used=%d/%d",
                        usage.get("tier", "unknown"),
                        usage.get("traces_used", 0),
                        usage.get("traces_limit", 0),
                    )
                    if usage.get("is_rate_limited"):
                        raise RuntimeError(
                            f"Rate limit exceeded: tier={usage.get('tier')}, "
                            f"used={usage.get('traces_used')}/{usage.get('traces_limit')}"
                        )

        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise ValueError("Invalid API key provided to aiobs")
            else:
                raise RuntimeError(f"Failed to validate API key: HTTP {e.code}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to shepherd server: {e.reason}")

    def _record_usage(self, trace_count: int) -> None:
        """Record usage to shepherd-server.

        Args:
            trace_count: Number of traces to record.

        Raises:
            ValueError: If the API key is invalid.
            RuntimeError: If rate limit is exceeded or server error occurs.
        """
        if not self._api_key:
            return

        import urllib.request
        import urllib.error

        url = f"{SHEPHERD_SERVER_URL}/v1/usage"
        data = json.dumps({"trace_count": trace_count}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                if result.get("success"):
                    logger.debug(
                        "Usage recorded: %d traces, %d remaining",
                        trace_count,
                        result.get("usage", {}).get("traces_remaining", "unknown"),
                    )

        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise ValueError("Invalid API key provided to aiobs")
            elif e.code == 429:
                try:
                    error_body = json.loads(e.read().decode("utf-8"))
                    raise RuntimeError(
                        f"Rate limit exceeded: {error_body.get('error', 'Unknown error')} "
                        f"(tier: {error_body.get('usage', {}).get('tier', 'unknown')}, "
                        f"used: {error_body.get('usage', {}).get('traces_used', 0)}/"
                        f"{error_body.get('usage', {}).get('traces_limit', 0)})"
                    )
                except RuntimeError:
                    raise
                except Exception:
                    raise RuntimeError("Rate limit exceeded for API key")
            else:
                raise RuntimeError(f"Failed to record usage: HTTP {e.code}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to shepherd server: {e.reason}")

    def _flush_to_server(self, export: ObservabilityExport) -> None:
        """Send trace data to the flush server.

        Args:
            export: The ObservabilityExport payload to send.

        Raises:
            ValueError: If the API key is invalid.
            RuntimeError: If server error occurs.
        """
        if not self._api_key:
            return

        import urllib.request
        import urllib.error

        flush_server_url = os.getenv("AIOBS_FLUSH_SERVER_URL", AIOBS_FLUSH_SERVER_URL)
        url = f"{flush_server_url}/v1/traces"
        data = json.dumps(export.model_dump()).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                logger.debug(
                    "Traces flushed to server: %s",
                    result.get("message", "success"),
                )

        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise ValueError("Invalid API key provided to aiobs")
            else:
                logger.warning(f"Failed to flush traces to server: HTTP {e.code}")
        except urllib.error.URLError as e:
            logger.warning(f"Failed to connect to flush server: {e.reason}")

    def _record_event(self, payload: Any) -> None:
        with self._lock:
            sid = self._active_session
            if not sid:
                return
            if isinstance(payload, (ObsEvent, ObsFunctionEvent)):
                ev = payload
            else:
                try:
                    # Try to detect if this is a function event
                    if payload.get("provider") == "function" and "name" in payload:
                        ev = ObsFunctionEvent(**payload)
                    else:
                        ev = ObsEvent(**payload)
                except Exception:
                    # Best-effort fallback for unexpected shapes
                    ev = ObsEvent(
                        provider=str(payload.get("provider")),
                        api=str(payload.get("api")),
                        request=payload.get("request"),
                        response=payload.get("response"),
                        error=payload.get("error"),
                        started_at=float(payload.get("started_at", _now())),
                        ended_at=float(payload.get("ended_at", _now())),
                        duration_ms=float(payload.get("duration_ms", 0.0)),
                        callsite=payload.get("callsite"),
                    )
            self._events[sid].append(ev)

    # Span context management for nested tracing (OTel-based)
    def get_current_span_id(self) -> Optional[str]:
        """Get the current span ID from OTel context (for parent-child linking)."""
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span and span.is_recording():
                span_ctx = span.get_span_context()
                if span_ctx and span_ctx.is_valid:
                    return format(span_ctx.span_id, '016x')
        except Exception:
            pass
        # Fallback to legacy context var
        return _current_span_id.get()

    def set_current_span_id(self, span_id: Optional[str]) -> contextvars.Token[Optional[str]]:
        """Set the current span ID in context. Returns a token to restore previous value.
        
        Note: This is primarily for backward compatibility. OTel context is managed
        automatically when using tracer.start_as_current_span().
        """
        return _current_span_id.set(span_id)

    def reset_span_id(self, token: contextvars.Token[Optional[str]]) -> None:
        """Reset the span ID to its previous value using the token."""
        _current_span_id.reset(token)


def _now() -> float:
    return time.time()


def _build_trace_tree(events: List[Union[ObservedEvent, ObservedFunctionEvent]]) -> List[Dict[str, Any]]:
    """Build a nested tree structure from flat events using span_id/parent_span_id.
    
    Includes both standard events (provider API calls) and function events (@observe decorated).
    """
    if not events:
        return []

    # Create lookup by span_id
    events_by_span: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        span_id = ev.span_id
        if span_id:
            node = ev.model_dump()
            node["children"] = []
            # Add event_type marker for easier identification
            node["event_type"] = "function" if isinstance(ev, ObservedFunctionEvent) else "provider"
            events_by_span[span_id] = node

    # Build tree by linking children to parents
    roots: List[Dict[str, Any]] = []
    for ev in events:
        span_id = ev.span_id
        parent_id = ev.parent_span_id
        
        node_data = ev.model_dump()
        node_data["event_type"] = "function" if isinstance(ev, ObservedFunctionEvent) else "provider"
        
        if not span_id:
            # Events without span_id: check if they have a parent
            if parent_id and parent_id in events_by_span:
                # Add as child of parent (even without own span_id)
                if "children" not in node_data:
                    node_data["children"] = []
                events_by_span[parent_id]["children"].append(node_data)
            else:
                # No parent or parent not found -> root
                if "children" not in node_data:
                    node_data["children"] = []
                roots.append(node_data)
            continue

        node = events_by_span[span_id]
        if parent_id and parent_id in events_by_span:
            # Add as child of parent
            events_by_span[parent_id]["children"].append(node)
        else:
            # No parent or parent not found -> root
            roots.append(node)

    # Sort roots and children by started_at for consistent ordering
    def sort_by_time(nodes: List[Dict[str, Any]]) -> None:
        nodes.sort(key=lambda n: n.get("started_at", 0))
        for node in nodes:
            if node.get("children"):
                sort_by_time(node["children"])

    sort_by_time(roots)
    return roots


def _extract_enh_prompt_traces(trace_tree: List[Dict[str, Any]]) -> List[str]:
    """Extract enh_prompt_id values from the trace tree.
    
    Returns a list of enh_prompt_id values for nodes with enh_prompt=True.
    """
    result: List[str] = []
    
    def walk(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            # Check if this node has enh_prompt=True and has an enh_prompt_id
            if node.get("enh_prompt") is True:
                enh_prompt_id = node.get("enh_prompt_id")
                if enh_prompt_id:
                    result.append(enh_prompt_id)
            # Recurse into children
            children = node.get("children", [])
            if children:
                walk(children)
    
    walk(trace_tree)
    return result
