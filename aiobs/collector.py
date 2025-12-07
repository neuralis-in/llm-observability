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

logger = logging.getLogger(__name__)

# Default shepherd server URL for usage tracking
SHEPHERD_SERVER_URL = "https://shepherd-api-48963996968.us-central1.run.app"

# Default flush server URL for trace storage
AIOBS_FLUSH_SERVER_URL = "https://aiobs-flush-server-48963996968.us-central1.run.app"

# Context variable to track current span for nested tracing
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
    """Simple, global-style collector with pluggable provider instrumentation.

    API:
      - observe(): enable instrumentation and start a session
      - end(): finish current session
      - flush(): write captured data to JSON (default: ./<session-id>.json)
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ObsSession] = {}
        self._events: Dict[str, List[ObsEvent]] = {}
        self._active_session: Optional[str] = None
        self._lock = threading.RLock()
        self._instrumented = False
        self._unpatchers: List[Callable[[], None]] = []
        self._providers: List[Any] = []  # instances of BaseProvider
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
        # If no providers explicitly registered, attempt to include built-ins
        if not self._providers:
            try:
                from .providers.openai import OpenAIProvider  # lazy import

                if OpenAIProvider.is_available():
                    self._providers.append(OpenAIProvider())
            except Exception:
                pass

            try:
                from .providers.gemini import GeminiProvider  # lazy import

                if GeminiProvider.is_available():
                    self._providers.append(GeminiProvider())
            except Exception:
                pass

        # Install each provider's instrumentation
        for provider in list(self._providers):
            try:
                unpatch = provider.install(self)
                if unpatch:
                    self._unpatchers.append(unpatch)
            except Exception:
                # Non-fatal
                continue

    # Optional: allow external registration of providers
    def register_provider(self, provider: Any) -> None:
        with self._lock:
            self._providers.append(provider)

    def reset(self) -> None:
        """Reset collector state and unpatch providers (for tests/dev)."""
        with self._lock:
            # End session and clear data
            self._active_session = None
            self._sessions.clear()
            self._events.clear()
            self._api_key = None

            # Unpatch providers
            for up in reversed(self._unpatchers):
                try:
                    up()
                except Exception:
                    pass
            self._unpatchers.clear()
            self._providers.clear()
            self._instrumented = False

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

    # Span context management for nested tracing
    def get_current_span_id(self) -> Optional[str]:
        """Get the current span ID from context (for parent-child linking)."""
        return _current_span_id.get()

    def set_current_span_id(self, span_id: Optional[str]) -> contextvars.Token[Optional[str]]:
        """Set the current span ID in context. Returns a token to restore previous value."""
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
