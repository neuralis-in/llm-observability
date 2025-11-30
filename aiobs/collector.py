from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Union, TYPE_CHECKING, TypeVar

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

F = TypeVar("F", bound=Callable[..., Any])

# Default shepherd server URL for usage tracking
SHEPHERD_SERVER_URL = "https://shepherd-api-48963996968.us-central1.run.app"

# Context variable to track current span for nested tracing
_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_span_id", default=None
)


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
        # Sessions detached for background task handling
        self._detached_sessions: Set[str] = set()

    # Public API
    def observe(
        self,
        session_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """Enable instrumentation (once) and start a new session.

        Args:
            session_name: Optional name for the session.
            api_key: API key (aiobs_sk_...) for usage tracking with shepherd-server.
                     Can also be set via AIOBS_API_KEY environment variable.

        Returns a session id.

        Raises:
            ValueError: If no API key is provided or the API key is invalid.
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

            session_id = str(uuid.uuid4())
            now = _now()
            self._sessions[session_id] = ObsSession(
                id=session_id,
                name=session_name or session_id,
                started_at=now,
                ended_at=None,
                meta=ObsSessionMeta(pid=os.getpid(), cwd=os.getcwd()),
            )
            self._events[session_id] = []
            self._active_session = session_id
            return session_id

    def end(self) -> None:
        with self._lock:
            if not self._active_session:
                return
            # Skip if session is detached (will be ended by background task)
            if self._active_session in self._detached_sessions:
                self._active_session = None
                return
            sess = self._sessions[self._active_session]
            self._sessions[self._active_session] = sess.model_copy(update={"ended_at": _now()})
            self._active_session = None

    def flush(
        self,
        path: Optional[str] = None,
        include_trace_tree: bool = True,
        exporter: Optional["BaseExporter"] = None,
        **exporter_kwargs: Any,
    ) -> Optional[Union[str, "ExportResult"]]:
        """Flush all sessions and events to a file or custom exporter.

        Args:
            path: Output file path. Defaults to LLM_OBS_OUT env var or '<session-id>.json'.
                  Ignored if exporter is provided.
            include_trace_tree: Whether to include the nested trace_tree structure. Defaults to True.
            exporter: Optional exporter instance (e.g., GCSExporter, CustomExporter).
                      If provided, data is exported using this exporter instead of writing to a local file.
            **exporter_kwargs: Additional keyword arguments passed to the exporter's export() method.

        Returns:
            If exporter is provided: ExportResult from the exporter.
            If local file: The output file path used.
            If all sessions are detached (handled by background tasks): None.
        """
        with self._lock:
            # Check if all sessions are detached (being handled by background tasks)
            if self._sessions and all(sid in self._detached_sessions for sid in self._sessions):
                # All sessions are detached, skip flush - background tasks will handle it
                return None
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
                # Record usage if API key is configured
                if self._api_key and trace_count > 0:
                    self._record_usage(trace_count)
                # Clear in-memory store after successful export
                self._sessions.clear()
                self._events.clear()
                self._active_session = None
                return result

            # Default: write to local file
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

            # Record usage if API key is configured
            if self._api_key and trace_count > 0:
                self._record_usage(trace_count)

            # Optionally clear in-memory store after flush
            self._sessions.clear()
            self._events.clear()
            self._active_session = None
            return out_path

    def background_task(
        self,
        func: F,
        exporter: Optional["BaseExporter"] = None,
        path: Optional[str] = None,
        include_trace_tree: bool = True,
        **flush_kwargs: Any,
    ) -> F:
        """Wrap a background task function to capture traces and auto-flush on completion.

        When this method is called, the current session is "detached" from the
        calling context. This means:
        1. Subsequent calls to end() and flush() in the calling context become no-ops
        2. The wrapped function automatically calls end() and flush() when it completes
        3. All traces from the background task are properly captured

        This is designed to work seamlessly with FastAPI's BackgroundTasks, asyncio tasks,
        or any deferred execution pattern.

        Args:
            func: The function to wrap (sync or async).
            exporter: Optional exporter instance for flush (e.g., GCSExporter).
            path: Optional output file path for flush (ignored if exporter is provided).
            include_trace_tree: Whether to include trace tree in flush output.
            **flush_kwargs: Additional arguments passed to flush().

        Returns:
            A wrapped function that will auto-flush on completion.

        Example:
            @app.post("/generate")
            async def generate(request: Request, background_tasks: BackgroundTasks):
                try:
                    observer.observe(session_name="my_session")

                    background_tasks.add_task(
                        observer.background_task(my_task_func, exporter=my_exporter),
                        request.data
                    )

                    return {"status": "started"}
                finally:
                    observer.end()   # No-op because session is detached
                    observer.flush() # No-op because session is detached
        """
        with self._lock:
            session_id = self._active_session
            api_key = self._api_key
            if session_id:
                # Mark this session as detached
                self._detached_sessions.add(session_id)

        def _do_flush(sid: str) -> None:
            """Perform the actual flush for a detached session."""
            with self._lock:
                # Remove from detached set
                self._detached_sessions.discard(sid)
                # Restore session as active for proper end/flush
                self._active_session = sid
                self._api_key = api_key

            # End and flush the session
            self.end()
            self.flush(
                path=path,
                include_trace_tree=include_trace_tree,
                exporter=exporter,
                **flush_kwargs,
            )

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Restore session context for the background task
                with self._lock:
                    self._active_session = session_id
                    self._api_key = api_key
                try:
                    return await func(*args, **kwargs)
                finally:
                    if session_id:
                        _do_flush(session_id)

            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Restore session context for the background task
                with self._lock:
                    self._active_session = session_id
                    self._api_key = api_key
                try:
                    return func(*args, **kwargs)
                finally:
                    if session_id:
                        _do_flush(session_id)

            return sync_wrapper  # type: ignore[return-value]

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
            self._detached_sessions.clear()

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
