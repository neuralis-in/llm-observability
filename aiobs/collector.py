from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .models import (
    Session as ObsSession,
    SessionMeta as ObsSessionMeta,
    Event as ObsEvent,
    FunctionEvent as ObsFunctionEvent,
    ObservedEvent,
    ObservedFunctionEvent,
    ObservabilityExport,
)

# Context variable to track current span for nested tracing
_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_span_id", default=None
)


class Collector:
    """Simple, global-style collector with pluggable provider instrumentation.

    API:
      - observe(): enable instrumentation and start a session
      - end(): finish current session
      - flush(): write captured data to JSON (default: ./llm_observability.json)
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ObsSession] = {}
        self._events: Dict[str, List[ObsEvent]] = {}
        self._active_session: Optional[str] = None
        self._lock = threading.RLock()
        self._instrumented = False
        self._unpatchers: List[Callable[[], None]] = []
        self._providers: List[Any] = []  # instances of BaseProvider

    # Public API
    def observe(self, session_name: Optional[str] = None) -> str:
        """Enable instrumentation (once) and start a new session.

        Returns a session id.
        """
        with self._lock:
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
            sess = self._sessions[self._active_session]
            self._sessions[self._active_session] = sess.model_copy(update={"ended_at": _now()})
            self._active_session = None

    def flush(self, path: Optional[str] = None, include_trace_tree: bool = True) -> str:
        """Flush all sessions and events to a single JSON file.

        Args:
            path: Output file path. Defaults to LLM_OBS_OUT env var or 'llm_observability.json'.
            include_trace_tree: Whether to include the nested trace_tree structure. Defaults to True.

        Returns the output path used.
        """
        with self._lock:
            out_path = path or os.getenv("LLM_OBS_OUT", "llm_observability.json")
            # Ensure directory exists if a nested path
            out_dir = os.path.dirname(out_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

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

            # Build trace tree from function events (if enabled)
            trace_tree = _build_trace_tree(function_events) if include_trace_tree else []

            # Build a single JSON payload via pydantic models
            export = ObservabilityExport(
                sessions=list(self._sessions.values()),
                events=standard_events,
                function_events=function_events,
                trace_tree=trace_tree if include_trace_tree else None,
                generated_at=_now(),
            )

            # Write/overwrite JSON file
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(export.model_dump(), f, ensure_ascii=False, indent=2)

            # Optionally clear in-memory store after flush
            self._sessions.clear()
            self._events.clear()
            self._active_session = None
            return out_path

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

            # Unpatch providers
            for up in reversed(self._unpatchers):
                try:
                    up()
                except Exception:
                    pass
            self._unpatchers.clear()
            self._providers.clear()
            self._instrumented = False

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


def _build_trace_tree(events: List[ObservedFunctionEvent]) -> List[Dict[str, Any]]:
    """Build a nested tree structure from flat events using span_id/parent_span_id."""
    if not events:
        return []

    # Create lookup by span_id
    events_by_span: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        span_id = ev.span_id
        if span_id:
            node = ev.model_dump()
            node["children"] = []
            events_by_span[span_id] = node

    # Build tree by linking children to parents
    roots: List[Dict[str, Any]] = []
    for ev in events:
        span_id = ev.span_id
        parent_id = ev.parent_span_id
        if not span_id:
            # Events without span_id go to roots
            roots.append(ev.model_dump())
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
