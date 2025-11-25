from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class SessionMeta(BaseModel):
    pid: int
    cwd: str


class Session(BaseModel):
    id: str
    name: str
    started_at: float
    ended_at: Optional[float] = Field(default=None)
    meta: SessionMeta


class Callsite(BaseModel):
    file: Optional[str] = Field(default=None)
    line: Optional[int] = Field(default=None)
    function: Optional[str] = Field(default=None)


class Event(BaseModel):
    provider: str
    api: str
    request: Any
    response: Optional[Any] = None
    error: Optional[str] = Field(default=None)
    started_at: float
    ended_at: float
    duration_ms: float
    callsite: Optional[Callsite] = Field(default=None)
    span_id: Optional[str] = Field(default=None)
    parent_span_id: Optional[str] = Field(default=None)


class FunctionEvent(BaseModel):
    """Event model for tracing decorated functions."""
    provider: str = Field(default="function")
    api: str  # Function name (qualified)
    name: str  # Display name for the function
    module: Optional[str] = Field(default=None)
    args: Optional[List[Any]] = Field(default=None)
    kwargs: Optional[dict] = Field(default=None)
    result: Optional[Any] = Field(default=None)
    error: Optional[str] = Field(default=None)
    started_at: float
    ended_at: float
    duration_ms: float
    callsite: Optional[Callsite] = Field(default=None)
    span_id: Optional[str] = Field(default=None)
    parent_span_id: Optional[str] = Field(default=None)


class ObservedEvent(Event):
    session_id: str


class ObservedFunctionEvent(FunctionEvent):
    """Function event with session_id for export."""
    session_id: str


class ObservabilityExport(BaseModel):
    sessions: List[Session]
    events: List[ObservedEvent]
    function_events: List[ObservedFunctionEvent] = Field(default_factory=list)
    trace_tree: Optional[List[Any]] = Field(default=None)  # Nested tree structure (optional)
    generated_at: float
    version: int = 1
