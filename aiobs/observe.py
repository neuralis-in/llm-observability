"""@observe decorator for tracing function execution."""

from __future__ import annotations

import asyncio
import functools
import inspect
import os
import time
from typing import Any, Callable, Optional, TypeVar, Union, overload

from .models import FunctionEvent, Callsite

F = TypeVar("F", bound=Callable[..., Any])


def _get_callsite(skip_frames: int = 2) -> Optional[Callsite]:
    """Extract callsite information from the call stack."""
    try:
        frames = inspect.stack()[skip_frames:]
        for fi in frames:
            fname = os.path.abspath(fi.filename)
            # Skip internal aiobs frames
            if f"{os.sep}aiobs{os.sep}" in fname:
                continue
            try:
                rel = os.path.relpath(fname, start=os.getcwd())
            except Exception:
                rel = fname
            return Callsite(file=rel, line=fi.lineno, function=fi.function)
    except Exception:
        pass
    return None


def _safe_repr(obj: Any, max_length: int = 500) -> Any:
    """Safely serialize an object for storage, truncating if too long."""
    try:
        # For basic types, return as-is
        if obj is None or isinstance(obj, (bool, int, float, str)):
            if isinstance(obj, str) and len(obj) > max_length:
                return obj[:max_length] + "..."
            return obj
        # For lists/tuples, recursively process
        if isinstance(obj, (list, tuple)):
            return [_safe_repr(item, max_length) for item in obj[:10]]  # Limit to 10 items
        # For dicts, recursively process
        if isinstance(obj, dict):
            return {
                str(k)[:100]: _safe_repr(v, max_length)
                for k, v in list(obj.items())[:20]  # Limit to 20 keys
            }
        # For objects with __dict__, try to extract key info
        if hasattr(obj, "__dict__"):
            return f"<{type(obj).__name__}>"
        # Fallback to string repr
        s = repr(obj)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s
    except Exception:
        return f"<{type(obj).__name__}>"


@overload
def observe(func: F) -> F: ...


@overload
def observe(
    *,
    name: Optional[str] = None,
    capture_args: bool = True,
    capture_result: bool = True,
) -> Callable[[F], F]: ...


def observe(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    capture_args: bool = True,
    capture_result: bool = True,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to trace function execution.

    Can be used with or without arguments:
        @observe
        def my_func(): ...

        @observe(name="custom_name")
        def my_func(): ...

    Args:
        func: The function to wrap (when used without parentheses)
        name: Optional custom name for the traced function
        capture_args: Whether to capture function arguments (default: True)
        capture_result: Whether to capture the return value (default: True)

    Returns:
        The wrapped function that records execution traces
    """

    def decorator(fn: F) -> F:
        # Get function metadata
        fn_name = name or fn.__name__
        fn_module = getattr(fn, "__module__", None)
        fn_qualname = getattr(fn, "__qualname__", fn.__name__)

        # Determine if the function is async
        is_async = asyncio.iscoroutinefunction(fn)

        if is_async:
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Import here to avoid circular imports
                from . import observer

                started = time.time()
                callsite = _get_callsite(skip_frames=2)
                error_msg: Optional[str] = None
                result: Any = None

                # Capture args if enabled
                captured_args = None
                captured_kwargs = None
                if capture_args:
                    try:
                        captured_args = [_safe_repr(a) for a in args]
                        captured_kwargs = {k: _safe_repr(v) for k, v in kwargs.items()}
                    except Exception:
                        pass

                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    ended = time.time()

                    # Capture result if enabled
                    captured_result = None
                    if capture_result and error_msg is None:
                        try:
                            captured_result = _safe_repr(result)
                        except Exception:
                            pass

                    event = FunctionEvent(
                        provider="function",
                        api=f"{fn_module}.{fn_qualname}" if fn_module else fn_qualname,
                        name=fn_name,
                        module=fn_module,
                        args=captured_args,
                        kwargs=captured_kwargs,
                        result=captured_result,
                        error=error_msg,
                        started_at=started,
                        ended_at=ended,
                        duration_ms=round((ended - started) * 1000, 3),
                        callsite=callsite,
                    )
                    observer._record_event(event)

            return async_wrapper  # type: ignore[return-value]

        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Import here to avoid circular imports
                from . import observer

                started = time.time()
                callsite = _get_callsite(skip_frames=2)
                error_msg: Optional[str] = None
                result: Any = None

                # Capture args if enabled
                captured_args = None
                captured_kwargs = None
                if capture_args:
                    try:
                        captured_args = [_safe_repr(a) for a in args]
                        captured_kwargs = {k: _safe_repr(v) for k, v in kwargs.items()}
                    except Exception:
                        pass

                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    ended = time.time()

                    # Capture result if enabled
                    captured_result = None
                    if capture_result and error_msg is None:
                        try:
                            captured_result = _safe_repr(result)
                        except Exception:
                            pass

                    event = FunctionEvent(
                        provider="function",
                        api=f"{fn_module}.{fn_qualname}" if fn_module else fn_qualname,
                        name=fn_name,
                        module=fn_module,
                        args=captured_args,
                        kwargs=captured_kwargs,
                        result=captured_result,
                        error=error_msg,
                        started_at=started,
                        ended_at=ended,
                        duration_ms=round((ended - started) * 1000, 3),
                        callsite=callsite,
                    )
                    observer._record_event(event)

            return sync_wrapper  # type: ignore[return-value]

    # Handle both @observe and @observe(...) syntax
    if func is not None:
        return decorator(func)
    return decorator

