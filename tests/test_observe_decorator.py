"""Tests for the @observe decorator."""

import asyncio
import json
import os

import pytest

from aiobs import observer, observe
from aiobs.models import FunctionEvent


class TestObserveDecoratorBasic:
    """Test basic @observe decorator functionality."""

    def test_sync_function_traced(self):
        """Basic sync function should be traced."""

        @observe
        def add(a, b):
            return a + b

        observer.observe("test-sync")
        result = add(2, 3)
        observer.end()

        assert result == 5

        # Check event was recorded
        events = list(observer._events.values())[0]
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, FunctionEvent)
        assert ev.name == "add"
        assert ev.args == [2, 3]
        assert ev.result == 5
        assert ev.error is None
        assert ev.duration_ms > 0

    def test_custom_name(self):
        """Custom name should override function name."""

        @observe(name="custom_add")
        def add(a, b):
            return a + b

        observer.observe("test-custom-name")
        add(1, 2)
        observer.end()

        events = list(observer._events.values())[0]
        assert events[0].name == "custom_add"

    def test_function_with_kwargs(self):
        """Function with kwargs should capture them."""

        @observe
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        observer.observe("test-kwargs")
        result = greet("World", greeting="Hi")
        observer.end()

        assert result == "Hi, World!"
        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.args == ["World"]
        assert ev.kwargs == {"greeting": "Hi"}

    def test_function_with_args_and_kwargs(self):
        """Function with *args and **kwargs should be handled."""

        @observe
        def variadic(a, *args, **kwargs):
            return a + sum(args) + sum(kwargs.values())

        observer.observe("test-variadic")
        result = variadic(1, 2, 3, x=4, y=5)
        observer.end()

        assert result == 15
        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.args == [1, 2, 3]
        assert ev.kwargs == {"x": 4, "y": 5}


class TestObserveDecoratorAsync:
    """Test @observe with async functions."""

    def test_async_function_traced(self):
        """Async function should be traced."""

        @observe
        async def async_add(a, b):
            await asyncio.sleep(0.001)
            return a + b

        observer.observe("test-async")
        result = asyncio.run(async_add(10, 20))
        observer.end()

        assert result == 30
        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.name == "async_add"
        assert ev.args == [10, 20]
        assert ev.result == 30
        assert ev.duration_ms >= 1  # At least 1ms from sleep

    def test_async_with_custom_name(self):
        """Async function with custom name."""

        @observe(name="async_operation")
        async def fetch():
            await asyncio.sleep(0.001)
            return {"data": "value"}

        observer.observe("test-async-name")
        result = asyncio.run(fetch())
        observer.end()

        assert result == {"data": "value"}
        events = list(observer._events.values())[0]
        assert events[0].name == "async_operation"


class TestObserveDecoratorOptions:
    """Test @observe decorator options."""

    def test_capture_args_false(self):
        """capture_args=False should not capture arguments."""

        @observe(capture_args=False)
        def sensitive(password):
            return "authenticated"

        observer.observe("test-no-args")
        sensitive("secret123")
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.args is None
        assert ev.kwargs is None
        assert ev.result == "authenticated"

    def test_capture_result_false(self):
        """capture_result=False should not capture return value."""

        @observe(capture_result=False)
        def big_data():
            return {"large": "data" * 1000}

        observer.observe("test-no-result")
        result = big_data()
        observer.end()

        assert "large" in result
        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.result is None
        assert ev.args == []

    def test_both_options_false(self):
        """Both capture options can be disabled."""

        @observe(capture_args=False, capture_result=False)
        def minimal(x):
            return x * 2

        observer.observe("test-minimal")
        minimal(5)
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.args is None
        assert ev.result is None
        assert ev.name == "minimal"
        assert ev.duration_ms >= 0


class TestObserveDecoratorErrorHandling:
    """Test @observe error handling."""

    def test_exception_captured(self):
        """Exceptions should be captured and re-raised."""

        @observe
        def will_fail():
            raise ValueError("Test error")

        observer.observe("test-error")
        with pytest.raises(ValueError, match="Test error"):
            will_fail()
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.error == "ValueError: Test error"
        assert ev.result is None

    def test_async_exception_captured(self):
        """Async exceptions should be captured and re-raised."""

        @observe
        async def async_fail():
            await asyncio.sleep(0.001)
            raise RuntimeError("Async error")

        observer.observe("test-async-error")
        with pytest.raises(RuntimeError, match="Async error"):
            asyncio.run(async_fail())
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.error == "RuntimeError: Async error"

    def test_exception_with_args_captured(self):
        """Args should still be captured even when exception occurs."""

        @observe
        def divide(a, b):
            return a / b

        observer.observe("test-error-args")
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.args == [10, 0]
        assert "ZeroDivisionError" in ev.error


class TestObserveDecoratorNested:
    """Test nested @observe decorated functions."""

    def test_nested_calls(self):
        """Nested decorated functions should all be traced."""

        @observe(name="outer")
        def outer(x):
            return inner(x * 2)

        @observe(name="inner")
        def inner(x):
            return x + 1

        observer.observe("test-nested")
        result = outer(5)
        observer.end()

        assert result == 11
        events = list(observer._events.values())[0]
        assert len(events) == 2

        # Inner should be recorded first (LIFO in finally blocks)
        names = [ev.name for ev in events]
        assert "inner" in names
        assert "outer" in names

    def test_deeply_nested(self):
        """Multiple levels of nesting should work."""

        @observe(name="level1")
        def level1(x):
            return level2(x + 1)

        @observe(name="level2")
        def level2(x):
            return level3(x + 1)

        @observe(name="level3")
        def level3(x):
            return x + 1

        observer.observe("test-deep-nested")
        result = level1(0)
        observer.end()

        assert result == 3
        events = list(observer._events.values())[0]
        assert len(events) == 3


class TestObserveDecoratorExport:
    """Test JSON export with @observe decorated functions."""

    def test_export_structure(self, tmp_path):
        """Exported JSON should have function_events array."""

        @observe
        def traced_func(x):
            return x * 2

        observer.observe("test-export")
        traced_func(5)
        observer.end()

        out_path = tmp_path / "obs.json"
        observer.flush(str(out_path))

        data = json.loads(out_path.read_text())
        assert "function_events" in data
        assert "events" in data
        assert len(data["function_events"]) == 1
        assert len(data["events"]) == 0

    def test_export_function_event_fields(self, tmp_path):
        """Function event should have all required fields."""

        @observe(name="test_func")
        def my_func(a, b=10):
            return a + b

        observer.observe("test-export-fields")
        my_func(5, b=20)
        observer.end()

        out_path = tmp_path / "obs.json"
        observer.flush(str(out_path))

        data = json.loads(out_path.read_text())
        ev = data["function_events"][0]

        # Check all expected fields
        assert ev["provider"] == "function"
        assert ev["name"] == "test_func"
        assert ev["api"].endswith("my_func")
        assert ev["args"] == [5]
        assert ev["kwargs"] == {"b": 20}
        assert ev["result"] == 25
        assert ev["error"] is None
        assert "started_at" in ev
        assert "ended_at" in ev
        assert "duration_ms" in ev
        assert "session_id" in ev

    def test_export_with_errors(self, tmp_path):
        """Error events should be properly exported."""

        @observe
        def fail_func():
            raise ValueError("Export test error")

        observer.observe("test-export-error")
        try:
            fail_func()
        except ValueError:
            pass
        observer.end()

        out_path = tmp_path / "obs.json"
        observer.flush(str(out_path))

        data = json.loads(out_path.read_text())
        ev = data["function_events"][0]
        assert ev["error"] == "ValueError: Export test error"
        assert ev["result"] is None


class TestObserveDecoratorSerialization:
    """Test safe serialization of complex objects."""

    def test_complex_args_serialized(self):
        """Complex arguments should be safely serialized."""

        class CustomObj:
            def __init__(self, value):
                self.value = value

        @observe
        def process(obj):
            return obj.value

        observer.observe("test-complex-args")
        process(CustomObj(42))
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        # Should be serialized as type name, not crash
        assert ev.args[0] == "<CustomObj>"

    def test_large_string_truncated(self):
        """Large strings should be truncated."""

        @observe
        def echo(s):
            return s

        observer.observe("test-large-string")
        large = "x" * 1000
        echo(large)
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        # Should be truncated with ...
        assert len(ev.args[0]) < 1000
        assert ev.args[0].endswith("...")

    def test_nested_dict_serialized(self):
        """Nested dicts should be serialized."""

        @observe
        def process_dict(d):
            return d

        observer.observe("test-nested-dict")
        result = process_dict({"a": {"b": {"c": 1}}})
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.args[0] == {"a": {"b": {"c": 1}}}


class TestObserveDecoratorNoSession:
    """Test @observe when no session is active."""

    def test_no_crash_without_session(self):
        """Decorated functions should work even without active session."""

        @observe
        def standalone():
            return "result"

        # No observer.observe() called
        result = standalone()
        assert result == "result"
        # Should not crash, just not record anything

