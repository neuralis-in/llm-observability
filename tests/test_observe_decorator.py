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


class TestObserveDecoratorEnhPrompt:
    """Test @observe with enh_prompt feature."""

    def test_enh_prompt_false_by_default(self):
        """enh_prompt should be False by default."""

        @observe
        def regular_func(x):
            return x * 2

        observer.observe("test-enh-prompt-default")
        regular_func(5)
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.enh_prompt is False
        assert ev.enh_prompt_id is None
        assert ev.auto_enhance_after is None

    def test_enh_prompt_enabled(self):
        """enh_prompt=True should set flag and generate enh_prompt_id."""

        @observe(enh_prompt=True)
        def enhanced_func(x):
            return x * 2

        observer.observe("test-enh-prompt-enabled")
        enhanced_func(5)
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.enh_prompt is True
        assert ev.enh_prompt_id is not None
        # enh_prompt_id should be a valid UUID string
        import uuid
        uuid.UUID(ev.enh_prompt_id)  # Should not raise

    def test_enh_prompt_with_auto_enhance_after(self):
        """auto_enhance_after should be captured when enh_prompt=True."""

        @observe(enh_prompt=True, auto_enhance_after=10)
        def enhanced_func(x):
            return x * 2

        observer.observe("test-auto-enhance-after")
        enhanced_func(5)
        observer.end()

        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.enh_prompt is True
        assert ev.auto_enhance_after == 10

    def test_multiple_enh_prompt_functions_unique_ids(self):
        """Each enh_prompt function call should get a unique enh_prompt_id."""

        @observe(enh_prompt=True, auto_enhance_after=5)
        def func_a(x):
            return x + 1

        @observe(enh_prompt=True, auto_enhance_after=10)
        def func_b(x):
            return x + 2

        observer.observe("test-multiple-enh-prompt")
        func_a(1)
        func_b(2)
        func_a(3)  # Second call to func_a
        observer.end()

        events = list(observer._events.values())[0]
        assert len(events) == 3

        enh_prompt_ids = [ev.enh_prompt_id for ev in events]
        # All IDs should be unique
        assert len(set(enh_prompt_ids)) == 3
        # All should be valid UUIDs
        import uuid
        for eid in enh_prompt_ids:
            uuid.UUID(eid)

    def test_nested_enh_prompt_functions(self):
        """Nested enh_prompt functions should each have their own enh_prompt_id."""

        @observe(enh_prompt=True, auto_enhance_after=5)
        def outer(x):
            return inner(x * 2)

        @observe(enh_prompt=True, auto_enhance_after=10)
        def inner(x):
            return x + 1

        observer.observe("test-nested-enh-prompt")
        outer(5)
        observer.end()

        events = list(observer._events.values())[0]
        assert len(events) == 2

        outer_ev = next(ev for ev in events if ev.name == "outer")
        inner_ev = next(ev for ev in events if ev.name == "inner")

        assert outer_ev.enh_prompt is True
        assert inner_ev.enh_prompt is True
        assert outer_ev.enh_prompt_id != inner_ev.enh_prompt_id
        assert outer_ev.auto_enhance_after == 5
        assert inner_ev.auto_enhance_after == 10

    def test_mixed_enh_prompt_and_regular_functions(self):
        """Mix of enh_prompt and regular functions should work correctly."""

        @observe(enh_prompt=True, auto_enhance_after=5)
        def enhanced(x):
            return regular(x * 2)

        @observe
        def regular(x):
            return x + 1

        observer.observe("test-mixed-enh-prompt")
        enhanced(5)
        observer.end()

        events = list(observer._events.values())[0]
        assert len(events) == 2

        enhanced_ev = next(ev for ev in events if ev.name == "enhanced")
        regular_ev = next(ev for ev in events if ev.name == "regular")

        assert enhanced_ev.enh_prompt is True
        assert enhanced_ev.enh_prompt_id is not None
        assert regular_ev.enh_prompt is False
        assert regular_ev.enh_prompt_id is None

    def test_async_enh_prompt(self):
        """enh_prompt should work with async functions."""
        import asyncio

        @observe(enh_prompt=True, auto_enhance_after=3)
        async def async_enhanced(x):
            await asyncio.sleep(0.001)
            return x * 2

        observer.observe("test-async-enh-prompt")
        result = asyncio.run(async_enhanced(5))
        observer.end()

        assert result == 10
        events = list(observer._events.values())[0]
        ev = events[0]
        assert ev.enh_prompt is True
        assert ev.enh_prompt_id is not None
        assert ev.auto_enhance_after == 3


class TestObserveDecoratorEnhPromptExport:
    """Test JSON export with enh_prompt features."""

    def test_export_enh_prompt_traces_list(self, tmp_path):
        """enh_prompt_traces should be a list of enh_prompt_ids."""

        @observe(enh_prompt=True, auto_enhance_after=5)
        def func_a(x):
            return x + 1

        @observe(enh_prompt=True, auto_enhance_after=10)
        def func_b(x):
            return x + 2

        @observe
        def regular(x):
            return x * 2

        observer.observe("test-export-enh-prompt-traces")
        func_a(1)
        func_b(2)
        regular(3)
        observer.end()

        out_path = tmp_path / "obs.json"
        observer.flush(str(out_path))

        data = json.loads(out_path.read_text())

        # enh_prompt_traces should contain only IDs from enh_prompt=True functions
        assert "enh_prompt_traces" in data
        assert isinstance(data["enh_prompt_traces"], list)
        assert len(data["enh_prompt_traces"]) == 2

        # Get IDs from function_events for verification
        enh_ids_from_events = [
            ev["enh_prompt_id"]
            for ev in data["function_events"]
            if ev.get("enh_prompt") is True
        ]
        assert set(data["enh_prompt_traces"]) == set(enh_ids_from_events)

    def test_export_enh_prompt_id_in_trace_tree(self, tmp_path):
        """enh_prompt_id should be present in trace_tree nodes."""

        @observe(enh_prompt=True, auto_enhance_after=5)
        def outer(x):
            return inner(x * 2)

        @observe(enh_prompt=True, auto_enhance_after=10)
        def inner(x):
            return x + 1

        observer.observe("test-export-trace-tree-enh-id")
        outer(5)
        observer.end()

        out_path = tmp_path / "obs.json"
        observer.flush(str(out_path))

        data = json.loads(out_path.read_text())

        # Check trace_tree has enh_prompt_id
        assert len(data["trace_tree"]) == 1
        root = data["trace_tree"][0]
        assert root["name"] == "outer"
        assert root["enh_prompt"] is True
        assert root["enh_prompt_id"] is not None
        assert root["auto_enhance_after"] == 5

        # Check nested child
        assert len(root["children"]) == 1
        child = root["children"][0]
        assert child["name"] == "inner"
        assert child["enh_prompt"] is True
        assert child["enh_prompt_id"] is not None
        assert child["auto_enhance_after"] == 10

        # IDs should be different
        assert root["enh_prompt_id"] != child["enh_prompt_id"]

    def test_export_function_event_fields_enh_prompt(self, tmp_path):
        """Function events should include all enh_prompt fields."""

        @observe(enh_prompt=True, auto_enhance_after=7)
        def my_func(a):
            return a * 2

        observer.observe("test-export-enh-fields")
        my_func(5)
        observer.end()

        out_path = tmp_path / "obs.json"
        observer.flush(str(out_path))

        data = json.loads(out_path.read_text())
        ev = data["function_events"][0]

        assert ev["enh_prompt"] is True
        assert ev["enh_prompt_id"] is not None
        assert ev["auto_enhance_after"] == 7

    def test_export_no_enh_prompt_traces_when_none(self, tmp_path):
        """enh_prompt_traces should be None/empty when no enh_prompt functions."""

        @observe
        def regular_func(x):
            return x * 2

        observer.observe("test-export-no-enh-traces")
        regular_func(5)
        observer.end()

        out_path = tmp_path / "obs.json"
        observer.flush(str(out_path))

        data = json.loads(out_path.read_text())

        # Should be empty list or None
        enh_traces = data.get("enh_prompt_traces")
        assert enh_traces is None or enh_traces == []

    def test_export_enh_prompt_traces_across_multiple_calls(self, tmp_path):
        """Multiple calls to same enh_prompt function should each generate unique ID."""

        @observe(enh_prompt=True, auto_enhance_after=5)
        def repeatable(x):
            return x * 2

        observer.observe("test-export-multiple-calls")
        repeatable(1)
        repeatable(2)
        repeatable(3)
        observer.end()

        out_path = tmp_path / "obs.json"
        observer.flush(str(out_path))

        data = json.loads(out_path.read_text())

        assert len(data["enh_prompt_traces"]) == 3
        assert len(data["function_events"]) == 3

        # All IDs should be unique
        assert len(set(data["enh_prompt_traces"])) == 3

