import json
import os
from aiobs import observer, observe
from aiobs.models import Event as ObsEvent


def test_observer_flush_json_structure(tmp_path):
    # Start session
    observer.observe("core-structure")

    # Record a minimal synthetic event using typed model
    ev = ObsEvent(
        provider="test",
        api="dummy.call",
        request={"a": 1},
        response={"b": 2},
        error=None,
        started_at=0.0,
        ended_at=1.0,
        duration_ms=1000.0,
        callsite=None,
    )
    observer._record_event(ev)

    # Flush and validate JSON
    out_path = tmp_path / "obs.json"
    observer.end()
    written = observer.flush(str(out_path))
    assert os.path.exists(written)

    data = json.loads(out_path.read_text())
    assert set(data.keys()) == {"sessions", "events", "function_events", "trace_tree", "enh_prompt_traces", "generated_at", "version"}
    assert isinstance(data["sessions"], list) and data["sessions"], "sessions should not be empty"
    assert isinstance(data["events"], list) and data["events"], "events should not be empty"
    e = data["events"][0]
    for key in [
        "provider",
        "api",
        "request",
        "response",
        "started_at",
        "ended_at",
        "duration_ms",
        "session_id",
    ]:
        assert key in e


def test_observer_flush_without_trace_tree(tmp_path):
    """Test that trace_tree can be disabled via include_trace_tree=False."""
    @observe
    def outer():
        return inner()

    @observe
    def inner():
        return 42

    observer.observe("no-trace-tree")
    outer()
    observer.end()

    out_path = tmp_path / "obs.json"
    written = observer.flush(str(out_path), include_trace_tree=False)
    assert os.path.exists(written)

    data = json.loads(out_path.read_text())
    # trace_tree key should be present but set to None
    assert "trace_tree" in data
    assert data["trace_tree"] is None
    # function_events should still have span_id and parent_span_id
    assert len(data["function_events"]) == 2
    for ev in data["function_events"]:
        assert "span_id" in ev
        assert "parent_span_id" in ev


def test_observer_flush_with_trace_tree(tmp_path):
    """Test that trace_tree is included by default and has correct nesting."""
    @observe
    def parent_func():
        return child_func()

    @observe
    def child_func():
        return "result"

    observer.observe("with-trace-tree")
    parent_func()
    observer.end()

    out_path = tmp_path / "obs.json"
    written = observer.flush(str(out_path))  # default: include_trace_tree=True
    assert os.path.exists(written)

    data = json.loads(out_path.read_text())
    # trace_tree should be a list with nested structure
    assert "trace_tree" in data
    assert isinstance(data["trace_tree"], list)
    assert len(data["trace_tree"]) == 1  # One root: parent_func

    root = data["trace_tree"][0]
    assert root["name"] == "parent_func"
    assert "children" in root
    assert len(root["children"]) == 1  # One child: child_func
    assert root["children"][0]["name"] == "child_func"
