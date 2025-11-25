import json
import os
from aiobs import observer
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
    assert set(data.keys()) == {"sessions", "events", "function_events", "generated_at", "version"}
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
