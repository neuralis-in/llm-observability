import os
import sys
from unittest.mock import patch


# Ensure repo root is importable (so tests can import aiobs)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


@pytest.fixture(autouse=True)
def reset_observer_state():
    # Fresh collector state for each test
    from aiobs import observer
    from aiobs.tracer import clear_spans, clear_logs

    observer.reset()
    clear_spans()
    clear_logs()
    try:
        yield
    finally:
        observer.reset()
        clear_spans()
        clear_logs()


@pytest.fixture(autouse=True)
def set_test_api_key(monkeypatch):
    """Set a test API key for all tests."""
    monkeypatch.setenv("AIOBS_API_KEY", "aiobs_sk_test_key_for_testing")
    # Enable GenAI message content capture for OTel instrumentors
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")


@pytest.fixture(autouse=True)
def mock_shepherd_api():
    """Mock shepherd API calls to avoid network calls in tests."""
    with patch("aiobs.collector.Collector._validate_api_key"), \
         patch("aiobs.collector.Collector._record_usage"), \
         patch("aiobs.collector.Collector._flush_to_server"):
        yield
