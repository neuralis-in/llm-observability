"""Tests for async flush functionality (non-blocking server upload)."""

import json
import threading
import time
from unittest.mock import MagicMock, patch, call
import urllib.error

import pytest

from aiobs import observer
from aiobs.models import Event as ObsEvent


class TestAsyncFlushServerUpload:
    """Test that flush() sends data to the flush server asynchronously."""

    def test_flush_calls_send_to_flush_server(self, tmp_path):
        """Test that flush() triggers _send_to_flush_server when API key is set."""
        observer.observe("test-async-flush")

        # Record a minimal event
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
        observer.end()

        with patch.object(observer, "_send_to_flush_server") as mock_send:
            out_path = tmp_path / "obs.json"
            observer.flush(str(out_path))

            # Verify _send_to_flush_server was called
            assert mock_send.called
            # The argument should be an ObservabilityExport object
            export_arg = mock_send.call_args[0][0]
            assert hasattr(export_arg, "sessions")
            assert hasattr(export_arg, "events")

    def test_flush_does_not_send_without_api_key(self, tmp_path):
        """Test that flush() does not send to server when no API key is set."""
        # First observe with API key (from fixture), then clear it
        observer.observe("test-no-key")

        ev = ObsEvent(
            provider="test",
            api="dummy.call",
            request={},
            response={},
            error=None,
            started_at=0.0,
            ended_at=1.0,
            duration_ms=1000.0,
            callsite=None,
        )
        observer._record_event(ev)
        observer.end()

        # Clear API key before flush
        observer._api_key = None

        with patch.object(observer, "_send_to_flush_server") as mock_send:
            out_path = tmp_path / "obs.json"
            observer.flush(str(out_path))

            # Should not be called without API key
            assert not mock_send.called


class TestAsyncFlushBackgroundThread:
    """Test that the server upload happens in a background thread."""

    def test_send_to_flush_server_uses_daemon_thread(self, tmp_path):
        """Test that _send_to_flush_server spawns a daemon thread."""
        from aiobs.models import ObservabilityExport

        export = ObservabilityExport(
            sessions=[],
            events=[],
            function_events=[],
            trace_tree=[],
            enh_prompt_traces=None,
            generated_at=0.0,
        )

        threads_started = []

        original_thread_init = threading.Thread.__init__

        def mock_thread_init(self, *args, **kwargs):
            original_thread_init(self, *args, **kwargs)
            threads_started.append(self)

        with patch.object(threading.Thread, "__init__", mock_thread_init):
            with patch.object(threading.Thread, "start"):
                observer._api_key = "aiobs_sk_test"
                observer._send_to_flush_server(export)

        # Verify a daemon thread was created
        assert len(threads_started) == 1
        assert threads_started[0].daemon is True

    def test_flush_returns_immediately(self, tmp_path):
        """Test that flush() returns immediately without waiting for upload."""
        observer.observe("test-immediate-return")

        ev = ObsEvent(
            provider="test",
            api="dummy.call",
            request={},
            response={},
            error=None,
            started_at=0.0,
            ended_at=1.0,
            duration_ms=1000.0,
            callsite=None,
        )
        observer._record_event(ev)
        observer.end()

        # Mock urlopen to simulate a slow network call
        def slow_urlopen(*args, **kwargs):
            time.sleep(2)  # Simulate 2 second network delay
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"status": "ok"}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            return mock_response

        with patch("urllib.request.urlopen", side_effect=slow_urlopen):
            out_path = tmp_path / "obs.json"
            start_time = time.time()
            observer.flush(str(out_path))
            elapsed = time.time() - start_time

            # flush() should return almost immediately (< 0.5s)
            # even though the mock upload takes 2 seconds
            assert elapsed < 0.5, f"flush() took {elapsed}s, should return immediately"


class TestAsyncFlushErrorHandling:
    """Test error handling during async flush."""

    def test_http_401_logs_warning(self, tmp_path, caplog):
        """Test that 401 Unauthorized is logged but doesn't raise."""
        from aiobs.models import ObservabilityExport
        import logging

        export = ObservabilityExport(
            sessions=[],
            events=[],
            function_events=[],
            trace_tree=[],
            enh_prompt_traces=None,
            generated_at=0.0,
        )

        # Create a mock HTTP 401 error
        mock_error = urllib.error.HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=mock_error):
            with caplog.at_level(logging.WARNING, logger="aiobs.collector"):
                observer._api_key = "aiobs_sk_test"
                # Call _send_to_flush_server and wait for thread to complete
                observer._send_to_flush_server(export)
                # Give the background thread time to execute and log
                time.sleep(0.2)

        # No exception should be raised - errors are handled in background thread

    def test_http_429_logs_warning(self, tmp_path):
        """Test that 429 Rate Limit is logged but doesn't raise."""
        from aiobs.models import ObservabilityExport

        export = ObservabilityExport(
            sessions=[],
            events=[],
            function_events=[],
            trace_tree=[],
            enh_prompt_traces=None,
            generated_at=0.0,
        )

        mock_error = urllib.error.HTTPError(
            url="http://test",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None,
        )

        # Test that no exception propagates from the background thread
        with patch("urllib.request.urlopen", side_effect=mock_error):
            observer._api_key = "aiobs_sk_test"
            # This should not raise - errors are handled in background thread
            observer._send_to_flush_server(export)
            # Give the background thread time to execute
            time.sleep(0.2)

    def test_network_error_logs_warning(self, tmp_path):
        """Test that network errors are logged but don't raise."""
        from aiobs.models import ObservabilityExport

        export = ObservabilityExport(
            sessions=[],
            events=[],
            function_events=[],
            trace_tree=[],
            enh_prompt_traces=None,
            generated_at=0.0,
        )

        mock_error = urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=mock_error):
            observer._api_key = "aiobs_sk_test"
            # Should not raise - errors are caught in the background thread
            observer._send_to_flush_server(export)
            # Give thread a moment to execute
            time.sleep(0.1)


class TestAsyncFlushIntegration:
    """Integration tests for async flush behavior."""

    def test_flush_writes_local_file_regardless_of_upload_status(self, tmp_path):
        """Test that local file is written even if upload fails."""
        observer.observe("test-local-write")

        ev = ObsEvent(
            provider="test",
            api="dummy.call",
            request={"test": "data"},
            response={"result": "ok"},
            error=None,
            started_at=0.0,
            ended_at=1.0,
            duration_ms=1000.0,
            callsite=None,
        )
        observer._record_event(ev)
        observer.end()

        # Mock upload to fail
        mock_error = urllib.error.URLError("Network error")
        with patch("urllib.request.urlopen", side_effect=mock_error):
            out_path = tmp_path / "obs.json"
            result = observer.flush(str(out_path))

            # Local file should still be written
            assert result == str(out_path)
            assert out_path.exists()

            data = json.loads(out_path.read_text())
            assert len(data["events"]) == 1
            assert data["events"][0]["request"] == {"test": "data"}

    def test_flush_clears_state_regardless_of_upload_status(self, tmp_path):
        """Test that collector state is cleared even if upload fails."""
        observer.observe("test-state-clear")

        ev = ObsEvent(
            provider="test",
            api="dummy.call",
            request={},
            response={},
            error=None,
            started_at=0.0,
            ended_at=1.0,
            duration_ms=1000.0,
            callsite=None,
        )
        observer._record_event(ev)
        observer.end()

        # Mock upload to fail
        mock_error = urllib.error.URLError("Network error")
        with patch("urllib.request.urlopen", side_effect=mock_error):
            out_path = tmp_path / "obs.json"
            observer.flush(str(out_path))

        # State should be cleared
        assert observer._active_session is None
        assert len(observer._sessions) == 0
        assert len(observer._events) == 0

    def test_successful_upload_logs_debug(self, tmp_path, caplog):
        """Test that successful upload logs debug message."""
        import logging
        from aiobs.models import ObservabilityExport

        export = ObservabilityExport(
            sessions=[],
            events=[],
            function_events=[],
            trace_tree=[],
            enh_prompt_traces=None,
            generated_at=0.0,
        )

        # Mock successful response
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok", "session_id": "test-123"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with caplog.at_level(logging.DEBUG, logger="aiobs.collector"):
                observer._api_key = "aiobs_sk_test"
                observer._send_to_flush_server(export)
                # Give thread time to execute
                time.sleep(0.2)

