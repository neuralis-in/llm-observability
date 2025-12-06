"""Tests for aiobs.exporters module."""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiobs import observer, observe
from aiobs.models import Event as ObsEvent, ObservabilityExport, Session, SessionMeta
from aiobs.exporters import (
    BaseExporter,
    ExportResult,
    ExportError,
    CustomExporter,
    CompositeExporter,
    GCSExporter,
)


# =============================================================================
# BaseExporter Tests
# =============================================================================

class TestExportResult:
    def test_success_result(self):
        result = ExportResult(
            success=True,
            destination="gs://bucket/path",
            bytes_written=1024,
            metadata={"key": "value"},
        )
        assert result.success is True
        assert result.destination == "gs://bucket/path"
        assert result.bytes_written == 1024
        assert result.metadata == {"key": "value"}
        assert result.error is None

    def test_failure_result(self):
        result = ExportResult(
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"
        assert result.destination is None

    def test_repr_success(self):
        result = ExportResult(success=True, destination="test://dest", bytes_written=100)
        assert "success=True" in repr(result)
        assert "test://dest" in repr(result)

    def test_repr_failure(self):
        result = ExportResult(success=False, error="oops")
        assert "success=False" in repr(result)
        assert "oops" in repr(result)


class TestExportError:
    def test_basic_error(self):
        err = ExportError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.cause is None

    def test_error_with_cause(self):
        cause = ValueError("original error")
        err = ExportError("Wrapped error", cause=cause)
        assert str(err) == "Wrapped error"
        assert err.cause is cause


# =============================================================================
# CustomExporter Tests
# =============================================================================

class TestCustomExporter:
    def test_handler_required_callable(self):
        with pytest.raises(ValueError, match="handler must be a callable"):
            CustomExporter(handler="not a function")

    def test_basic_handler(self):
        calls = []

        def handler(data, options):
            calls.append((data, options))
            return ExportResult(success=True, destination="custom://test")

        exporter = CustomExporter(handler=handler)
        assert exporter.name == "custom"

        # Create mock data
        data = _create_mock_export()
        result = exporter.export(data)

        assert len(calls) == 1
        assert calls[0][0] == data
        assert result.success is True
        assert result.destination == "custom://test"

    def test_handler_with_options(self):
        received_options = []

        def handler(data, options):
            received_options.append(options)
            return None

        exporter = CustomExporter(
            handler=handler,
            default_options={"default_key": "default_value"},
        )

        data = _create_mock_export()
        exporter.export(data, extra_key="extra_value")

        assert len(received_options) == 1
        assert received_options[0] == {
            "default_key": "default_value",
            "extra_key": "extra_value",
        }

    def test_handler_returns_none(self):
        def handler(data, options):
            return None

        exporter = CustomExporter(handler=handler)
        data = _create_mock_export()
        result = exporter.export(data)

        assert result.success is True
        assert result.metadata == {"handler": "custom"}

    def test_handler_returns_dict(self):
        def handler(data, options):
            return {
                "success": True,
                "destination": "dict://result",
                "bytes_written": 500,
                "metadata": {"custom": "data"},
            }

        exporter = CustomExporter(handler=handler)
        data = _create_mock_export()
        result = exporter.export(data)

        assert result.success is True
        assert result.destination == "dict://result"
        assert result.bytes_written == 500
        assert result.metadata == {"custom": "data"}

    def test_handler_raises_exception(self):
        def handler(data, options):
            raise ValueError("Handler failed")

        exporter = CustomExporter(handler=handler)
        data = _create_mock_export()

        with pytest.raises(ExportError, match="Custom export handler failed"):
            exporter.export(data)

    def test_custom_name(self):
        exporter = CustomExporter(handler=lambda d, o: None, name="my-exporter")
        assert exporter.name == "my-exporter"


# =============================================================================
# CompositeExporter Tests
# =============================================================================

class TestCompositeExporter:
    def test_requires_at_least_one_exporter(self):
        with pytest.raises(ValueError, match="At least one exporter is required"):
            CompositeExporter([])

    def test_runs_all_exporters(self):
        calls = []

        def make_handler(name):
            def handler(data, options):
                calls.append(name)
                return ExportResult(success=True, destination=f"{name}://dest")
            return handler

        exp1 = CustomExporter(handler=make_handler("exp1"), name="exp1")
        exp2 = CustomExporter(handler=make_handler("exp2"), name="exp2")

        composite = CompositeExporter([exp1, exp2])
        data = _create_mock_export()
        result = composite.export(data)

        assert calls == ["exp1", "exp2"]
        assert result.success is True
        assert result.metadata["exporters_count"] == 2
        assert len(result.metadata["results"]) == 2

    def test_collects_failures_without_stop_on_error(self):
        def success_handler(data, options):
            return ExportResult(success=True, destination="success://dest")

        def fail_handler(data, options):
            raise ExportError("Intentional failure")

        exp1 = CustomExporter(handler=fail_handler, name="fail")
        exp2 = CustomExporter(handler=success_handler, name="success")

        composite = CompositeExporter([exp1, exp2], stop_on_error=False)
        data = _create_mock_export()
        result = composite.export(data)

        # Should not be all success
        assert result.success is False
        assert "fail:" in result.error
        assert "Intentional failure" in result.error
        # Both exporters should have run
        assert len(result.metadata["results"]) == 2

    def test_stops_on_error_when_enabled(self):
        calls = []

        def fail_handler(data, options):
            calls.append("fail")
            raise ExportError("Intentional failure")

        def success_handler(data, options):
            calls.append("success")
            return ExportResult(success=True)

        exp1 = CustomExporter(handler=fail_handler, name="fail")
        exp2 = CustomExporter(handler=success_handler, name="success")

        composite = CompositeExporter([exp1, exp2], stop_on_error=True)
        data = _create_mock_export()

        with pytest.raises(ExportError):
            composite.export(data)

        # Second exporter should not have run
        assert calls == ["fail"]

    def test_add_exporter(self):
        exp1 = CustomExporter(handler=lambda d, o: None, name="exp1")
        composite = CompositeExporter([exp1])

        exp2 = CustomExporter(handler=lambda d, o: None, name="exp2")
        result = composite.add(exp2)

        # Should return self for chaining
        assert result is composite
        assert len(composite._exporters) == 2


# =============================================================================
# GCSExporter Tests
# =============================================================================

class TestGCSExporter:
    def test_initialization(self):
        exporter = GCSExporter(
            bucket="my-bucket",
            prefix="traces/",
            project="my-project",
            credentials_path="/path/to/creds.json",
        )
        assert exporter.bucket == "my-bucket"
        assert exporter.prefix == "traces/"
        assert exporter.project == "my-project"
        assert exporter.credentials_path == "/path/to/creds.json"
        assert exporter.name == "gcs"

    def test_prefix_normalization(self):
        # Without trailing slash
        exp1 = GCSExporter(bucket="b", prefix="traces")
        assert exp1.prefix == "traces/"

        # With trailing slash
        exp2 = GCSExporter(bucket="b", prefix="traces/")
        assert exp2.prefix == "traces/"

        # Empty prefix
        exp3 = GCSExporter(bucket="b", prefix="")
        assert exp3.prefix == ""

    def test_filename_template(self):
        exporter = GCSExporter(
            bucket="b",
            filename_template="{date}-{session_id}.json",
        )
        assert exporter.filename_template == "{date}-{session_id}.json"

    def test_generate_filename(self):
        exporter = GCSExporter(bucket="b", filename_template="{session_id}.json")
        data = _create_mock_export(session_id="test-session-123")
        filename = exporter._generate_filename(data)
        assert filename == "test-session-123.json"

    def test_generate_filename_empty_sessions(self):
        exporter = GCSExporter(bucket="b", filename_template="{session_id}.json")
        data = ObservabilityExport(
            sessions=[],
            events=[],
            function_events=[],
            generated_at=0.0,
        )
        filename = exporter._generate_filename(data)
        assert filename == "unknown.json"

    def test_export_success(self):
        """Test successful export by injecting a mock client."""
        # Setup mocks
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        exporter = GCSExporter(
            bucket="test-bucket",
            prefix="traces/",
            project="test-project",
        )
        # Inject mock client directly
        exporter._client = mock_client

        data = _create_mock_export(session_id="sess-123")
        result = exporter.export(data)

        # Verify bucket and blob were accessed
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("traces/sess-123.json")
        mock_blob.upload_from_string.assert_called_once()

        # Verify result
        assert result.success is True
        assert result.destination == "gs://test-bucket/traces/sess-123.json"
        assert result.metadata["bucket"] == "test-bucket"
        assert result.metadata["blob_path"] == "traces/sess-123.json"

    def test_export_upload_content(self):
        """Test that the exported JSON contains correct data."""
        import json

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        exporter = GCSExporter(bucket="test-bucket")
        exporter._client = mock_client

        data = _create_mock_export(session_id="test-sess")
        exporter.export(data)

        # Get the uploaded content
        call_args = mock_blob.upload_from_string.call_args
        uploaded_bytes = call_args[0][0]
        uploaded_data = json.loads(uploaded_bytes.decode("utf-8"))

        assert len(uploaded_data["sessions"]) == 1
        assert uploaded_data["sessions"][0]["id"] == "test-sess"
        assert uploaded_data["version"] == 1

    def test_export_missing_dependency(self):
        exporter = GCSExporter(bucket="test-bucket")
        exporter._client = None  # Ensure client needs to be created

        # Patch the import to fail
        with patch.dict("sys.modules", {"google.cloud": None, "google.cloud.storage": None}):
            with pytest.raises(ExportError, match="google-cloud-storage is required"):
                exporter._get_client()

    def test_from_env_missing_bucket(self, monkeypatch):
        monkeypatch.delenv("AIOBS_GCS_BUCKET", raising=False)
        with pytest.raises(ExportError, match="AIOBS_GCS_BUCKET is required"):
            GCSExporter.from_env()

    def test_from_env_success(self, monkeypatch):
        monkeypatch.setenv("AIOBS_GCS_BUCKET", "env-bucket")
        monkeypatch.setenv("AIOBS_GCS_PREFIX", "env-prefix/")
        monkeypatch.setenv("AIOBS_GCS_PROJECT", "env-project")

        exporter = GCSExporter.from_env()
        assert exporter.bucket == "env-bucket"
        assert exporter.prefix == "env-prefix/"
        assert exporter.project == "env-project"

    def test_export_with_custom_metadata(self):
        """Test that custom metadata is set on the blob."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        exporter = GCSExporter(bucket="test-bucket")
        exporter._client = mock_client

        data = _create_mock_export()

        custom_meta = {"environment": "production", "version": "1.0"}
        exporter.export(data, metadata=custom_meta)

        # Verify metadata was set on the blob
        assert mock_blob.metadata == custom_meta

    def test_export_failure(self):
        """Test that export failure raises ExportError."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.upload_from_string.side_effect = Exception("Upload failed")

        exporter = GCSExporter(bucket="test-bucket")
        exporter._client = mock_client

        data = _create_mock_export()

        with pytest.raises(ExportError, match="Failed to export to GCS"):
            exporter.export(data)

    def test_export_with_filename_override(self):
        """Test that filename kwarg overrides template."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        exporter = GCSExporter(
            bucket="test-bucket",
            prefix="traces/",
            filename_template="{session_id}.json",
        )
        exporter._client = mock_client

        data = _create_mock_export(session_id="original-session")
        exporter.export(data, filename="custom-filename.json")

        # Should use the override, not the template
        mock_bucket.blob.assert_called_once_with("traces/custom-filename.json")


# =============================================================================
# Integration Tests with Observer
# =============================================================================

class TestExporterIntegration:
    def test_flush_with_custom_exporter(self):
        exported_data = []

        def handler(data, options):
            exported_data.append(data)
            return ExportResult(
                success=True,
                destination="test://captured",
                bytes_written=len(json.dumps(data.model_dump())),
            )

        exporter = CustomExporter(handler=handler)

        # Use the observer
        observer.observe("integration-test")

        # Record a synthetic event
        ev = ObsEvent(
            provider="test",
            api="test.call",
            request={"input": "hello"},
            response={"output": "world"},
            error=None,
            started_at=0.0,
            ended_at=1.0,
            duration_ms=1000.0,
        )
        observer._record_event(ev)
        observer.end()

        # Flush with exporter
        result = observer.flush(exporter=exporter)

        assert result.success is True
        assert result.destination == "test://captured"
        assert len(exported_data) == 1
        assert len(exported_data[0].sessions) == 1
        assert len(exported_data[0].events) == 1

    def test_flush_with_composite_exporter(self):
        results = {"exp1": [], "exp2": []}

        def make_handler(name):
            def handler(data, options):
                results[name].append(data)
                return ExportResult(success=True, destination=f"{name}://done")
            return handler

        exp1 = CustomExporter(handler=make_handler("exp1"), name="exp1")
        exp2 = CustomExporter(handler=make_handler("exp2"), name="exp2")
        composite = CompositeExporter([exp1, exp2])

        observer.observe("composite-test")
        observer.end()
        result = observer.flush(exporter=composite)

        assert result.success is True
        assert len(results["exp1"]) == 1
        assert len(results["exp2"]) == 1

    def test_flush_with_decorated_functions(self):
        exported_data = []

        def handler(data, options):
            exported_data.append(data)
            return ExportResult(success=True)

        @observe
        def outer_func():
            return inner_func()

        @observe
        def inner_func():
            return 42

        exporter = CustomExporter(handler=handler)

        observer.observe("decorated-test")
        outer_func()
        observer.end()
        observer.flush(exporter=exporter)

        assert len(exported_data) == 1
        data = exported_data[0]
        assert len(data.function_events) == 2
        # Verify trace tree structure
        assert data.trace_tree is not None
        assert len(data.trace_tree) == 1
        assert data.trace_tree[0]["name"] == "outer_func"
        assert len(data.trace_tree[0]["children"]) == 1
        assert data.trace_tree[0]["children"][0]["name"] == "inner_func"

    def test_flush_clears_data_after_export(self):
        call_count = [0]

        def handler(data, options):
            call_count[0] += 1
            return ExportResult(success=True)

        exporter = CustomExporter(handler=handler)

        # First session
        observer.observe("session-1")
        observer.end()
        observer.flush(exporter=exporter)
        assert call_count[0] == 1

        # Second session - should start fresh
        observer.observe("session-2")
        observer.end()
        result = observer.flush(exporter=exporter)

        assert call_count[0] == 2
        assert result.success is True

    def test_flush_with_persist_false_no_file(self, tmp_path):
        """Test that flush with persist=False does not create a file."""
        import os

        observer.observe("persist-false-exporter")

        # Record a synthetic event
        ev = ObsEvent(
            provider="test",
            api="test.call",
            request={"input": "hello"},
            response={"output": "world"},
            error=None,
            started_at=0.0,
            ended_at=1.0,
            duration_ms=1000.0,
        )
        observer._record_event(ev)
        observer.end()

        # Flush with persist=False
        out_path = tmp_path / "should_not_exist.json"
        result = observer.flush(path=str(out_path), persist=False)

        # File should NOT be created
        assert not os.path.exists(out_path)
        assert result is None

    def test_flush_with_exporter_and_persist_false(self, tmp_path):
        """Test that exporter still works when persist=False is set."""
        import os

        exported_data = []

        def handler(data, options):
            exported_data.append(data)
            return ExportResult(success=True, destination="test://exported")

        exporter = CustomExporter(handler=handler)

        observer.observe("exporter-persist-false")

        ev = ObsEvent(
            provider="test",
            api="test.call",
            request={"input": "data"},
            response={"output": "result"},
            error=None,
            started_at=0.0,
            ended_at=1.0,
            duration_ms=1000.0,
        )
        observer._record_event(ev)
        observer.end()

        # Flush with exporter (persist is ignored when exporter is provided)
        out_path = tmp_path / "with_exporter.json"
        result = observer.flush(path=str(out_path), exporter=exporter, persist=False)

        # Exporter should have been called
        assert len(exported_data) == 1
        assert result.success is True
        assert result.destination == "test://exported"


# =============================================================================
# Helper Functions
# =============================================================================

def _create_mock_export(session_id: str = "test-session") -> ObservabilityExport:
    """Create a mock ObservabilityExport for testing."""
    return ObservabilityExport(
        sessions=[
            Session(
                id=session_id,
                name="Test Session",
                started_at=0.0,
                ended_at=1.0,
                meta=SessionMeta(pid=12345, cwd="/test"),
            )
        ],
        events=[],
        function_events=[],
        generated_at=1.0,
    )

