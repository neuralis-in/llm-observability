import json
import os
from types import SimpleNamespace
import pytest

pytest.importorskip("google.genai")

from aiobs import observer, observe


def test_gemini_generate_videos_instrumentation(monkeypatch, tmp_path):
    """Test that Gemini generate_videos calls are captured."""
    from google.genai.models import Models

    def fake_generate_videos(self, *args, **kwargs):  # noqa: ARG001
        # Create a fake operation response matching Gemini's structure
        video = SimpleNamespace(
            uri="gs://bucket/video.mp4",
            mime_type="video/mp4",
        )
        generated_video = SimpleNamespace(video=video)
        response = SimpleNamespace(generated_videos=[generated_video])
        return SimpleNamespace(
            name="operations/generate-videos-123",
            done=False,
            response=response,
        )

    # Monkeypatch BEFORE observe() so the provider wraps our fake
    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos, raising=True)

    # Start observer (installs provider instrumentation)
    observer.observe("gemini-video-instrumentation")

    # Create a minimal fake Models instance and call generate_videos
    fake_models = Models.__new__(Models)
    _ = fake_models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt="A cinematic shot of waves crashing on a beach at sunset",
    )

    # Flush and verify event captured
    out_path = tmp_path / "obs.json"
    observer.end()
    written = observer.flush(str(out_path))
    assert os.path.exists(written)

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured by Gemini instrumentation"
    
    ev = events[0]
    assert ev["provider"] == "gemini"
    assert ev["api"] == "models.generate_videos"
    assert ev["request"]["model"] == "veo-3.1-generate-preview"
    assert ev["request"]["prompt"] == "A cinematic shot of waves crashing on a beach at sunset"
    assert ev["response"]["operation_name"] == "operations/generate-videos-123"
    assert ev["response"]["done"] is False
    assert ev["span_id"] is not None
    assert ev["duration_ms"] >= 0


def test_gemini_generate_videos_with_config(monkeypatch, tmp_path):
    """Test that video generation config is captured properly."""
    from google.genai.models import Models

    def fake_generate_videos(self, *args, **kwargs):  # noqa: ARG001
        video = SimpleNamespace(uri="gs://bucket/video.mp4", mime_type="video/mp4")
        generated_video = SimpleNamespace(video=video)
        response = SimpleNamespace(generated_videos=[generated_video])
        return SimpleNamespace(
            name="operations/generate-videos-456",
            done=True,
            response=response,
        )

    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos, raising=True)

    observer.observe("gemini-video-config-test")

    fake_models = Models.__new__(Models)
    _ = fake_models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt="A majestic lion in the savannah",
        config={
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "number_of_videos": 1,
        }
    )

    out_path = tmp_path / "obs.json"
    observer.end()
    observer.flush(str(out_path))

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured"
    
    ev = events[0]
    assert ev["request"]["config"] == {
        "aspect_ratio": "16:9",
        "resolution": "720p",
        "number_of_videos": 1,
    }
    assert ev["response"]["done"] is True


def test_gemini_generate_videos_with_image(monkeypatch, tmp_path):
    """Test that image input is captured (without binary data)."""
    from google.genai.models import Models

    def fake_generate_videos(self, *args, **kwargs):  # noqa: ARG001
        video = SimpleNamespace(uri="gs://bucket/video.mp4", mime_type="video/mp4")
        generated_video = SimpleNamespace(video=video)
        response = SimpleNamespace(generated_videos=[generated_video])
        return SimpleNamespace(
            name="operations/generate-videos-789",
            done=False,
            response=response,
        )

    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos, raising=True)

    observer.observe("gemini-video-image-test")

    fake_models = Models.__new__(Models)
    _ = fake_models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt="Animate this image",
        image={
            "gcs_uri": "gs://bucket/input_image.png",
            "mime_type": "image/png",
        }
    )

    out_path = tmp_path / "obs.json"
    observer.end()
    observer.flush(str(out_path))

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured"
    
    ev = events[0]
    assert ev["request"]["image"] == {
        "gcs_uri": "gs://bucket/input_image.png",
        "mime_type": "image/png",
    }


def test_gemini_generate_videos_error(monkeypatch, tmp_path):
    """Test that errors are captured when generate_videos fails."""
    from google.genai.models import Models

    def fake_generate_videos_error(self, *args, **kwargs):  # noqa: ARG001
        raise ValueError("Video generation quota exceeded")

    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos_error, raising=True)

    observer.observe("gemini-video-error-test")

    fake_models = Models.__new__(Models)
    with pytest.raises(ValueError, match="Video generation quota exceeded"):
        fake_models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt="Generate a video"
        )

    out_path = tmp_path / "obs.json"
    observer.end()
    observer.flush(str(out_path))

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured"
    
    ev = events[0]
    assert ev["provider"] == "gemini"
    assert ev["api"] == "models.generate_videos"
    assert ev["error"] is not None
    assert "ValueError" in ev["error"]
    assert "Video generation quota exceeded" in ev["error"]


def test_gemini_generate_videos_parent_span_linking(monkeypatch, tmp_path):
    """Test that generate_videos calls inside @observe functions get parent_span_id."""
    from google.genai.models import Models

    def fake_generate_videos(self, *args, **kwargs):  # noqa: ARG001
        video = SimpleNamespace(uri="gs://bucket/video.mp4", mime_type="video/mp4")
        generated_video = SimpleNamespace(video=video)
        response = SimpleNamespace(generated_videos=[generated_video])
        return SimpleNamespace(
            name="operations/generate-videos-span",
            done=False,
            response=response,
        )

    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos, raising=True)

    observer.observe("gemini-video-parent-span-test")

    @observe
    def my_function_that_generates_video():
        fake_models = Models.__new__(Models)
        return fake_models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt="A sunset timelapse"
        )

    _ = my_function_that_generates_video()

    out_path = tmp_path / "obs.json"
    observer.end()
    observer.flush(str(out_path))

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    function_events = data.get("function_events", [])
    
    assert events, "No provider events captured"
    assert function_events, "No function events captured"
    
    # The function event should have a span_id
    func_ev = function_events[0]
    assert func_ev["span_id"] is not None
    func_span_id = func_ev["span_id"]
    
    # The video generation event should have parent_span_id matching function's span_id
    video_ev = events[0]
    assert video_ev["parent_span_id"] == func_span_id


def test_gemini_generate_videos_in_trace_tree(monkeypatch, tmp_path):
    """Test that generate_videos events appear in trace_tree with correct structure."""
    from google.genai.models import Models

    def fake_generate_videos(self, *args, **kwargs):  # noqa: ARG001
        video = SimpleNamespace(uri="gs://bucket/video.mp4", mime_type="video/mp4")
        generated_video = SimpleNamespace(video=video)
        response = SimpleNamespace(generated_videos=[generated_video])
        return SimpleNamespace(
            name="operations/generate-videos-tree",
            done=True,
            response=response,
        )

    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos, raising=True)

    observer.observe("gemini-video-trace-tree-test")

    @observe
    def video_generation_pipeline():
        fake_models = Models.__new__(Models)
        return fake_models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt="Aurora borealis dancing"
        )

    _ = video_generation_pipeline()

    out_path = tmp_path / "obs.json"
    observer.end()
    observer.flush(str(out_path))

    data = json.loads(out_path.read_text())
    trace_tree = data.get("trace_tree", [])
    
    assert trace_tree, "No trace tree generated"
    
    # Find the function node (should be a root)
    func_node = None
    for node in trace_tree:
        if node.get("event_type") == "function":
            func_node = node
            break
    
    assert func_node is not None, "Function node not found in trace tree"
    assert func_node["name"] == "video_generation_pipeline"
    
    # Check that video generation event is a child
    children = func_node.get("children", [])
    assert children, "Function node has no children"
    
    video_child = children[0]
    assert video_child["event_type"] == "provider"
    assert video_child["provider"] == "gemini"
    assert video_child["api"] == "models.generate_videos"


def test_gemini_generate_videos_video_extension(monkeypatch, tmp_path):
    """Test that video extension (input video) is captured."""
    from google.genai.models import Models

    def fake_generate_videos(self, *args, **kwargs):  # noqa: ARG001
        video = SimpleNamespace(uri="gs://bucket/extended_video.mp4", mime_type="video/mp4")
        generated_video = SimpleNamespace(video=video)
        response = SimpleNamespace(generated_videos=[generated_video])
        return SimpleNamespace(
            name="operations/extend-video-123",
            done=False,
            response=response,
        )

    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos, raising=True)

    observer.observe("gemini-video-extension-test")

    fake_models = Models.__new__(Models)
    _ = fake_models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt="Continue the scene with the butterfly landing",
        video={
            "gcs_uri": "gs://bucket/original_video.mp4",
            "mime_type": "video/mp4",
        },
        config={
            "number_of_videos": 1,
            "resolution": "720p",
        }
    )

    out_path = tmp_path / "obs.json"
    observer.end()
    observer.flush(str(out_path))

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured"
    
    ev = events[0]
    assert ev["request"]["video"] == {
        "gcs_uri": "gs://bucket/original_video.mp4",
        "mime_type": "video/mp4",
    }
    assert ev["request"]["prompt"] == "Continue the scene with the butterfly landing"


def test_gemini_generate_videos_generated_videos_capture(monkeypatch, tmp_path):
    """Test that generated videos info is captured in response."""
    from google.genai.models import Models

    def fake_generate_videos(self, *args, **kwargs):  # noqa: ARG001
        video1 = SimpleNamespace(
            uri="gs://bucket/video1.mp4",
            mime_type="video/mp4",
        )
        video2 = SimpleNamespace(
            uri="gs://bucket/video2.mp4",
            mime_type="video/mp4",
        )
        generated_video1 = SimpleNamespace(video=video1)
        generated_video2 = SimpleNamespace(video=video2)
        response = SimpleNamespace(generated_videos=[generated_video1, generated_video2])
        return SimpleNamespace(
            name="operations/multi-video-123",
            done=True,
            response=response,
        )

    # Mock model_dump for SimpleNamespace objects
    def mock_model_dump(self):
        return {k: getattr(v, 'model_dump', lambda: v.__dict__)() 
                if hasattr(v, '__dict__') and not isinstance(v, str) else v 
                for k, v in self.__dict__.items()}
    
    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos, raising=True)

    observer.observe("gemini-video-generated-capture-test")

    fake_models = Models.__new__(Models)
    _ = fake_models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt="Generate two variations",
        config={"number_of_videos": 2}
    )

    out_path = tmp_path / "obs.json"
    observer.end()
    observer.flush(str(out_path))

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured"
    
    ev = events[0]
    assert ev["response"]["done"] is True
    assert ev["response"]["operation_name"] == "operations/multi-video-123"


def test_gemini_generate_videos_pydantic_config(monkeypatch, tmp_path):
    """Test that pydantic config objects are properly serialized."""
    from google.genai.models import Models

    # Create a mock pydantic-like config object
    class MockConfig:
        def __init__(self):
            self.aspect_ratio = "16:9"
            self.resolution = "1080p"
            self.generate_audio = True
        
        def model_dump(self):
            return {
                "aspect_ratio": self.aspect_ratio,
                "resolution": self.resolution,
                "generate_audio": self.generate_audio,
            }

    def fake_generate_videos(self, *args, **kwargs):  # noqa: ARG001
        video = SimpleNamespace(uri="gs://bucket/video.mp4", mime_type="video/mp4")
        generated_video = SimpleNamespace(video=video)
        response = SimpleNamespace(generated_videos=[generated_video])
        return SimpleNamespace(
            name="operations/pydantic-config-123",
            done=False,
            response=response,
        )

    monkeypatch.setattr(Models, "generate_videos", fake_generate_videos, raising=True)

    observer.observe("gemini-video-pydantic-config-test")

    fake_models = Models.__new__(Models)
    _ = fake_models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt="High quality video",
        config=MockConfig()
    )

    out_path = tmp_path / "obs.json"
    observer.end()
    observer.flush(str(out_path))

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured"
    
    ev = events[0]
    assert ev["request"]["config"] == {
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "generate_audio": True,
    }

