from __future__ import annotations

import inspect
import os
import time
from typing import Any, Callable, Dict, Optional

from .base_api import BaseGeminiAPIModule
from .models import (
    GenerateVideosRequest,
    GenerateVideosResponse,
)
from ....models import Event as ObsEvent, Callsite as ObsCallsite


class GenerateVideosAPI(BaseGeminiAPIModule):
    name = "models.generate_videos"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from google.genai.models import Models  # noqa: F401
            return hasattr(Models, "generate_videos")
        except Exception:
            return False

    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        try:
            from google.genai.models import Models as _Models  # type: ignore
        except Exception:
            return None

        if not hasattr(_Models, "generate_videos"):
            return None

        original_generate_videos = _Models.generate_videos

        def _extract_request(args: tuple, kwargs: dict) -> GenerateVideosRequest:
            req: Dict[str, Any] = {}
            # generate_videos signature: generate_videos(self, *, model, prompt=None, image=None, video=None, config=None, ...)
            # args[0] is self (the Models instance)
            if args:
                req["_args"] = args[1:]  # Skip self
            req.update(kwargs or {})
            
            model = req.get("model")
            prompt = req.get("prompt")
            image = req.get("image")
            video = req.get("video")
            config = req.get("config")
            
            # Convert config to dict if it's a pydantic model or has model_dump
            config_dict = None
            if config is not None:
                try:
                    if hasattr(config, "model_dump"):
                        config_dict = config.model_dump()
                    elif hasattr(config, "dict"):
                        config_dict = config.dict()
                    elif isinstance(config, dict):
                        config_dict = config
                except Exception:
                    config_dict = str(config)
            
            # Simplify image for storage (avoid storing raw bytes)
            image_simplified = None
            if image is not None:
                try:
                    if hasattr(image, "model_dump"):
                        image_simplified = image.model_dump()
                    elif hasattr(image, "dict"):
                        image_simplified = image.dict()
                    elif isinstance(image, dict):
                        image_simplified = {k: v for k, v in image.items() if k != "image_bytes"}
                    else:
                        image_simplified = {"type": type(image).__name__}
                except Exception:
                    image_simplified = {"type": str(type(image))}
            
            # Simplify video for storage (avoid storing raw bytes)
            video_simplified = None
            if video is not None:
                try:
                    if hasattr(video, "model_dump"):
                        video_simplified = video.model_dump()
                    elif hasattr(video, "dict"):
                        video_simplified = video.dict()
                    elif isinstance(video, dict):
                        video_simplified = {k: v for k, v in video.items() if k != "video_bytes"}
                    else:
                        video_simplified = {"type": type(video).__name__}
                except Exception:
                    video_simplified = {"type": str(type(video))}
            
            return GenerateVideosRequest(
                model=model,
                prompt=prompt,
                image=image_simplified,
                video=video_simplified,
                config=config_dict,
                other={k: v for k, v in req.items() 
                       if k not in {"model", "prompt", "image", "video", "config", "_args"}},
            )

        def _extract_response(resp: Any) -> GenerateVideosResponse:
            rmodel = getattr(resp, "model", None)
            operation_name = getattr(resp, "name", None)
            done = getattr(resp, "done", None)
            generated_videos = None
            
            # Extract generated videos from response
            try:
                response_obj = getattr(resp, "response", None)
                if response_obj is not None and hasattr(response_obj, "generated_videos"):
                    gen_videos = response_obj.generated_videos
                    if gen_videos:
                        generated_videos = []
                        for vid in gen_videos:
                            vid_dict = {}
                            if hasattr(vid, "model_dump"):
                                vid_dict = vid.model_dump()
                            elif hasattr(vid, "dict"):
                                vid_dict = vid.dict()
                            elif isinstance(vid, dict):
                                vid_dict = vid
                            # Remove large binary data if present
                            if "video" in vid_dict and isinstance(vid_dict["video"], dict):
                                vid_dict["video"] = {k: v for k, v in vid_dict["video"].items() 
                                                     if k not in {"video_bytes", "image_bytes"}}
                            generated_videos.append(vid_dict)
            except Exception:
                pass
            
            return GenerateVideosResponse(
                model=rmodel,
                operation_name=operation_name,
                done=done,
                generated_videos=generated_videos,
            )

        def _get_callsite() -> Optional[ObsCallsite]:
            try:
                frames = inspect.stack()[2:]
                for fi in frames:
                    fname = os.path.abspath(fi.filename)
                    if (
                        f"{os.sep}aiobs{os.sep}" in fname
                        or f"{os.sep}site-packages{os.sep}google{os.sep}" in fname
                        or "google/genai" in fname
                    ):
                        continue
                    try:
                        rel = os.path.relpath(fname, start=os.getcwd())
                    except Exception:
                        rel = fname
                    return ObsCallsite(file=rel, line=fi.lineno, function=fi.function)
            except Exception:
                pass
            return None

        def wrapped(self, *args, **kwargs):  # type: ignore[no-redef]
            import uuid
            started = time.time()
            request_info = _extract_request((self,) + args, kwargs)
            callsite = _get_callsite()
            error: Optional[str] = None
            response_info: Optional[GenerateVideosResponse] = None
            
            # Capture parent span context for trace tree linking
            parent_span_id = collector.get_current_span_id()
            span_id = str(uuid.uuid4())
            
            try:
                resp = original_generate_videos(self, *args, **kwargs)
                response_info = _extract_response(resp)
                return resp
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                ended = time.time()
                event = ObsEvent(
                    provider="gemini",
                    api="models.generate_videos",
                    callsite=callsite,
                    request=request_info,
                    response=response_info,
                    error=error,
                    started_at=started,
                    ended_at=ended,
                    duration_ms=round((ended - started) * 1000, 3),
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                )
                collector._record_event(event)

        setattr(_Models, "generate_videos", wrapped)  # type: ignore[attr-defined]

        def unpatch() -> None:
            try:
                setattr(_Models, "generate_videos", original_generate_videos)  # type: ignore[attr-defined]
            except Exception:
                pass

        return unpatch

