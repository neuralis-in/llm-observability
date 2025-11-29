from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .base import BaseGeminiRequest, BaseGeminiResponse


class VideoGenerationConfig(BaseModel):
    """Configuration for video generation."""
    aspect_ratio: Optional[str] = Field(default=None)
    number_of_videos: Optional[int] = Field(default=None)
    resolution: Optional[str] = Field(default=None)
    duration_seconds: Optional[int] = Field(default=None)
    negative_prompt: Optional[str] = Field(default=None)
    generate_audio: Optional[bool] = Field(default=None)
    enhance_prompt: Optional[bool] = Field(default=None)
    person_generation: Optional[str] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    output_gcs_uri: Optional[str] = Field(default=None)


class GenerateVideosRequest(BaseGeminiRequest):
    """Request model for generate_videos API."""
    prompt: Optional[str] = Field(default=None)
    image: Optional[Dict[str, Any]] = Field(default=None)
    video: Optional[Dict[str, Any]] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)
    other: Dict[str, Any] = Field(default_factory=dict)


class GeneratedVideo(BaseModel):
    """A generated video result."""
    video: Optional[Dict[str, Any]] = Field(default=None)


class GenerateVideosResponse(BaseGeminiResponse):
    """Response model for generate_videos API."""
    operation_name: Optional[str] = Field(default=None)
    done: Optional[bool] = Field(default=None)
    generated_videos: Optional[List[Dict[str, Any]]] = Field(default=None)

