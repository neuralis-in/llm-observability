"""LLM abstraction layer for aiobs.

This module provides a unified interface for interacting with different
LLM providers. It's primarily used internally by LLM-based evaluators
like HallucinationDetectionEval.

Example:
    from openai import OpenAI
    from aiobs.llm import LLM
    
    # Create LLM from client (auto-detects provider)
    client = OpenAI()
    llm = LLM.from_client(client, model="gpt-4o")
    
    # Generate completion
    response = llm.complete("What is 2+2?")
    print(response.content)  # "4"
    
    # Async usage
    response = await llm.complete_async("What is 2+2?")

Supported Providers:
    - OpenAI (and OpenAI-compatible APIs: Azure, Groq, Together, etc.)
    - Google Gemini
    - Anthropic Claude
"""

from __future__ import annotations

from .base import BaseLLM, LLMMessage, LLMResponse
from .factory import LLM
from .openai import OpenAILLM
from .gemini import GeminiLLM
from .anthropic import AnthropicLLM

__all__ = [
    # Factory
    "LLM",
    # Base classes
    "BaseLLM",
    "LLMMessage",
    "LLMResponse",
    # Provider adapters
    "OpenAILLM",
    "GeminiLLM",
    "AnthropicLLM",
]

