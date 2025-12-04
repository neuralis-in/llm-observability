"""LLM factory for auto-detecting and creating LLM adapters."""

from __future__ import annotations

from typing import Any, Optional, Type

from .base import BaseLLM
from .openai import OpenAILLM
from .gemini import GeminiLLM
from .anthropic import AnthropicLLM


# Registry of LLM adapters in order of detection priority
_LLM_ADAPTERS: list[Type[BaseLLM]] = [
    OpenAILLM,
    GeminiLLM,
    AnthropicLLM,
]


class LLM:
    """Factory class for creating LLM adapters.
    
    Provides a unified interface for different LLM providers through
    automatic client detection or explicit provider specification.
    
    Example:
        from openai import OpenAI
        from aiobs.llm import LLM
        
        # Auto-detect from client
        client = OpenAI()
        llm = LLM.from_client(client, model="gpt-4o")
        
        response = llm.complete("What is 2+2?")
        print(response.content)  # "4"
        
        # Async usage
        response = await llm.complete_async("What is 2+2?")
    """
    
    @staticmethod
    def from_client(
        client: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> BaseLLM:
        """Create an LLM adapter by auto-detecting the client type.
        
        Args:
            client: The LLM provider's client instance.
            model: Model name/identifier.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Appropriate LLM adapter instance.
            
        Raises:
            ValueError: If client type is not recognized.
            
        Example:
            from openai import OpenAI
            llm = LLM.from_client(OpenAI(), model="gpt-4o")
            
            from google import genai
            llm = LLM.from_client(genai.Client(), model="gemini-2.0-flash")
            
            from anthropic import Anthropic
            llm = LLM.from_client(Anthropic(), model="claude-3-sonnet-20240229")
        """
        for adapter_cls in _LLM_ADAPTERS:
            if adapter_cls.is_compatible(client):
                return adapter_cls(
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        
        # Try to provide helpful error message
        client_type = type(client).__name__
        module = type(client).__module__
        raise ValueError(
            f"Unrecognized client type: {module}.{client_type}. "
            f"Supported clients: OpenAI, Google GenAI (Gemini), Anthropic. "
            f"Make sure you're passing the client instance, not the module."
        )
    
    @staticmethod
    def openai(
        client: Any,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> OpenAILLM:
        """Create an OpenAI LLM adapter explicitly.
        
        Args:
            client: OpenAI client instance.
            model: Model name (default: "gpt-4o-mini").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            OpenAILLM adapter instance.
        """
        return OpenAILLM(
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    @staticmethod
    def gemini(
        client: Any,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> GeminiLLM:
        """Create a Gemini LLM adapter explicitly.
        
        Args:
            client: Google GenAI client instance.
            model: Model name (default: "gemini-2.0-flash").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            GeminiLLM adapter instance.
        """
        return GeminiLLM(
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    @staticmethod
    def anthropic(
        client: Any,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 1024,
    ) -> AnthropicLLM:
        """Create an Anthropic LLM adapter explicitly.
        
        Args:
            client: Anthropic client instance.
            model: Model name (default: "claude-3-sonnet-20240229").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (default: 1024).
            
        Returns:
            AnthropicLLM adapter instance.
        """
        return AnthropicLLM(
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

