"""Base LLM interface for aiobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """A message in a conversation."""
    
    role: str = Field(description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(description="Message content")


class LLMResponse(BaseModel):
    """Response from an LLM completion."""
    
    content: str = Field(description="The generated text content")
    model: str = Field(description="Model that generated the response")
    usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage: prompt_tokens, completion_tokens, total_tokens"
    )
    raw_response: Optional[Any] = Field(
        default=None,
        description="Raw response object from the provider"
    )


class BaseLLM(ABC):
    """Abstract base class for LLM adapters.
    
    Provides a unified interface for different LLM providers.
    """
    
    provider: str = "base"
    
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize the LLM adapter.
        
        Args:
            client: The provider's client instance.
            model: Model name/identifier.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion synchronously.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            LLMResponse with generated content.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion asynchronously.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            LLMResponse with generated content.
        """
        raise NotImplementedError
    
    def complete_messages(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from a list of messages.
        
        Args:
            messages: List of conversation messages.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            LLMResponse with generated content.
        """
        # Default implementation: extract system and user prompts
        system_prompt = None
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                prompt_parts.append(f"{msg.role}: {msg.content}")
        
        prompt = "\n".join(prompt_parts) if prompt_parts else ""
        return self.complete(prompt, system_prompt=system_prompt, **kwargs)
    
    async def complete_messages_async(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from messages asynchronously.
        
        Args:
            messages: List of conversation messages.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            LLMResponse with generated content.
        """
        system_prompt = None
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                prompt_parts.append(f"{msg.role}: {msg.content}")
        
        prompt = "\n".join(prompt_parts) if prompt_parts else ""
        return await self.complete_async(prompt, system_prompt=system_prompt, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, provider={self.provider!r})"

