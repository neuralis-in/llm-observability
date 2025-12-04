"""Anthropic Claude LLM adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseLLM, LLMMessage, LLMResponse


class AnthropicLLM(BaseLLM):
    """LLM adapter for Anthropic Claude API.
    
    Example:
        from anthropic import Anthropic
        from aiobs.llm import LLM
        
        client = Anthropic()
        llm = LLM.from_client(client, model="claude-3-sonnet-20240229")
        response = llm.complete("Hello!")
    """
    
    provider: str = "anthropic"
    
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 1024,
    ) -> None:
        """Initialize Anthropic LLM adapter.
        
        Args:
            client: Anthropic client instance.
            model: Model name (e.g., "claude-3-sonnet-20240229").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (required for Anthropic).
        """
        # Anthropic requires max_tokens
        super().__init__(client, model, temperature, max_tokens or 1024)
    
    @classmethod
    def is_compatible(cls, client: Any) -> bool:
        """Check if client is Anthropic-compatible.
        
        Args:
            client: Client instance to check.
            
        Returns:
            True if client has Anthropic-compatible interface.
        """
        return (
            hasattr(client, "messages") 
            and hasattr(client.messages, "create")
        )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response into LLMResponse.
        
        Args:
            response: Raw Anthropic response.
            
        Returns:
            Parsed LLMResponse.
        """
        # Extract text content
        content = ""
        if hasattr(response, "content") and response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
        
        # Extract usage
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        
        return LLMResponse(
            content=content,
            model=response.model if hasattr(response, "model") else self.model,
            usage=usage,
            raw_response=response,
        )
    
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
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            LLMResponse with generated content.
        """
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if system_prompt:
            call_kwargs["system"] = system_prompt
        
        call_kwargs.update(kwargs)
        
        response = self.client.messages.create(**call_kwargs)
        return self._parse_response(response)
    
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
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            LLMResponse with generated content.
        """
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if system_prompt:
            call_kwargs["system"] = system_prompt
        
        call_kwargs.update(kwargs)
        
        # Check for async method
        if hasattr(self.client.messages, "acreate"):
            response = await self.client.messages.acreate(**call_kwargs)
        else:
            # Fallback: run sync in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(**call_kwargs)
            )
        
        return self._parse_response(response)
    
    def complete_messages(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from a list of messages.
        
        Args:
            messages: List of conversation messages.
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            LLMResponse with generated content.
        """
        system_prompt = None
        api_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if system_prompt:
            call_kwargs["system"] = system_prompt
        
        call_kwargs.update(kwargs)
        
        response = self.client.messages.create(**call_kwargs)
        return self._parse_response(response)

