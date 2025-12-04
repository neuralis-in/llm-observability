"""OpenAI LLM adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseLLM, LLMMessage, LLMResponse


class OpenAILLM(BaseLLM):
    """LLM adapter for OpenAI and OpenAI-compatible APIs.
    
    Works with:
    - OpenAI
    - Azure OpenAI
    - Groq
    - Together AI
    - Any OpenAI-compatible API
    
    Example:
        from openai import OpenAI
        from aiobs.llm import LLM
        
        client = OpenAI()
        llm = LLM.from_client(client, model="gpt-4o")
        response = llm.complete("Hello!")
    """
    
    provider: str = "openai"
    
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize OpenAI LLM adapter.
        
        Args:
            client: OpenAI client instance.
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        super().__init__(client, model, temperature, max_tokens)
    
    @classmethod
    def is_compatible(cls, client: Any) -> bool:
        """Check if client is OpenAI-compatible.
        
        Args:
            client: Client instance to check.
            
        Returns:
            True if client has OpenAI-compatible interface.
        """
        return (
            hasattr(client, "chat") 
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        )
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build messages list for API call.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            
        Returns:
            List of message dicts.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse.
        
        Args:
            response: Raw OpenAI response.
            
        Returns:
            Parsed LLMResponse.
        """
        content = response.choices[0].message.content or ""
        
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return LLMResponse(
            content=content,
            model=response.model,
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
        messages = self._build_messages(prompt, system_prompt)
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        
        call_kwargs.update(kwargs)
        
        response = self.client.chat.completions.create(**call_kwargs)
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
        messages = self._build_messages(prompt, system_prompt)
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        
        call_kwargs.update(kwargs)
        
        # Check if client has async support
        if hasattr(self.client.chat.completions, "acreate"):
            response = await self.client.chat.completions.acreate(**call_kwargs)
        else:
            # Fallback: run sync in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(**call_kwargs)
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
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        
        call_kwargs.update(kwargs)
        
        response = self.client.chat.completions.create(**call_kwargs)
        return self._parse_response(response)

