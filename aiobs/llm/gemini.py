"""Google Gemini LLM adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseLLM, LLMMessage, LLMResponse


class GeminiLLM(BaseLLM):
    """LLM adapter for Google Gemini API.
    
    Example:
        from google import genai
        from aiobs.llm import LLM
        
        client = genai.Client()
        llm = LLM.from_client(client, model="gemini-2.0-flash")
        response = llm.complete("Hello!")
    """
    
    provider: str = "gemini"
    
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize Gemini LLM adapter.
        
        Args:
            client: Google GenAI client instance.
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        super().__init__(client, model, temperature, max_tokens)
    
    @classmethod
    def is_compatible(cls, client: Any) -> bool:
        """Check if client is Gemini-compatible.
        
        Args:
            client: Client instance to check.
            
        Returns:
            True if client has Gemini-compatible interface.
        """
        return (
            hasattr(client, "models") 
            and hasattr(client.models, "generate_content")
        )
    
    def _build_config(self) -> Dict[str, Any]:
        """Build generation config.
        
        Returns:
            Config dict for Gemini API.
        """
        config: Dict[str, Any] = {
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            config["max_output_tokens"] = self.max_tokens
        
        return config
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Gemini response into LLMResponse.
        
        Args:
            response: Raw Gemini response.
            
        Returns:
            Parsed LLMResponse.
        """
        # Extract text content
        content = ""
        if hasattr(response, "text"):
            content = response.text
        elif hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                if hasattr(candidate.content, "parts") and candidate.content.parts:
                    content = candidate.content.parts[0].text
        
        # Extract usage if available
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }
        
        return LLMResponse(
            content=content,
            model=self.model,
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
        config = self._build_config()
        config.update(kwargs.pop("generation_config", {}))
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "contents": prompt,
            "config": config,
        }
        
        if system_prompt:
            call_kwargs["config"]["system_instruction"] = system_prompt
        
        call_kwargs.update(kwargs)
        
        response = self.client.models.generate_content(**call_kwargs)
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
        config = self._build_config()
        config.update(kwargs.pop("generation_config", {}))
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "contents": prompt,
            "config": config,
        }
        
        if system_prompt:
            call_kwargs["config"]["system_instruction"] = system_prompt
        
        call_kwargs.update(kwargs)
        
        # Check for async method
        if hasattr(self.client.models, "generate_content_async"):
            response = await self.client.models.generate_content_async(**call_kwargs)
        else:
            # Fallback: run sync in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(**call_kwargs)
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
        # Extract system prompt and build contents
        system_prompt = None
        contents = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})
        
        config = self._build_config()
        config.update(kwargs.pop("generation_config", {}))
        
        if system_prompt:
            config["system_instruction"] = system_prompt
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "contents": contents,
            "config": config,
        }
        
        call_kwargs.update(kwargs)
        
        response = self.client.models.generate_content(**call_kwargs)
        return self._parse_response(response)

