"""Hallucination detection evaluator using LLM-as-judge."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type

from ..base import BaseEval
from ..models import (
    EvalInput,
    EvalResult,
    EvalStatus,
    HallucinationDetectionConfig,
    AssertionDetail,
)
from ...llm import LLM, BaseLLM


# System prompt for the hallucination judge
HALLUCINATION_JUDGE_SYSTEM_PROMPT = """You are an expert fact-checker and hallucination detector. Your task is to analyze an AI model's output and determine if it contains hallucinations (fabricated, false, or unsupported information).

A hallucination is:
- Information that is factually incorrect
- Claims not supported by the provided context/documents
- Made-up details, names, dates, statistics, or quotes
- Confident statements about things that cannot be verified

You must be thorough and precise in your analysis."""

# Prompt template for hallucination detection
HALLUCINATION_JUDGE_PROMPT = """Analyze the following AI model output for hallucinations.

## User Question/Input:
{user_input}

{context_section}

## Model Output to Evaluate:
{model_output}

## Your Task:
1. Identify any hallucinations in the model output
2. For each hallucination found, explain why it's a hallucination
3. Provide a hallucination score from 0.0 to 1.0 where:
   - 1.0 = No hallucinations, fully accurate and grounded
   - 0.7-0.9 = Minor issues (slight inaccuracies or unsupported but plausible claims)
   - 0.4-0.6 = Moderate hallucinations (some fabricated details)
   - 0.1-0.3 = Significant hallucinations (major fabrications)
   - 0.0 = Completely hallucinated/false

Respond in the following JSON format:
```json
{{
    "score": <float between 0.0 and 1.0>,
    "has_hallucinations": <true or false>,
    "hallucinations": [
        {{
            "claim": "<the hallucinated claim>",
            "reason": "<why this is a hallucination>",
            "severity": "<minor|moderate|severe>"
        }}
    ],
    "analysis": "<brief overall analysis>"
}}
```

Respond ONLY with the JSON, no additional text."""


class HallucinationDetectionEval(BaseEval):
    """Evaluator that detects hallucinations in model outputs using LLM-as-judge.
    
    This evaluator uses another LLM to analyze model outputs and identify
    hallucinations - fabricated, false, or unsupported information.
    
    Example:
        from openai import OpenAI
        from aiobs.evals import HallucinationDetectionEval, EvalInput
        
        # Create evaluator with OpenAI client
        client = OpenAI()
        evaluator = HallucinationDetectionEval(client=client, model="gpt-4o")
        
        # Evaluate model output
        result = evaluator.evaluate(
            EvalInput(
                user_input="What is the capital of France?",
                model_output="Paris is the capital of France. It was founded in 250 BC by Julius Caesar.",
                context={"documents": ["Paris is the capital and largest city of France."]}
            )
        )
        
        print(result.status)  # EvalStatus.FAILED (hallucination detected)
        print(result.score)   # 0.5 (moderate hallucination)
    """
    
    name: str = "hallucination_detection"
    description: str = "Detects hallucinations in model outputs using LLM-as-judge"
    config_class: Type[HallucinationDetectionConfig] = HallucinationDetectionConfig
    
    def __init__(
        self,
        client: Any,
        model: str,
        config: Optional[HallucinationDetectionConfig] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize the hallucination detection evaluator.
        
        Args:
            client: LLM provider client (OpenAI, Gemini, Anthropic, etc.).
            model: Model name to use for hallucination detection.
            config: Optional configuration for evaluation behavior.
            temperature: Temperature for the judge LLM (overrides config).
            max_tokens: Maximum tokens for judge response.
        """
        super().__init__(config)
        self.config: HallucinationDetectionConfig = self.config
        
        # Create LLM adapter from client
        self._llm: BaseLLM = LLM.from_client(
            client=client,
            model=model,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens,
        )
        
        # Store model name for logging
        if self.config.model is None:
            self.config.model = model
    
    @classmethod
    def with_openai(
        cls,
        client: Any,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> "HallucinationDetectionEval":
        """Create evaluator with an OpenAI client.
        
        Args:
            client: OpenAI client instance.
            model: Model name (default: gpt-4o-mini).
            **kwargs: Additional config options.
            
        Returns:
            Configured HallucinationDetectionEval instance.
        """
        config = HallucinationDetectionConfig(**kwargs) if kwargs else None
        return cls(client=client, model=model, config=config)
    
    @classmethod
    def with_gemini(
        cls,
        client: Any,
        model: str = "gemini-2.0-flash",
        **kwargs: Any,
    ) -> "HallucinationDetectionEval":
        """Create evaluator with a Gemini client.
        
        Args:
            client: Google GenAI client instance.
            model: Model name (default: gemini-2.0-flash).
            **kwargs: Additional config options.
            
        Returns:
            Configured HallucinationDetectionEval instance.
        """
        config = HallucinationDetectionConfig(**kwargs) if kwargs else None
        return cls(client=client, model=model, config=config)
    
    @classmethod
    def with_anthropic(
        cls,
        client: Any,
        model: str = "claude-3-sonnet-20240229",
        **kwargs: Any,
    ) -> "HallucinationDetectionEval":
        """Create evaluator with an Anthropic client.
        
        Args:
            client: Anthropic client instance.
            model: Model name (default: claude-3-sonnet-20240229).
            **kwargs: Additional config options.
            
        Returns:
            Configured HallucinationDetectionEval instance.
        """
        config = HallucinationDetectionConfig(**kwargs) if kwargs else None
        return cls(client=client, model=model, config=config)
    
    def _build_prompt(self, eval_input: EvalInput) -> str:
        """Build the evaluation prompt.
        
        Args:
            eval_input: The input to evaluate.
            
        Returns:
            Formatted prompt string.
        """
        # Build context section
        context_section = ""
        if self.config.check_against_context and eval_input.context:
            context_text = self._format_context(eval_input.context)
            if context_text:
                context_section = f"## Provided Context/Documents:\n{context_text}\n"
        
        return HALLUCINATION_JUDGE_PROMPT.format(
            user_input=eval_input.user_input,
            context_section=context_section,
            model_output=eval_input.model_output,
        )
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary into a string.
        
        Args:
            context: Context dictionary.
            
        Returns:
            Formatted context string.
        """
        parts = []
        
        # Handle common context keys
        if "documents" in context:
            docs = context["documents"]
            if isinstance(docs, list):
                for i, doc in enumerate(docs, 1):
                    parts.append(f"Document {i}:\n{doc}")
            else:
                parts.append(f"Documents:\n{docs}")
        
        if "sources" in context:
            sources = context["sources"]
            if isinstance(sources, list):
                for i, src in enumerate(sources, 1):
                    parts.append(f"Source {i}:\n{src}")
            else:
                parts.append(f"Sources:\n{sources}")
        
        # Handle any other keys
        for key, value in context.items():
            if key not in ("documents", "sources"):
                if isinstance(value, (list, dict)):
                    parts.append(f"{key}:\n{json.dumps(value, indent=2)}")
                else:
                    parts.append(f"{key}:\n{value}")
        
        return "\n\n".join(parts)
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the judge LLM's response.
        
        Args:
            response_text: Raw response from the judge LLM.
            
        Returns:
            Parsed response dictionary.
        """
        # Try to extract JSON from the response
        try:
            # First try direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in the text
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: return error structure
        return {
            "score": 0.5,
            "has_hallucinations": False,
            "hallucinations": [],
            "analysis": f"Failed to parse judge response: {response_text[:200]}",
            "parse_error": True,
        }
    
    def evaluate(self, eval_input: EvalInput, **kwargs: Any) -> EvalResult:
        """Evaluate model output for hallucinations.
        
        Args:
            eval_input: Input containing model_output to check.
            **kwargs: Additional arguments (unused).
            
        Returns:
            EvalResult indicating presence/absence of hallucinations.
        """
        try:
            # Build and send prompt to judge LLM
            prompt = self._build_prompt(eval_input)
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=HALLUCINATION_JUDGE_SYSTEM_PROMPT,
            )
            
            # Parse response
            parsed = self._parse_response(response.content)
            
            return self._build_result(parsed, eval_input)
            
        except Exception as e:
            return EvalResult.error_result(
                eval_name=self.eval_name,
                error=e,
            )
    
    async def evaluate_async(self, eval_input: EvalInput, **kwargs: Any) -> EvalResult:
        """Evaluate model output for hallucinations asynchronously.
        
        Args:
            eval_input: Input containing model_output to check.
            **kwargs: Additional arguments (unused).
            
        Returns:
            EvalResult indicating presence/absence of hallucinations.
        """
        try:
            # Build and send prompt to judge LLM
            prompt = self._build_prompt(eval_input)
            response = await self._llm.complete_async(
                prompt=prompt,
                system_prompt=HALLUCINATION_JUDGE_SYSTEM_PROMPT,
            )
            
            # Parse response
            parsed = self._parse_response(response.content)
            
            return self._build_result(parsed, eval_input)
            
        except Exception as e:
            return EvalResult.error_result(
                eval_name=self.eval_name,
                error=e,
            )
    
    def _build_result(
        self,
        parsed: Dict[str, Any],
        eval_input: EvalInput,
    ) -> EvalResult:
        """Build EvalResult from parsed judge response.
        
        Args:
            parsed: Parsed response from judge LLM.
            eval_input: Original evaluation input.
            
        Returns:
            EvalResult with hallucination analysis.
        """
        score = float(parsed.get("score", 0.5))
        has_hallucinations = parsed.get("has_hallucinations", score < self.config.hallucination_threshold)
        hallucinations = parsed.get("hallucinations", [])
        analysis = parsed.get("analysis", "")
        
        # Determine pass/fail
        if self.config.strict:
            passed = not has_hallucinations
        else:
            passed = score >= self.config.hallucination_threshold
        
        # Build assertions for each hallucination found
        assertions: List[AssertionDetail] = []
        
        if hallucinations:
            for h in hallucinations[:self.config.max_claims]:
                assertions.append(AssertionDetail(
                    name=f"hallucination:{h.get('severity', 'unknown')}",
                    passed=False,
                    expected="Factual, grounded claim",
                    actual=h.get("claim", "Unknown claim"),
                    message=h.get("reason", "Hallucination detected"),
                ))
        else:
            assertions.append(AssertionDetail(
                name="no_hallucinations",
                passed=True,
                expected="No hallucinations",
                actual="No hallucinations detected",
                message="Output appears to be grounded and factual",
            ))
        
        # Build message
        if has_hallucinations:
            severity_counts = {}
            for h in hallucinations:
                sev = h.get("severity", "unknown")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            message = f"Hallucinations detected: {len(hallucinations)} issue(s) - {severity_counts}"
        else:
            message = "No hallucinations detected"
        
        # Build details
        details: Dict[str, Any] = {
            "score": score,
            "has_hallucinations": has_hallucinations,
            "hallucination_count": len(hallucinations),
            "analysis": analysis,
            "judge_model": self._llm.model,
            "threshold": self.config.hallucination_threshold,
        }
        
        if hallucinations and self.config.include_details:
            details["hallucinations"] = hallucinations
        
        if parsed.get("parse_error"):
            details["parse_error"] = True
        
        return EvalResult(
            status=EvalStatus.PASSED if passed else EvalStatus.FAILED,
            score=score,
            eval_name=self.eval_name,
            message=message,
            assertions=assertions if self.config.include_details else None,
            details=details if self.config.include_details else None,
        )

