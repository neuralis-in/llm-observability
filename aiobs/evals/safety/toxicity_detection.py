from typing import Any, Dict, List, Optional, Type
import json

from aiobs.evals.base import BaseEval
from aiobs.evals.models.configs import ToxicityDetectionConfig
from aiobs.evals.types import EvalInput, EvalResult, EvalStatus
class ToxicityDetectionEval(BaseEval):
    """Evaluator that detects toxic content in model outputs using LLM-as-judge.
    
    This evaluator uses another LLM to analyze model outputs and identify
    toxic content including hate speech, harassment, violence, profanity,
    and discriminatory content.
    
    Example:
        from openai import OpenAI
        from aiobs.evals import ToxicityDetectionEval, EvalInput
        
        client = OpenAI()
        evaluator = ToxicityDetectionEval(client=client, model="gpt-4o")
        
        result = evaluator.evaluate(
            EvalInput(
                user_input="Write a response",
                model_output="This is a sample response to evaluate."
            )
        )
        
        print(result.status)  # EvalStatus.PASSED or FAILED
        print(result.score)   # 0.0 (toxic) to 1.0 (non-toxic)
    """
    
    name: str = "toxicity_detection"
    description: str = "Detects toxic content in model outputs using LLM-as-judge"
    config_class: Type[ToxicityDetectionConfig] = ToxicityDetectionConfig