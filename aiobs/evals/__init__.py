"""Evaluation framework for aiobs.

This module provides a comprehensive evaluation framework for assessing
LLM outputs across multiple dimensions: correctness, safety, reliability,
and performance.

Usage:
    from aiobs.evals import RegexAssertion, EvalInput
    
    # Create an evaluator
    evaluator = RegexAssertion.from_patterns(
        patterns=[r".*Paris.*"],
        case_sensitive=False
    )
    
    # Create input and evaluate
    eval_input = EvalInput(
        user_input="What is the capital of France?",
        model_output="The capital of France is Paris."
    )
    
    result = evaluator(eval_input)
    print(result.status)  # EvalStatus.PASSED

Available Evaluators:
    - RegexAssertion: Check output against regex patterns
    - SchemaAssertion: Validate JSON output against JSON Schema
    - GroundTruthEval: Compare output to expected ground truth
    - HallucinationDetectionEval: Detect hallucinations using LLM-as-judge
    - LatencyConsistencyEval: Check latency statistics
    - PIIDetectionEval: Detect personally identifiable information
"""

from __future__ import annotations

# Base class
from .base import BaseEval

# Models
from .models import (
    EvalInput,
    EvalResult,
    EvalStatus,
    AssertionDetail,
    # Configs
    BaseEvalConfig,
    RegexAssertionConfig,
    SchemaAssertionConfig,
    GroundTruthConfig,
    GroundTruthMatchMode,
    LatencyConsistencyConfig,
    PIIDetectionConfig,
    PIIType,
    HallucinationDetectionConfig,
    SQLQueryValidatorConfig,
)

# Correctness evaluators
from .correctness import (
    RegexAssertion,
    SchemaAssertion,
    GroundTruthEval,
    HallucinationDetectionEval,
    SQLQueryValidator,
)

# Reliability evaluators
from .reliability import (
    LatencyConsistencyEval,
)

# Safety evaluators
from .safety import (
    PIIDetectionEval,
)

__all__ = [
    # Base
    "BaseEval",
    # Models
    "EvalInput",
    "EvalResult",
    "EvalStatus",
    "AssertionDetail",
    # Configs
    "BaseEvalConfig",
    "RegexAssertionConfig",
    "SchemaAssertionConfig",
    "GroundTruthConfig",
    "GroundTruthMatchMode",
    "LatencyConsistencyConfig",
    "PIIDetectionConfig",
    "PIIType",
    "HallucinationDetectionConfig",
    "SQLQueryValidatorConfig",
    # Correctness evaluators
    "RegexAssertion",
    "SchemaAssertion",
    "GroundTruthEval",
    "HallucinationDetectionEval",
    "SQLQueryValidator",
    # Reliability evaluators
    "LatencyConsistencyEval",
    # Safety evaluators
    "PIIDetectionEval",
]

