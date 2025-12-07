"""Configuration models for evaluators."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Pattern
import re

from pydantic import BaseModel, Field, field_validator


class BaseEvalConfig(BaseModel):
    """Base configuration for all evaluators."""
    
    name: Optional[str] = Field(
        default=None,
        description="Custom name for this evaluator instance"
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop evaluation on first failure (for multi-assertion evals)"
    )
    include_details: bool = Field(
        default=True,
        description="Include detailed information in results"
    )


class RegexAssertionConfig(BaseEvalConfig):
    """Configuration for regex assertion evaluator."""
    
    patterns: List[str] = Field(
        default_factory=list,
        description="List of regex patterns that output must match"
    )
    negative_patterns: List[str] = Field(
        default_factory=list,
        description="List of regex patterns that output must NOT match"
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether regex matching is case-sensitive"
    )
    match_mode: str = Field(
        default="any",
        description="Match mode: 'any' (at least one pattern matches) or 'all' (all patterns must match)"
    )
    
    @field_validator("match_mode")
    @classmethod
    def validate_match_mode(cls, v: str) -> str:
        """Validate match_mode is 'any' or 'all'."""
        if v not in ("any", "all"):
            raise ValueError("match_mode must be 'any' or 'all'")
        return v
    
    def get_compiled_patterns(self) -> List[Pattern[str]]:
        """Get compiled regex patterns.
        
        Returns:
            List of compiled regex Pattern objects.
        """
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return [re.compile(p, flags) for p in self.patterns]
    
    def get_compiled_negative_patterns(self) -> List[Pattern[str]]:
        """Get compiled negative regex patterns.
        
        Returns:
            List of compiled regex Pattern objects.
        """
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return [re.compile(p, flags) for p in self.negative_patterns]


class SchemaAssertionConfig(BaseEvalConfig):
    """Configuration for JSON schema assertion evaluator."""
    
    json_schema: Dict[str, Any] = Field(
        description="JSON Schema to validate the output against"
    )
    strict: bool = Field(
        default=True,
        description="Whether to fail on additional properties not in schema"
    )
    parse_json: bool = Field(
        default=True,
        description="Whether to parse the output as JSON before validation"
    )
    extract_json: bool = Field(
        default=True,
        description="Try to extract JSON from markdown code blocks if direct parse fails"
    )


class GroundTruthMatchMode(str, Enum):
    """Match modes for ground truth comparison."""
    
    EXACT = "exact"
    CONTAINS = "contains"
    NORMALIZED = "normalized"  # Whitespace/case normalized
    SEMANTIC = "semantic"  # Placeholder for future embedding-based comparison


class GroundTruthConfig(BaseEvalConfig):
    """Configuration for ground truth comparison evaluator."""
    
    match_mode: GroundTruthMatchMode = Field(
        default=GroundTruthMatchMode.NORMALIZED,
        description="How to compare output with expected: exact, contains, normalized, or semantic"
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether comparison is case-sensitive"
    )
    normalize_whitespace: bool = Field(
        default=True,
        description="Whether to normalize whitespace before comparison"
    )
    strip_punctuation: bool = Field(
        default=False,
        description="Whether to strip punctuation before comparison"
    )
    similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for fuzzy matching (0-1)"
    )


class LatencyConsistencyConfig(BaseEvalConfig):
    """Configuration for latency consistency evaluator."""
    
    max_latency_ms: Optional[float] = Field(
        default=None,
        description="Maximum acceptable latency in milliseconds"
    )
    max_std_dev_ms: Optional[float] = Field(
        default=None,
        description="Maximum acceptable standard deviation in milliseconds"
    )
    max_p95_ms: Optional[float] = Field(
        default=None,
        description="Maximum acceptable 95th percentile latency in milliseconds"
    )
    max_p99_ms: Optional[float] = Field(
        default=None,
        description="Maximum acceptable 99th percentile latency in milliseconds"
    )
    coefficient_of_variation_threshold: float = Field(
        default=0.5,
        ge=0.0,
        description="Maximum acceptable coefficient of variation (std_dev / mean)"
    )


class PIIType(str, Enum):
    """Types of PII to detect."""
    
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    NAME = "name"  # Requires NER, not implemented in basic version
    CUSTOM = "custom"


class PIIDetectionConfig(BaseEvalConfig):
    """Configuration for PII detection evaluator."""
    
    detect_types: List[PIIType] = Field(
        default_factory=lambda: [
            PIIType.EMAIL,
            PIIType.PHONE,
            PIIType.SSN,
            PIIType.CREDIT_CARD,
        ],
        description="Types of PII to detect"
    )
    custom_patterns: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom regex patterns for PII detection (name -> pattern)"
    )
    redact: bool = Field(
        default=False,
        description="Whether to include redacted version in results"
    )
    fail_on_detection: bool = Field(
        default=True,
        description="Whether to fail the eval if PII is detected"
    )
    check_input: bool = Field(
        default=False,
        description="Also check user input for PII"
    )
    check_system_prompt: bool = Field(
        default=False,
        description="Also check system prompt for PII"
    )


class HallucinationDetectionConfig(BaseEvalConfig):
    """Configuration for hallucination detection evaluator.
    
    Uses an LLM-as-judge approach to detect hallucinations in model outputs.
    The evaluator checks if the model output contains fabricated information
    that is not supported by the provided context.
    """
    
    model: Optional[str] = Field(
        default=None,
        description="Model name for the judge LLM (for logging purposes)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for the judge LLM (lower = more deterministic)"
    )
    check_against_context: bool = Field(
        default=True,
        description="Check if output is grounded in provided context"
    )
    check_against_input: bool = Field(
        default=True,
        description="Check if output is relevant to the user input"
    )
    strict: bool = Field(
        default=False,
        description="Strict mode: any hallucination results in failure"
    )
    hallucination_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Score threshold below which output is considered hallucinated (0-1)"
    )
    extract_claims: bool = Field(
        default=True,
        description="Extract and evaluate individual claims from the output"
    )
    max_claims: int = Field(
        default=10,
        ge=1,
        description="Maximum number of claims to extract and evaluate"
    )


class SQLQueryValidatorConfig(BaseEvalConfig):
    """Configuration for SQL query validation evaluator.
    
    Validates that model output is a syntactically correct SQL query
    using sqlglot. Supports multiple SQL dialects for dialect-specific
    validation.
    """
    
    dialect: Optional[str] = Field(
        default=None,
        description="SQL dialect for parsing (postgres, mysql, sqlite, etc.). "
                    "If None, uses default sqlglot parsing."
    )

