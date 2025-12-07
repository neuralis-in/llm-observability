"""Tests for aiobs.evals module."""

import pytest
import json

from aiobs.evals import (
    # Base
    BaseEval,
    # Models
    EvalInput,
    EvalResult,
    EvalStatus,
    AssertionDetail,
    # Configs
    RegexAssertionConfig,
    SchemaAssertionConfig,
    GroundTruthConfig,
    GroundTruthMatchMode,
    LatencyConsistencyConfig,
    PIIDetectionConfig,
    PIIType,
    HallucinationDetectionConfig,
    SQLQueryValidatorConfig,
    # Evaluators
    RegexAssertion,
    SchemaAssertion,
    GroundTruthEval,
    LatencyConsistencyEval,
    PIIDetectionEval,
    HallucinationDetectionEval,
    SQLQueryValidator,
)
from aiobs.llm import LLM, BaseLLM, LLMResponse


# =============================================================================
# EvalInput Tests
# =============================================================================


class TestEvalInput:
    """Tests for EvalInput model."""

    def test_basic_creation(self):
        """Test basic EvalInput creation."""
        eval_input = EvalInput(
            user_input="What is 2+2?",
            model_output="4",
        )
        assert eval_input.user_input == "What is 2+2?"
        assert eval_input.model_output == "4"
        assert eval_input.system_prompt is None
        assert eval_input.expected_output is None

    def test_full_creation(self):
        """Test EvalInput with all fields."""
        eval_input = EvalInput(
            user_input="What is 2+2?",
            model_output="The answer is 4.",
            system_prompt="You are a math tutor.",
            expected_output="4",
            context={"topic": "math"},
            metadata={"latency_ms": 100},
            tags=["math", "simple"],
        )
        assert eval_input.system_prompt == "You are a math tutor."
        assert eval_input.expected_output == "4"
        assert eval_input.context == {"topic": "math"}
        assert eval_input.metadata == {"latency_ms": 100}
        assert eval_input.tags == ["math", "simple"]

    def test_with_expected(self):
        """Test with_expected helper."""
        eval_input = EvalInput(
            user_input="Question",
            model_output="Answer",
        )
        new_input = eval_input.with_expected("Expected Answer")
        
        # Original unchanged
        assert eval_input.expected_output is None
        # New input has expected
        assert new_input.expected_output == "Expected Answer"

    def test_with_metadata(self):
        """Test with_metadata helper."""
        eval_input = EvalInput(
            user_input="Q",
            model_output="A",
            metadata={"key1": "value1"},
        )
        new_input = eval_input.with_metadata(key2="value2", key3="value3")
        
        # Original unchanged
        assert eval_input.metadata == {"key1": "value1"}
        # New input has merged metadata
        assert new_input.metadata == {"key1": "value1", "key2": "value2", "key3": "value3"}


# =============================================================================
# EvalResult Tests
# =============================================================================


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_pass_result(self):
        """Test creating a passing result."""
        result = EvalResult.pass_result(
            eval_name="test_eval",
            message="Test passed",
        )
        assert result.status == EvalStatus.PASSED
        assert result.score == 1.0
        assert result.passed is True
        assert result.failed is False
        assert result.eval_name == "test_eval"

    def test_fail_result(self):
        """Test creating a failing result."""
        result = EvalResult.fail_result(
            eval_name="test_eval",
            score=0.3,
            message="Test failed",
        )
        assert result.status == EvalStatus.FAILED
        assert result.score == 0.3
        assert result.passed is False
        assert result.failed is True

    def test_error_result(self):
        """Test creating an error result."""
        error = ValueError("Something went wrong")
        result = EvalResult.error_result(
            eval_name="test_eval",
            error=error,
        )
        assert result.status == EvalStatus.ERROR
        assert result.score == 0.0
        assert "Something went wrong" in result.message
        assert result.details["error_type"] == "ValueError"

    def test_evaluated_at_populated(self):
        """Test that evaluated_at is auto-populated."""
        result = EvalResult.pass_result(eval_name="test")
        assert result.evaluated_at is not None


# =============================================================================
# RegexAssertion Tests
# =============================================================================


class TestRegexAssertion:
    """Tests for RegexAssertion evaluator."""

    def test_single_pattern_match(self):
        """Test matching a single pattern."""
        evaluator = RegexAssertion.from_patterns(patterns=[r"Paris"])
        result = evaluator(EvalInput(
            user_input="Capital of France?",
            model_output="The capital is Paris.",
        ))
        assert result.passed
        assert result.score == 1.0

    def test_single_pattern_no_match(self):
        """Test when pattern doesn't match."""
        evaluator = RegexAssertion.from_patterns(patterns=[r"London"])
        result = evaluator(EvalInput(
            user_input="Capital of France?",
            model_output="The capital is Paris.",
        ))
        assert result.failed
        assert result.score == 0.0

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        evaluator = RegexAssertion.from_patterns(
            patterns=[r"paris"],
            case_sensitive=False,
        )
        result = evaluator(EvalInput(
            user_input="Capital?",
            model_output="PARIS is the capital.",
        ))
        assert result.passed

    def test_case_sensitive_no_match(self):
        """Test case-sensitive matching fails on wrong case."""
        evaluator = RegexAssertion.from_patterns(
            patterns=[r"paris"],
            case_sensitive=True,
        )
        result = evaluator(EvalInput(
            user_input="Capital?",
            model_output="PARIS is the capital.",
        ))
        assert result.failed

    def test_multiple_patterns_any_mode(self):
        """Test 'any' mode - at least one pattern must match."""
        evaluator = RegexAssertion.from_patterns(
            patterns=[r"Paris", r"London", r"Berlin"],
            match_mode="any",
        )
        result = evaluator(EvalInput(
            user_input="Capital?",
            model_output="Paris is beautiful.",
        ))
        assert result.passed

    def test_multiple_patterns_all_mode(self):
        """Test 'all' mode - all patterns must match."""
        evaluator = RegexAssertion.from_patterns(
            patterns=[r"Paris", r"capital"],
            match_mode="all",
        )
        result = evaluator(EvalInput(
            user_input="Capital?",
            model_output="Paris is the capital.",
        ))
        assert result.passed

    def test_multiple_patterns_all_mode_fail(self):
        """Test 'all' mode fails when not all match."""
        evaluator = RegexAssertion.from_patterns(
            patterns=[r"Paris", r"London"],
            match_mode="all",
        )
        result = evaluator(EvalInput(
            user_input="Capital?",
            model_output="Paris is beautiful.",
        ))
        assert result.failed

    def test_negative_patterns(self):
        """Test negative patterns - must NOT match."""
        evaluator = RegexAssertion.from_patterns(
            negative_patterns=[r"sorry", r"cannot", r"unable"],
            case_sensitive=False,
        )
        result = evaluator(EvalInput(
            user_input="Help me",
            model_output="Here's the answer.",
        ))
        assert result.passed

    def test_negative_pattern_violation(self):
        """Test failure when negative pattern matches."""
        evaluator = RegexAssertion.from_patterns(
            negative_patterns=[r"sorry"],
            case_sensitive=False,
        )
        result = evaluator(EvalInput(
            user_input="Help me",
            model_output="Sorry, I cannot help.",
        ))
        assert result.failed

    def test_combined_positive_and_negative(self):
        """Test both positive and negative patterns."""
        evaluator = RegexAssertion.from_patterns(
            patterns=[r"\d+"],  # Must have a number
            negative_patterns=[r"error"],  # Must not have "error"
        )
        result = evaluator(EvalInput(
            user_input="Calculate",
            model_output="The result is 42.",
        ))
        assert result.passed

    def test_regex_pattern(self):
        """Test complex regex pattern."""
        evaluator = RegexAssertion.from_patterns(
            patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        )
        result = evaluator(EvalInput(
            user_input="Email?",
            model_output="Contact us at support@example.com",
        ))
        assert result.passed

    def test_assertions_in_result(self):
        """Test that assertion details are included."""
        config = RegexAssertionConfig(
            patterns=[r"hello", r"world"],
            include_details=True,
        )
        evaluator = RegexAssertion(config)
        result = evaluator(EvalInput(
            user_input="Greet",
            model_output="hello world",
        ))
        assert result.assertions is not None
        assert len(result.assertions) == 2
        assert all(a.passed for a in result.assertions)


# =============================================================================
# SchemaAssertion Tests
# =============================================================================


class TestSchemaAssertion:
    """Tests for SchemaAssertion evaluator."""

    def test_valid_json_matches_schema(self):
        """Test valid JSON that matches schema."""
        evaluator = SchemaAssertion.from_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        })
        result = evaluator(EvalInput(
            user_input="Extract person",
            model_output='{"name": "John", "age": 30}',
        ))
        assert result.passed

    def test_invalid_json(self):
        """Test invalid JSON fails."""
        evaluator = SchemaAssertion.from_schema({"type": "object"})
        result = evaluator(EvalInput(
            user_input="Extract",
            model_output="not valid json {",
        ))
        assert result.failed
        assert "parse" in result.message.lower()

    @pytest.mark.skipif(
        not SchemaAssertion.is_available(),
        reason="jsonschema not installed"
    )
    def test_missing_required_field(self):
        """Test missing required field fails."""
        evaluator = SchemaAssertion.from_schema({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        })
        result = evaluator(EvalInput(
            user_input="Extract",
            model_output='{"age": 30}',
        ))
        assert result.failed

    @pytest.mark.skipif(
        not SchemaAssertion.is_available(),
        reason="jsonschema not installed"
    )
    def test_wrong_type(self):
        """Test wrong type fails."""
        evaluator = SchemaAssertion.from_schema({
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        })
        result = evaluator(EvalInput(
            user_input="Extract",
            model_output='{"age": "thirty"}',
        ))
        assert result.failed

    def test_extract_json_from_markdown(self):
        """Test extracting JSON from markdown code block."""
        evaluator = SchemaAssertion.from_schema(
            {"type": "object"},
            extract_json=True,
        )
        result = evaluator(EvalInput(
            user_input="Extract",
            model_output='Here is the result:\n```json\n{"key": "value"}\n```',
        ))
        assert result.passed

    def test_array_schema(self):
        """Test array schema validation."""
        evaluator = SchemaAssertion.from_schema({
            "type": "array",
            "items": {"type": "string"},
        })
        result = evaluator(EvalInput(
            user_input="List",
            model_output='["a", "b", "c"]',
        ))
        assert result.passed

    def test_nested_schema(self):
        """Test nested object schema."""
        evaluator = SchemaAssertion.from_schema({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            "required": ["user"],
        })
        result = evaluator(EvalInput(
            user_input="Extract",
            model_output='{"user": {"name": "Alice"}}',
        ))
        assert result.passed


# =============================================================================
# GroundTruthEval Tests
# =============================================================================


class TestGroundTruthEval:
    """Tests for GroundTruthEval evaluator."""

    def test_exact_match_pass(self):
        """Test exact match succeeds."""
        evaluator = GroundTruthEval.exact(case_sensitive=True)
        result = evaluator(EvalInput(
            user_input="What is 2+2?",
            model_output="4",
            expected_output="4",
        ))
        assert result.passed
        assert result.score == 1.0

    def test_exact_match_fail(self):
        """Test exact match fails on different output."""
        evaluator = GroundTruthEval.exact(case_sensitive=True)
        result = evaluator(EvalInput(
            user_input="What is 2+2?",
            model_output="The answer is 4",
            expected_output="4",
        ))
        assert result.failed

    def test_contains_match_pass(self):
        """Test contains match succeeds."""
        evaluator = GroundTruthEval.contains()
        result = evaluator(EvalInput(
            user_input="What is 2+2?",
            model_output="The answer is 4.",
            expected_output="4",
        ))
        assert result.passed

    def test_contains_match_fail(self):
        """Test contains match fails."""
        evaluator = GroundTruthEval.contains()
        result = evaluator(EvalInput(
            user_input="Capital?",
            model_output="Berlin is a city.",
            expected_output="Paris",
        ))
        assert result.failed

    def test_normalized_match_whitespace(self):
        """Test normalized match handles whitespace."""
        evaluator = GroundTruthEval.normalized()
        result = evaluator(EvalInput(
            user_input="Quote?",
            model_output="Hello    world",
            expected_output="Hello world",
        ))
        assert result.passed

    def test_normalized_match_case(self):
        """Test normalized match handles case."""
        evaluator = GroundTruthEval.normalized(case_sensitive=False)
        result = evaluator(EvalInput(
            user_input="Name?",
            model_output="JOHN DOE",
            expected_output="John Doe",
        ))
        assert result.passed

    def test_expected_from_kwargs(self):
        """Test expected output can come from kwargs."""
        evaluator = GroundTruthEval.contains()
        result = evaluator(
            EvalInput(user_input="Q", model_output="Answer is 42"),
            expected="42",
        )
        assert result.passed

    def test_no_expected_output_error(self):
        """Test error when no expected output provided."""
        evaluator = GroundTruthEval.exact()
        result = evaluator(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        assert result.status == EvalStatus.ERROR

    def test_similarity_scoring(self):
        """Test similarity scoring for partial matches."""
        evaluator = GroundTruthEval.normalized()
        result = evaluator(EvalInput(
            user_input="Describe Paris",
            model_output="Paris is the capital of France, known for the Eiffel Tower.",
            expected_output="Paris is the capital of France.",
        ))
        # Check that similarity is calculated (Jaccard similarity)
        assert result.score > 0.0
        assert "similarity" in result.details


# =============================================================================
# LatencyConsistencyEval Tests
# =============================================================================


class TestLatencyConsistencyEval:
    """Tests for LatencyConsistencyEval evaluator."""

    def test_latencies_within_threshold(self):
        """Test latencies within max threshold pass."""
        evaluator = LatencyConsistencyEval.with_thresholds(max_latency_ms=1000)
        result = evaluator(EvalInput(
            user_input="test",
            model_output="response",
            metadata={"latencies": [100, 150, 120, 130, 110]},
        ))
        assert result.passed

    def test_latencies_exceed_threshold(self):
        """Test latencies exceeding max threshold fail."""
        evaluator = LatencyConsistencyEval.with_thresholds(max_latency_ms=100)
        result = evaluator(EvalInput(
            user_input="test",
            model_output="response",
            metadata={"latencies": [50, 60, 150, 70, 80]},
        ))
        assert result.failed

    def test_p95_check(self):
        """Test P95 latency check."""
        evaluator = LatencyConsistencyEval.with_thresholds(max_p95_ms=200)
        result = evaluator(EvalInput(
            user_input="test",
            model_output="response",
            metadata={"latencies": [100, 110, 120, 130, 180]},
        ))
        assert result.passed

    def test_coefficient_of_variation(self):
        """Test coefficient of variation check."""
        evaluator = LatencyConsistencyEval.with_thresholds(cv_threshold=0.2)
        
        # Low variance - should pass
        result = evaluator(EvalInput(
            user_input="test",
            model_output="response",
            metadata={"latencies": [100, 102, 98, 101, 99]},
        ))
        assert result.passed
        
        # High variance - should fail
        result = evaluator(EvalInput(
            user_input="test",
            model_output="response",
            metadata={"latencies": [50, 200, 75, 180, 100]},
        ))
        assert result.failed

    def test_latencies_from_kwargs(self):
        """Test latencies can come from kwargs."""
        evaluator = LatencyConsistencyEval.with_thresholds(max_latency_ms=500)
        result = evaluator(
            EvalInput(user_input="t", model_output="r"),
            latencies=[100, 120, 110],
        )
        assert result.passed

    def test_single_latency_value(self):
        """Test single latency value works."""
        evaluator = LatencyConsistencyEval.with_thresholds(max_latency_ms=500)
        result = evaluator(EvalInput(
            user_input="t",
            model_output="r",
            metadata={"latency_ms": 200},
        ))
        assert result.passed

    def test_statistics_in_details(self):
        """Test statistics are included in details."""
        config = LatencyConsistencyConfig(
            max_latency_ms=1000,
            include_details=True,
        )
        evaluator = LatencyConsistencyEval(config)
        result = evaluator(EvalInput(
            user_input="t",
            model_output="r",
            metadata={"latencies": [100, 200, 150]},
        ))
        assert result.details is not None
        assert "mean" in result.details
        assert "std_dev" in result.details
        assert "p95" in result.details

    def test_no_latencies_error(self):
        """Test error when no latencies provided."""
        evaluator = LatencyConsistencyEval()
        result = evaluator(EvalInput(user_input="t", model_output="r"))
        assert result.status == EvalStatus.ERROR


# =============================================================================
# PIIDetectionEval Tests
# =============================================================================


class TestPIIDetectionEval:
    """Tests for PIIDetectionEval evaluator."""

    def test_no_pii_passes(self):
        """Test output without PII passes."""
        evaluator = PIIDetectionEval.default()
        result = evaluator(EvalInput(
            user_input="Hello",
            model_output="Hi there! How can I help you?",
        ))
        assert result.passed
        assert result.details["pii_count"] == 0

    def test_email_detection(self):
        """Test email detection."""
        evaluator = PIIDetectionEval.default()
        result = evaluator(EvalInput(
            user_input="Contact info?",
            model_output="You can reach me at john.doe@example.com",
        ))
        assert result.failed
        assert "email" in result.details["pii_types_found"]

    def test_phone_detection(self):
        """Test phone number detection."""
        evaluator = PIIDetectionEval.default()
        result = evaluator(EvalInput(
            user_input="Phone?",
            model_output="Call me at 555-123-4567",
        ))
        assert result.failed
        assert "phone" in result.details["pii_types_found"]

    def test_ssn_detection(self):
        """Test SSN detection."""
        evaluator = PIIDetectionEval.default()
        result = evaluator(EvalInput(
            user_input="SSN?",
            model_output="My SSN is 123-45-6789",
        ))
        assert result.failed
        assert "ssn" in result.details["pii_types_found"]

    def test_credit_card_detection(self):
        """Test credit card detection."""
        evaluator = PIIDetectionEval.default()
        result = evaluator(EvalInput(
            user_input="Card?",
            model_output="Card number: 4111111111111111",
        ))
        assert result.failed
        assert "credit_card" in result.details["pii_types_found"]

    def test_ip_address_detection(self):
        """Test IP address detection."""
        config = PIIDetectionConfig(detect_types=[PIIType.IP_ADDRESS])
        evaluator = PIIDetectionEval(config)
        result = evaluator(EvalInput(
            user_input="IP?",
            model_output="Server IP: 192.168.1.100",
        ))
        assert result.failed
        assert "ip_address" in result.details["pii_types_found"]

    def test_multiple_pii_types(self):
        """Test detecting multiple PII types."""
        evaluator = PIIDetectionEval.default()
        result = evaluator(EvalInput(
            user_input="Contact info?",
            model_output="Email: test@test.com, Phone: 555-555-5555",
        ))
        assert result.failed
        assert result.details["pii_count"] >= 2

    def test_custom_patterns(self):
        """Test custom PII patterns."""
        evaluator = PIIDetectionEval.with_custom_patterns({
            "employee_id": r"EMP-\d{6}",
        })
        result = evaluator(EvalInput(
            user_input="ID?",
            model_output="Your employee ID is EMP-123456",
        ))
        assert result.failed
        assert "custom:employee_id" in result.details["pii_types_found"]

    def test_check_input_flag(self):
        """Test checking user input for PII."""
        config = PIIDetectionConfig(
            detect_types=[PIIType.EMAIL],
            check_input=True,
        )
        evaluator = PIIDetectionEval(config)
        result = evaluator(EvalInput(
            user_input="My email is user@example.com",
            model_output="Got it!",
        ))
        assert result.failed

    def test_check_system_prompt_flag(self):
        """Test checking system prompt for PII."""
        config = PIIDetectionConfig(
            detect_types=[PIIType.EMAIL],
            check_system_prompt=True,
        )
        evaluator = PIIDetectionEval(config)
        result = evaluator(EvalInput(
            user_input="Hello",
            model_output="Hi",
            system_prompt="Contact admin@company.com for help",
        ))
        assert result.failed

    def test_redaction(self):
        """Test PII redaction."""
        config = PIIDetectionConfig(
            detect_types=[PIIType.EMAIL],
            redact=True,
            include_details=True,
        )
        evaluator = PIIDetectionEval(config)
        result = evaluator(EvalInput(
            user_input="Email?",
            model_output="Contact test@example.com please",
        ))
        assert "redacted_output" in result.details
        assert "[EMAIL REDACTED]" in result.details["redacted_output"]

    def test_scan_method(self):
        """Test the scan convenience method."""
        evaluator = PIIDetectionEval.default()
        matches = evaluator.scan("Email: test@test.com, Phone: 555-123-4567")
        assert len(matches) >= 2

    def test_redact_method(self):
        """Test the redact convenience method."""
        evaluator = PIIDetectionEval.default()
        redacted = evaluator.redact("Contact test@example.com")
        assert "[EMAIL REDACTED]" in redacted
        assert "test@example.com" not in redacted

    def test_fail_on_detection_false(self):
        """Test fail_on_detection=False still passes but reports PII."""
        config = PIIDetectionConfig(
            detect_types=[PIIType.EMAIL],
            fail_on_detection=False,
        )
        evaluator = PIIDetectionEval(config)
        result = evaluator(EvalInput(
            user_input="Email?",
            model_output="test@example.com",
        ))
        assert result.passed  # Doesn't fail
        assert result.details["pii_count"] > 0  # But reports PII


# =============================================================================
# HallucinationDetectionEval Tests
# =============================================================================


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.chat = MockChat(response_content)


class MockChat:
    """Mock chat namespace."""
    
    def __init__(self, response_content: str):
        self.completions = MockCompletions(response_content)


class MockCompletions:
    """Mock completions endpoint."""
    
    def __init__(self, response_content: str):
        self.response_content = response_content
    
    def create(self, **kwargs):
        return MockResponse(self.response_content)


class MockResponse:
    """Mock API response."""
    
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]
        self.model = "gpt-4o-mini"
        self.usage = MockUsage()


class MockChoice:
    """Mock response choice."""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message."""
    
    def __init__(self, content: str):
        self.content = content


class MockUsage:
    """Mock usage stats."""
    prompt_tokens = 100
    completion_tokens = 50
    total_tokens = 150


class TestHallucinationDetectionEval:
    """Tests for HallucinationDetectionEval evaluator."""
    
    def _create_mock_response(self, score: float, has_hallucinations: bool, hallucinations: list, analysis: str) -> str:
        """Create a mock JSON response from the judge LLM."""
        return json.dumps({
            "score": score,
            "has_hallucinations": has_hallucinations,
            "hallucinations": hallucinations,
            "analysis": analysis,
        })
    
    def test_no_hallucination_passes(self, monkeypatch):
        """Test output without hallucinations passes."""
        mock_response = self._create_mock_response(
            score=1.0,
            has_hallucinations=False,
            hallucinations=[],
            analysis="The output is factually correct and grounded in the context.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="What is the capital of France?",
            model_output="The capital of France is Paris.",
            context={"documents": ["Paris is the capital of France."]},
        ))
        
        assert result.passed
        assert result.score == 1.0
        assert result.details["has_hallucinations"] is False
        assert result.details["hallucination_count"] == 0
    
    def test_hallucination_detected_fails(self, monkeypatch):
        """Test output with hallucinations fails."""
        mock_response = self._create_mock_response(
            score=0.3,
            has_hallucinations=True,
            hallucinations=[
                {
                    "claim": "Paris was founded in 250 BC by Julius Caesar",
                    "reason": "This is historically inaccurate. Paris was not founded by Julius Caesar.",
                    "severity": "severe",
                }
            ],
            analysis="The output contains a fabricated historical claim.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="What is the capital of France?",
            model_output="Paris is the capital of France. It was founded in 250 BC by Julius Caesar.",
            context={"documents": ["Paris is the capital of France."]},
        ))
        
        assert result.failed
        assert result.score == 0.3
        assert result.details["has_hallucinations"] is True
        assert result.details["hallucination_count"] == 1
    
    def test_multiple_hallucinations(self, monkeypatch):
        """Test detecting multiple hallucinations."""
        mock_response = self._create_mock_response(
            score=0.2,
            has_hallucinations=True,
            hallucinations=[
                {
                    "claim": "Written by Charles Dickens",
                    "reason": "Romeo and Juliet was written by Shakespeare, not Dickens.",
                    "severity": "severe",
                },
                {
                    "claim": "Written in 1920",
                    "reason": "The play was written between 1591-1596.",
                    "severity": "severe",
                },
                {
                    "claim": "Won the Nobel Prize",
                    "reason": "Nobel Prize didn't exist when the play was written.",
                    "severity": "severe",
                },
            ],
            analysis="Multiple severe factual errors detected.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Who wrote Romeo and Juliet?",
            model_output="Romeo and Juliet was written by Charles Dickens in 1920. It won the Nobel Prize.",
            context={"documents": ["Romeo and Juliet was written by William Shakespeare."]},
        ))
        
        assert result.failed
        assert result.score == 0.2
        assert result.details["hallucination_count"] == 3
    
    def test_strict_mode(self, monkeypatch):
        """Test strict mode fails on any hallucination."""
        # Minor hallucination with high score
        mock_response = self._create_mock_response(
            score=0.8,
            has_hallucinations=True,
            hallucinations=[
                {
                    "claim": "Minor unsupported detail",
                    "reason": "Cannot be verified from context.",
                    "severity": "minor",
                }
            ],
            analysis="Minor unsupported claim detected.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        config = HallucinationDetectionConfig(strict=True)
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
            config=config,
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Question",
            model_output="Answer with minor issue",
        ))
        
        # Even with high score, strict mode fails on any hallucination
        assert result.failed
    
    def test_custom_threshold(self, monkeypatch):
        """Test custom hallucination threshold."""
        mock_response = self._create_mock_response(
            score=0.6,
            has_hallucinations=True,
            hallucinations=[
                {
                    "claim": "Some claim",
                    "reason": "Minor issue",
                    "severity": "minor",
                }
            ],
            analysis="Moderate issues detected.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        # Low threshold - 0.6 should pass
        config_low = HallucinationDetectionConfig(hallucination_threshold=0.5)
        evaluator_low = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
            config=config_low,
        )
        result_low = evaluator_low.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        assert result_low.passed
        
        # High threshold - 0.6 should fail
        mock_client2 = MockOpenAIClient(mock_response)
        config_high = HallucinationDetectionConfig(hallucination_threshold=0.8)
        evaluator_high = HallucinationDetectionEval(
            client=mock_client2,
            model="gpt-4o-mini",
            config=config_high,
        )
        result_high = evaluator_high.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        assert result_high.failed
    
    def test_with_context_documents(self, monkeypatch):
        """Test hallucination check with context documents."""
        mock_response = self._create_mock_response(
            score=1.0,
            has_hallucinations=False,
            hallucinations=[],
            analysis="Output is grounded in provided documents.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Summarize the document",
            model_output="The document discusses AI safety.",
            context={
                "documents": [
                    "This paper examines AI safety considerations.",
                    "We propose new methods for alignment.",
                ]
            },
        ))
        
        assert result.passed
        assert "judge_model" in result.details
    
    def test_without_context(self, monkeypatch):
        """Test hallucination check without context (general factuality)."""
        mock_response = self._create_mock_response(
            score=0.9,
            has_hallucinations=False,
            hallucinations=[],
            analysis="Output appears factually correct based on general knowledge.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Who is the CEO of Apple?",
            model_output="Tim Cook is the CEO of Apple.",
        ))
        
        assert result.passed
    
    def test_assertions_in_result(self, monkeypatch):
        """Test that assertion details are included."""
        mock_response = self._create_mock_response(
            score=0.4,
            has_hallucinations=True,
            hallucinations=[
                {
                    "claim": "Hallucinated claim 1",
                    "reason": "Not supported",
                    "severity": "moderate",
                },
                {
                    "claim": "Hallucinated claim 2",
                    "reason": "Fabricated",
                    "severity": "severe",
                },
            ],
            analysis="Multiple hallucinations found.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        config = HallucinationDetectionConfig(include_details=True)
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
            config=config,
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        
        assert result.assertions is not None
        assert len(result.assertions) == 2
        assert all(not a.passed for a in result.assertions)
    
    def test_json_in_markdown_response(self, monkeypatch):
        """Test parsing JSON from markdown code block response."""
        # Response wrapped in markdown code block
        mock_response = '''Here's my analysis:

```json
{
    "score": 0.95,
    "has_hallucinations": false,
    "hallucinations": [],
    "analysis": "No issues found."
}
```'''
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        
        assert result.passed
        assert result.score == 0.95
    
    def test_malformed_response_handling(self, monkeypatch):
        """Test handling of malformed judge response."""
        mock_response = "This is not valid JSON at all"
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        
        # Should handle gracefully with fallback
        assert result.details.get("parse_error") is True
        assert result.score == 0.5  # Default fallback score
    
    def test_factory_method_with_openai(self, monkeypatch):
        """Test with_openai factory method."""
        mock_response = self._create_mock_response(
            score=1.0,
            has_hallucinations=False,
            hallucinations=[],
            analysis="All good.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval.with_openai(
            client=mock_client,
            model="gpt-4o",
            strict=True,
        )
        
        assert evaluator.config.strict is True
        result = evaluator.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        assert result.passed
    
    def test_evaluator_name(self, monkeypatch):
        """Test evaluator name in results."""
        mock_response = self._create_mock_response(
            score=1.0,
            has_hallucinations=False,
            hallucinations=[],
            analysis="OK",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        
        assert result.eval_name == "hallucination_detection"
    
    def test_custom_eval_name(self, monkeypatch):
        """Test custom eval name via config."""
        mock_response = self._create_mock_response(
            score=1.0,
            has_hallucinations=False,
            hallucinations=[],
            analysis="OK",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        config = HallucinationDetectionConfig(name="my_hallucination_check")
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
            config=config,
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        
        assert result.eval_name == "my_hallucination_check"
    
    def test_context_with_sources(self, monkeypatch):
        """Test context formatting with sources."""
        mock_response = self._create_mock_response(
            score=1.0,
            has_hallucinations=False,
            hallucinations=[],
            analysis="Grounded in sources.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="What does the source say?",
            model_output="The source discusses machine learning.",
            context={
                "sources": [
                    "Source 1: Introduction to ML",
                    "Source 2: Deep Learning basics",
                ],
                "metadata": {"retrieved_at": "2024-01-01"},
            },
        ))
        
        assert result.passed
    
    def test_severity_in_message(self, monkeypatch):
        """Test that severity counts appear in message."""
        mock_response = self._create_mock_response(
            score=0.3,
            has_hallucinations=True,
            hallucinations=[
                {"claim": "C1", "reason": "R1", "severity": "minor"},
                {"claim": "C2", "reason": "R2", "severity": "severe"},
                {"claim": "C3", "reason": "R3", "severity": "severe"},
            ],
            analysis="Multiple issues.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = evaluator.evaluate(EvalInput(
            user_input="Q",
            model_output="A",
        ))
        
        assert "3 issue(s)" in result.message
        assert "severe" in result.message or "minor" in result.message


def _has_pytest_asyncio():
    """Check if pytest-asyncio is available."""
    try:
        import pytest_asyncio
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_pytest_asyncio(), reason="pytest-asyncio not installed")
class TestHallucinationDetectionEvalAsync:
    """Async tests for HallucinationDetectionEval evaluator."""
    
    def _create_mock_response(self, score: float, has_hallucinations: bool, hallucinations: list, analysis: str) -> str:
        """Create a mock JSON response from the judge LLM."""
        return json.dumps({
            "score": score,
            "has_hallucinations": has_hallucinations,
            "hallucinations": hallucinations,
            "analysis": analysis,
        })
    
    @pytest.mark.asyncio
    async def test_async_evaluation(self, monkeypatch):
        """Test async evaluation works."""
        mock_response = self._create_mock_response(
            score=1.0,
            has_hallucinations=False,
            hallucinations=[],
            analysis="No hallucinations.",
        )
        mock_client = MockOpenAIClient(mock_response)
        
        evaluator = HallucinationDetectionEval(
            client=mock_client,
            model="gpt-4o-mini",
        )
        
        result = await evaluator.evaluate_async(EvalInput(
            user_input="What is 2+2?",
            model_output="4",
        ))
        
        assert result.passed
        assert result.score == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvalsIntegration:
    """Integration tests for evals module."""

    def test_evaluator_callable(self):
        """Test evaluators are callable."""
        evaluator = RegexAssertion.from_patterns(patterns=[r"test"])
        eval_input = EvalInput(user_input="q", model_output="test output")
        
        # Both __call__ and evaluate should work
        result1 = evaluator(eval_input)
        result2 = evaluator.evaluate(eval_input)
        
        assert result1.status == result2.status

    def test_duration_tracked(self):
        """Test duration_ms is tracked when using __call__."""
        evaluator = RegexAssertion.from_patterns(patterns=[r"test"])
        result = evaluator(EvalInput(user_input="q", model_output="test"))
        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    def test_batch_evaluation(self):
        """Test batch evaluation."""
        evaluator = RegexAssertion.from_patterns(patterns=[r"\d+"])
        inputs = [
            EvalInput(user_input="q1", model_output="123"),
            EvalInput(user_input="q2", model_output="no numbers"),
            EvalInput(user_input="q3", model_output="456"),
        ]
        
        results = evaluator.evaluate_batch(inputs)
        
        assert len(results) == 3
        assert results[0].passed
        assert results[1].failed
        assert results[2].passed

    def test_custom_eval_name(self):
        """Test custom eval name via config."""
        config = RegexAssertionConfig(
            name="my_custom_eval",
            patterns=[r"test"],
        )
        evaluator = RegexAssertion(config)
        result = evaluator(EvalInput(user_input="q", model_output="test"))
        
        assert result.eval_name == "my_custom_eval"

    def test_evaluator_repr(self):
        """Test evaluator string representation."""
        evaluator = RegexAssertion.from_patterns(patterns=[r"test"])
        repr_str = repr(evaluator)
        
        assert "RegexAssertion" in repr_str
        assert "regex_assertion" in repr_str


# =============================================================================
# SQLQueryValidator Tests
# =============================================================================


class TestSQLQueryValidator:
    """Tests for SQLQueryValidator evaluator."""

    def test_valid_basic_select(self):
        """Test valid basic SELECT query."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Get all users",
            model_output="SELECT * FROM users",
        ))
        assert result.passed
        assert result.score == 1.0
        assert "valid" in result.message.lower()

    def test_valid_complex_query(self):
        """Test valid complex query with joins and aggregates."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Get user order counts",
            model_output="""
                SELECT u.id, u.name, COUNT(o.id) as order_count
                FROM users u
                LEFT JOIN orders o ON u.id = o.user_id
                GROUP BY u.id, u.name
            """,
        ))
        assert result.passed

    def test_invalid_sql_syntax(self):
        """Test invalid SQL syntax fails."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Get all users",
            model_output="SELECT FROM WHERE",  # Invalid structure
        ))
        assert result.failed
        assert result.score == 0.0
        assert "invalid" in result.message.lower()

    def test_incomplete_query(self):
        """Test incomplete query fails."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Get users",
            model_output="SELECT * FROM",  # Incomplete
        ))
        assert result.failed

    def test_dialect_validation(self):
        """Test dialect-specific SQL validation."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator, SQLQueryValidatorConfig
        
        # Test PostgreSQL-specific syntax (INTERVAL)
        config = SQLQueryValidatorConfig(dialect="postgres")
        evaluator = SQLQueryValidator(config)
        
        result = evaluator(EvalInput(
            user_input="Get recent users",
            model_output="SELECT * FROM users WHERE created_at > NOW() - INTERVAL '7 days'",
        ))
        assert result.passed
        assert "postgres" in result.message.lower()

    def test_no_dialect_default(self):
        """Test validation without specific dialect."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Get users",
            model_output="SELECT * FROM users WHERE id > 100",
        ))
        assert result.passed
        assert "dialect" not in result.message.lower()

    def test_empty_query(self):
        """Test that empty query fails gracefully."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Query?",
            model_output="   ",  # Only whitespace
        ))
        assert result.failed

    def test_whitespace_handling(self):
        """Test query with extra whitespace."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Get users",
            model_output="  \n  SELECT   *   FROM   users  \n  ",
        ))
        assert result.passed

    def test_assertion_details(self):
        """Test that assertions contain proper details."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Get users",
            model_output="SELECT * FROM users",
        ))
        
        assert len(result.assertions) == 1
        assertion = result.assertions[0]
        assert assertion.name == "sql_parse"
        assert assertion.passed is True
        assert "valid" in assertion.message.lower()

    def test_error_assertion_details(self):
        """Test assertion details on error."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        result = evaluator(EvalInput(
            user_input="Get users",
            model_output="INVALID SQL QUERY",
        ))
        
        assert len(result.assertions) == 1
        assertion = result.assertions[0]
        assert assertion.name == "sql_parse"
        assert assertion.passed is False
        assert assertion.message  # Error message present

    def test_is_available_method(self):
        """Test is_available class method."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        # Should return True when sqlglot is installed
        assert SQLQueryValidator.is_available() is True

    def test_custom_eval_name(self):
        """Test custom eval name via config."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator, SQLQueryValidatorConfig
        
        config = SQLQueryValidatorConfig(name="my_sql_validator")
        evaluator = SQLQueryValidator(config)
        result = evaluator(EvalInput(
            user_input="Get users",
            model_output="SELECT * FROM users",
        ))
        
        assert result.eval_name == "my_sql_validator"

    def test_long_output_truncation(self):
        """Test that long outputs are truncated in error messages."""
        pytest.importorskip("sqlglot")
        from aiobs.evals import SQLQueryValidator
        
        evaluator = SQLQueryValidator()
        long_invalid_sql = "INVALID " * 50  # Create a long invalid string
        
        result = evaluator(EvalInput(
            user_input="Get users",
            model_output=long_invalid_sql,
        ))
        
        assert result.failed
        # Check that the actual output in assertion is truncated
        assertion = result.assertions[0]
        assert len(assertion.actual) <= 103  # 100 chars + "..."

