Evals
=====

The evals module provides a comprehensive evaluation framework for assessing LLM outputs
across multiple dimensions: correctness, safety, reliability, and performance.

Overview
--------

Evaluators are useful for:

- **Correctness**: Verify outputs match expected patterns, schemas, or ground truth
- **Safety**: Detect PII leakage and sensitive information exposure
- **Reliability**: Check latency consistency and performance stability
- **Quality Assurance**: Automated testing of LLM responses

Status Overview
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 15

   * - Category
     - Eval Type
     - Status
   * - Correctness
     - Rule-based assertions (regex/schema)
     - ✅
   * - Correctness
     - Ground-truth comparison
     - ✅
   * - Correctness
     - Hallucination detection (LLM-as-judge)
     - ✅
   * - Safety
     - PII leakage detection
     - ✅
   * - Reliability
     - Latency consistency checks
     - ✅
   * - Domain-Specific
     - Extraction: schema accuracy
     - ✅

Quick Start
-----------

.. code-block:: python

   from aiobs.evals import EvalInput, RegexAssertion, PIIDetectionEval

   # Create input
   eval_input = EvalInput(
       user_input="What is the capital of France?",
       model_output="The capital of France is Paris.",
       system_prompt="You are a geography expert."
   )

   # Run regex evaluation
   regex_eval = RegexAssertion.from_patterns(patterns=[r"Paris"])
   result = regex_eval(eval_input)
   print(f"Status: {result.status.value}")  # "passed"

   # Check for PII
   pii_eval = PIIDetectionEval.default()
   result = pii_eval(eval_input)
   print(f"PII found: {result.details['pii_count']}")  # 0

Core Models
-----------

EvalInput
^^^^^^^^^

The standard input model for all evaluators:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``user_input``
     - ``str``
     - The user's input/query to the model (required)
   * - ``model_output``
     - ``str``
     - The model's generated response (required)
   * - ``system_prompt``
     - ``Optional[str]``
     - The system prompt provided to the model
   * - ``expected_output``
     - ``Optional[str]``
     - Expected/ground-truth output for comparison evals
   * - ``context``
     - ``Optional[Dict]``
     - Additional context (e.g., retrieved docs)
   * - ``metadata``
     - ``Optional[Dict]``
     - Additional metadata (e.g., latency, token counts)
   * - ``tags``
     - ``Optional[List[str]]``
     - Tags for categorizing eval inputs

EvalResult
^^^^^^^^^^

Result model returned by all evaluators:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``status``
     - ``EvalStatus``
     - The evaluation status: ``PASSED``, ``FAILED``, ``ERROR``, or ``SKIPPED``
   * - ``score``
     - ``float``
     - Numeric score between 0 (worst) and 1 (best)
   * - ``eval_name``
     - ``str``
     - Name of the evaluator that produced this result
   * - ``message``
     - ``Optional[str]``
     - Human-readable message explaining the result
   * - ``details``
     - ``Optional[Dict]``
     - Detailed information about the evaluation
   * - ``assertions``
     - ``Optional[List[AssertionDetail]]``
     - Individual assertion results (for multi-assertion evals)
   * - ``duration_ms``
     - ``Optional[float]``
     - Time taken to run the evaluation in milliseconds
   * - ``evaluated_at``
     - ``datetime``
     - Timestamp when evaluation was performed

EvalStatus
^^^^^^^^^^

Enum representing evaluation outcomes:

- ``EvalStatus.PASSED`` - Evaluation passed all checks
- ``EvalStatus.FAILED`` - Evaluation failed one or more checks
- ``EvalStatus.ERROR`` - An error occurred during evaluation
- ``EvalStatus.SKIPPED`` - Evaluation was skipped

Correctness Evaluators
----------------------

RegexAssertion
^^^^^^^^^^^^^^

Asserts that model output matches (or doesn't match) regex patterns.

.. code-block:: python

   from aiobs.evals import RegexAssertion, EvalInput

   # Patterns that MUST match
   evaluator = RegexAssertion.from_patterns(
       patterns=[r"Paris", r"\d+"],
       match_mode="all",  # All patterns must match (or "any")
       case_sensitive=False,
   )

   result = evaluator(EvalInput(
       user_input="Population of Paris?",
       model_output="Paris has about 2.1 million people."
   ))
   print(result.status.value)  # "passed"

   # Patterns that must NOT match (negative patterns)
   no_apology = RegexAssertion.from_patterns(
       negative_patterns=[r"\b(sorry|cannot|unable)\b"],
       case_sensitive=False,
   )

Configuration options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``patterns``
     - ``[]``
     - Patterns that output must match
   * - ``negative_patterns``
     - ``[]``
     - Patterns that output must NOT match
   * - ``case_sensitive``
     - ``True``
     - Whether matching is case-sensitive
   * - ``match_mode``
     - ``"any"``
     - ``"any"`` (at least one) or ``"all"`` (all must match)

SchemaAssertion
^^^^^^^^^^^^^^^

Validates that model output is valid JSON matching a JSON Schema.

.. code-block:: python

   from aiobs.evals import SchemaAssertion, EvalInput

   schema = {
       "type": "object",
       "properties": {
           "name": {"type": "string"},
           "age": {"type": "integer", "minimum": 0}
       },
       "required": ["name", "age"]
   }

   evaluator = SchemaAssertion.from_schema(schema)

   result = evaluator(EvalInput(
       user_input="Extract person info",
       model_output='{"name": "John", "age": 30}'
   ))
   print(result.status.value)  # "passed"

   # Also extracts JSON from markdown code blocks
   result = evaluator(EvalInput(
       user_input="Give me JSON",
       model_output='Here is the data:\n```json\n{"name": "Alice", "age": 25}\n```'
   ))
   print(result.status.value)  # "passed"

.. note::
   Full JSON Schema validation requires the ``jsonschema`` package.
   Install with: ``pip install jsonschema``

GroundTruthEval
^^^^^^^^^^^^^^^

Compares model output against expected ground truth.

.. code-block:: python

   from aiobs.evals import GroundTruthEval, EvalInput

   # Exact match
   exact_eval = GroundTruthEval.exact(case_sensitive=False)
   result = exact_eval(EvalInput(
       user_input="What is 2+2?",
       model_output="4",
       expected_output="4"
   ))

   # Contains match
   contains_eval = GroundTruthEval.contains()
   result = contains_eval(EvalInput(
       user_input="Capital of France?",
       model_output="The capital is Paris.",
       expected_output="Paris"
   ))

   # Normalized match (whitespace/case normalized)
   normalized_eval = GroundTruthEval.normalized(
       case_sensitive=False,
       strip_punctuation=True
   )

Match modes:

- ``exact`` - Exact string match
- ``contains`` - Output contains expected string
- ``normalized`` - Whitespace/case normalized comparison

HallucinationDetectionEval
^^^^^^^^^^^^^^^^^^^^^^^^^^

Detects hallucinations in model outputs using an LLM-as-judge approach.
This evaluator uses another LLM to analyze if the model output contains
fabricated, false, or unsupported information.

.. code-block:: python

   from openai import OpenAI
   from aiobs.evals import HallucinationDetectionEval, EvalInput

   # Initialize with your LLM client
   client = OpenAI()
   evaluator = HallucinationDetectionEval(
       client=client,
       model="gpt-4o-mini",  # Judge model
   )

   # Evaluate with context (for RAG use cases)
   result = evaluator(EvalInput(
       user_input="What is the capital of France?",
       model_output="Paris is the capital of France. It was founded by Julius Caesar in 250 BC.",
       context={
           "documents": ["Paris is the capital and largest city of France."]
       }
   ))

   print(result.status.value)  # "failed" (hallucination detected)
   print(result.score)  # 0.3
   print(result.details["hallucinations"])  # List of detected hallucinations

   # Evaluate without context (general factuality check)
   result = evaluator(EvalInput(
       user_input="Who is the CEO of Apple?",
       model_output="Tim Cook is the CEO of Apple.",
   ))
   print(result.status.value)  # "passed"

Factory methods for different providers:

.. code-block:: python

   # OpenAI
   evaluator = HallucinationDetectionEval.with_openai(
       client=OpenAI(),
       model="gpt-4o",
   )

   # Gemini
   from google import genai
   evaluator = HallucinationDetectionEval.with_gemini(
       client=genai.Client(),
       model="gemini-2.0-flash",
   )

   # Anthropic
   from anthropic import Anthropic
   evaluator = HallucinationDetectionEval.with_anthropic(
       client=Anthropic(),
       model="claude-3-sonnet-20240229",
   )

Configuration options:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``model``
     - ``None``
     - Model name for logging purposes
   * - ``temperature``
     - ``0.0``
     - Temperature for the judge LLM
   * - ``strict``
     - ``False``
     - Fail on any hallucination (regardless of score)
   * - ``hallucination_threshold``
     - ``0.5``
     - Score below which output is considered hallucinated
   * - ``check_against_context``
     - ``True``
     - Check if output is grounded in provided context
   * - ``extract_claims``
     - ``True``
     - Extract and evaluate individual claims
   * - ``max_claims``
     - ``10``
     - Maximum claims to extract and evaluate

Result details:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``score``
     - Hallucination score (1.0 = no hallucinations, 0.0 = completely hallucinated)
   * - ``has_hallucinations``
     - Boolean indicating if hallucinations were detected
   * - ``hallucination_count``
     - Number of hallucinations found
   * - ``hallucinations``
     - List of detected hallucinations with claim, reason, and severity
   * - ``analysis``
     - Overall analysis from the judge LLM
   * - ``judge_model``
     - Model used for evaluation

Safety Evaluators
-----------------

ToxicityDetectionEval
^^^^^^^^^^^^^^^^^^^^^

Detects toxic content in model outputs using an LLM-as-judge approach. This includes:

- Hateful statements
- Demeaning or disparaging content
- Inappropriate language
- Threats of physical or emotional violence
- Racist, biased, or discriminatory content

.. code-block:: python

   from openai import OpenAI
   from aiobs.evals import ToxicityDetectionEval, EvalInput

   # Initialize with your LLM client
   client = OpenAI()
   evaluator = ToxicityDetectionEval(
       client=client,
       model="gpt-4o",  # Use a powerful model for accurate detection
   )

   # Evaluate a non-toxic example
   non_toxic_result = evaluator.evaluate(
       EvalInput(
           user_input="Tell me something nice",
           model_output="You are a wonderful person with great potential!"
       )
   )
   print(non_toxic_result.status.value)  # "passed"

   # Evaluate a toxic example
   toxic_result = evaluator.evaluate(
       EvalInput(
           user_input="What do you think about people from other countries?",
           model_output="I hate immigrants, they should all go back to where they came from!"
       )
   )
   print(toxic_result.status.value)  # "failed"

   # With custom configuration
   from aiobs.evals.models.configs import ToxicityDetectionConfig
   custom_evaluator = ToxicityDetectionEval(
       client=client,
       model="gpt-4o",
       config=ToxicityDetectionConfig(
           toxicity_threshold=0.3,  # More sensitive detection
           categories=["hate_speech", "harassment"],  # Focus on specific categories
           fail_on_detection=True
       )
   )

Configuration options:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``model``
     - ``None``
     - Model name for the judge LLM (logging)
   * - ``temperature``
     - ``0.0``
     - Temperature for the judge LLM
   * - ``toxicity_threshold``
     - ``0.5``
     - Score threshold above which output is considered toxic (0-1)
   * - ``check_input``
     - ``False``
     - Also check user input for toxicity
   * - ``categories``
     - ``["hate_speech", "harassment", "violence", "profanity", "discrimination"]``
     - Categories of toxicity to detect
   * - ``fail_on_detection``
     - ``True``
     - Whether to fail the eval if toxicity is detected

Result details:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``toxicity_score``
     - Maximum toxicity score detected (0.0-1.0)
   * - ``detected_categories``
     - List of toxicity categories detected
   * - ``explanation``
     - Explanation from the judge LLM

PIIDetectionEval
^^^^^^^^^^^^^^^^

Detects personally identifiable information (PII) in model outputs.

.. code-block:: python

   from aiobs.evals import PIIDetectionEval, PIIType, EvalInput

   # Default detector (email, phone, SSN, credit card)
   evaluator = PIIDetectionEval.default()

   result = evaluator(EvalInput(
       user_input="Contact info?",
       model_output="Email me at john@example.com"
   ))
   print(result.status.value)  # "failed" (PII detected)
   print(result.details["pii_types_found"])  # ["email"]

   # Scan and redact PII
   matches = evaluator.scan("Call 555-123-4567")
   redacted = evaluator.redact("Call 555-123-4567")
   print(redacted)  # "Call [PHONE REDACTED]"

   # Custom patterns
   custom_eval = PIIDetectionEval.with_custom_patterns({
       "employee_id": r"EMP-\d{6}",
   })

   # Strict mode (checks input and system prompt too)
   strict_eval = PIIDetectionEval.strict()

Supported PII types:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Example Pattern
   * - ``EMAIL``
     - ``user@example.com``
   * - ``PHONE``
     - ``555-123-4567``, ``(555) 123-4567``
   * - ``SSN``
     - ``123-45-6789``
   * - ``CREDIT_CARD``
     - ``4111111111111111``
   * - ``IP_ADDRESS``
     - ``192.168.1.100``
   * - ``DATE_OF_BIRTH``
     - ``01/15/1990``

Reliability Evaluators
----------------------

LatencyConsistencyEval
^^^^^^^^^^^^^^^^^^^^^^

Checks latency statistics across multiple runs.

.. code-block:: python

   from aiobs.evals import LatencyConsistencyEval, EvalInput

   evaluator = LatencyConsistencyEval.with_thresholds(
       max_latency_ms=1000,        # Max single latency
       max_p95_ms=800,             # 95th percentile threshold
       cv_threshold=0.3,           # Coefficient of variation
   )

   result = evaluator(EvalInput(
       user_input="test",
       model_output="response",
       metadata={"latencies": [100, 120, 95, 110, 105]}
   ))

   print(result.status.value)  # "passed"
   print(result.details["mean"])  # 106.0
   print(result.details["p95"])   # 118.0
   print(result.details["cv"])    # 0.09

Configuration options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``max_latency_ms``
     - Maximum acceptable latency in ms
   * - ``max_std_dev_ms``
     - Maximum acceptable standard deviation
   * - ``max_p95_ms``
     - Maximum acceptable 95th percentile
   * - ``max_p99_ms``
     - Maximum acceptable 99th percentile
   * - ``coefficient_of_variation_threshold``
     - Maximum CV (std_dev / mean)

Statistics returned:

- ``count``, ``mean``, ``min``, ``max``, ``median``
- ``std_dev``, ``variance``, ``cv``
- ``p50``, ``p90``, ``p95``, ``p99``

Batch Evaluation
----------------

All evaluators support batch evaluation:

.. code-block:: python

   from aiobs.evals import RegexAssertion, EvalInput

   evaluator = RegexAssertion.from_patterns(patterns=[r"\d+"])

   inputs = [
       EvalInput(user_input="q1", model_output="Answer: 123"),
       EvalInput(user_input="q2", model_output="No numbers"),
       EvalInput(user_input="q3", model_output="Result: 456"),
   ]

   results = evaluator.evaluate_batch(inputs)

   for inp, result in zip(inputs, results):
       print(f"{inp.model_output[:20]}... → {result.status.value}")

Custom Evaluators
-----------------

Create custom evaluators by extending ``BaseEval``:

.. code-block:: python

   from aiobs.evals import BaseEval, EvalInput, EvalResult, EvalStatus
   from typing import Any

   class LengthEval(BaseEval):
       """Evaluates if output length is within bounds."""

       name = "length_eval"
       description = "Checks output length"

       def __init__(self, min_length: int = 0, max_length: int = 1000):
           super().__init__()
           self.min_length = min_length
           self.max_length = max_length

       def evaluate(self, eval_input: EvalInput, **kwargs: Any) -> EvalResult:
           length = len(eval_input.model_output)
           passed = self.min_length <= length <= self.max_length

           return EvalResult(
               status=EvalStatus.PASSED if passed else EvalStatus.FAILED,
               score=1.0 if passed else 0.0,
               eval_name=self.eval_name,
               message=f"Length {length} {'within' if passed else 'outside'} [{self.min_length}, {self.max_length}]",
               details={"length": length},
           )

   # Usage
   eval = LengthEval(min_length=10, max_length=500)
   result = eval(EvalInput(user_input="q", model_output="Short"))

Examples
--------

The repository includes eval examples at ``example/evals/``:

- ``regex_assertion_example.py`` - Pattern matching examples
- ``schema_assertion_example.py`` - JSON schema validation
- ``ground_truth_example.py`` - Ground truth comparison
- ``hallucination_detection_example.py`` - Hallucination detection with LLM-as-judge
- ``latency_consistency_example.py`` - Latency statistics
- ``pii_detection_example.py`` - PII detection and redaction

Run the examples::

   cd example/evals
   PYTHONPATH=../.. python regex_assertion_example.py

API Reference
-------------

.. automodule:: aiobs.evals
   :members:
   :undoc-members:
   :show-inheritance:

