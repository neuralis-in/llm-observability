"""Minimal example: ComputeCorrectnessEval evaluator.

Evaluates mathematical and computational correctness of outputs.
"""

from aiobs.evals import EvalInput, ComputeCorrectnessEval, ComputeMode

# --- Numeric Comparison ---
print("--- Numeric Comparison ---")

numeric_eval = ComputeCorrectnessEval.numeric(tolerance=0.01)

result = numeric_eval(EvalInput(
    user_input="What is the square root of 2?",
    model_output="1.414",
    expected_output="1.41421356",
))
print(f"Within tolerance: {result.status.value} (score: {result.score:.4f})")

# Extracts numbers from text automatically
result = numeric_eval(EvalInput(
    user_input="What is 2+2?",
    model_output="The answer is 4",
    expected_output="4",
))
print(f"Text extraction: {result.status.value}")

# --- Exact Comparison ---
print("\n--- Exact Comparison ---")

exact_eval = ComputeCorrectnessEval.exact()

result = exact_eval(EvalInput(
    user_input="Repeat 'hello'",
    model_output="hello",
    expected_output="hello",
))
print(f"Exact match: {result.status.value}")

# --- Expression Evaluation ---
print("\n--- Expression Evaluation ---")

expr_eval = ComputeCorrectnessEval.expression(tolerance=0.01)

result = expr_eval(EvalInput(
    user_input="What is 2 + 2 * 3?",
    model_output="8",
    expected_output="2 + 2 * 3",
))
print(f"Arithmetic expression: {result.status.value}")

result = expr_eval(EvalInput(
    user_input="Calculate sqrt(2)",
    model_output="1.414",
    expected_output="sqrt(2)",
))
print(f"Math function (sqrt): {result.status.value}")

# --- Relative Tolerance ---
print("\n--- Relative Tolerance ---")

relative_eval = ComputeCorrectnessEval.numeric(
    tolerance=0.01,  # 1% relative tolerance
    relative_tolerance=True,
)

result = relative_eval(EvalInput(
    user_input="Large number",
    model_output="100.5",
    expected_output="100",
))
print(f"1% relative tolerance: {result.status.value}")

# --- Custom Configuration ---
print("\n--- Custom Configuration ---")

from aiobs.evals import ComputeCorrectnessConfig

custom_eval = ComputeCorrectnessEval(ComputeCorrectnessConfig(
    name="my_math_checker",
    mode=ComputeMode.NUMERIC,
    tolerance=0.001,
))

result = custom_eval(EvalInput(
    user_input="Calculate 7*8",
    model_output="The result is 56",
    expected_output="56",
))
print(f"Custom config: {result.status.value}")

