"""Compute correctness evaluator for math and code outputs."""

from __future__ import annotations

import ast
import math
import re
from typing import Any, List, Optional, Type, Union

from ..base import BaseEval
from ..models import (
    EvalInput,
    EvalResult,
    EvalStatus,
    ComputeCorrectnessConfig,
    ComputeMode,
    AssertionDetail,
)


class ComputeCorrectnessEval(BaseEval):
    """Evaluator that checks correctness of mathematical and computational outputs.

    Supports multiple evaluation modes:
    - numeric: Compares numeric values with tolerance
    - exact: Exact string/value comparison
    - expression: Evaluates mathematical expressions
    - code_output: Safely executes code and compares outputs

    Example:
        # Numeric comparison with tolerance
        config = ComputeCorrectnessConfig(
            mode=ComputeMode.NUMERIC,
            tolerance=0.01
        )
        evaluator = ComputeCorrectnessEval(config)

        result = evaluator.evaluate(
            EvalInput(
                user_input="What is the square root of 2?",
                model_output="1.414",
                expected_output="1.41421356"
            )
        )

        # Expression evaluation
        evaluator = ComputeCorrectnessEval.expression()
        result = evaluator.evaluate(
            EvalInput(
                user_input="Solve 2 + 2 * 3",
                model_output="8",
                expected_output="8"
            )
        )
    """

    name: str = "compute_correctness"
    description: str = "Evaluates correctness of mathematical and computational outputs"
    config_class: Type[ComputeCorrectnessConfig] = ComputeCorrectnessConfig

    def __init__(self, config: Optional[ComputeCorrectnessConfig] = None) -> None:
        """Initialize with configuration.

        Args:
            config: Configuration for computation checking.
        """
        super().__init__(config)
        self.config: ComputeCorrectnessConfig = self.config

    @classmethod
    def numeric(
        cls,
        tolerance: float = 1e-6,
        relative_tolerance: bool = False,
    ) -> "ComputeCorrectnessEval":
        """Create evaluator for numeric comparison.

        Args:
            tolerance: Absolute or relative tolerance for comparison.
            relative_tolerance: If True, tolerance is relative to expected value.

        Returns:
            Configured ComputeCorrectnessEval instance.
        """
        return cls(
            ComputeCorrectnessConfig(
                mode=ComputeMode.NUMERIC,
                tolerance=tolerance,
                relative_tolerance=relative_tolerance,
            )
        )

    @classmethod
    def exact(cls) -> "ComputeCorrectnessEval":
        """Create evaluator for exact comparison.

        Returns:
            Configured ComputeCorrectnessEval instance.
        """
        return cls(ComputeCorrectnessConfig(mode=ComputeMode.EXACT))

    @classmethod
    def expression(cls, tolerance: float = 1e-6) -> "ComputeCorrectnessEval":
        """Create evaluator for mathematical expression evaluation.

        Args:
            tolerance: Tolerance for numeric comparison after evaluation.

        Returns:
            Configured ComputeCorrectnessEval instance.
        """
        return cls(
            ComputeCorrectnessConfig(
                mode=ComputeMode.EXPRESSION,
                tolerance=tolerance,
            )
        )

    @classmethod
    def code_output(
        cls,
        timeout_seconds: float = 5.0,
        allowed_modules: Optional[List[str]] = None,
    ) -> "ComputeCorrectnessEval":
        """Create evaluator for code output verification.

        Args:
            timeout_seconds: Maximum execution time for code.
            allowed_modules: List of allowed module names for imports.

        Returns:
            Configured ComputeCorrectnessEval instance.
        """
        return cls(
            ComputeCorrectnessConfig(
                mode=ComputeMode.CODE_OUTPUT,
                timeout_seconds=timeout_seconds,
                allowed_modules=allowed_modules or [],
            )
        )

    def evaluate(self, eval_input: EvalInput, **kwargs: Any) -> EvalResult:
        """Evaluate computational correctness.

        Args:
            eval_input: Input containing model_output and expected_output.
            **kwargs: Can contain 'expected' to override eval_input.expected_output.

        Returns:
            EvalResult indicating correctness.
        """
        import time

        start_time = time.time()

        # Get expected output
        expected = kwargs.get("expected", eval_input.expected_output)

        if expected is None:
            return EvalResult.error_result(
                eval_name=self.eval_name,
                error=ValueError("No expected_output provided in eval_input or kwargs"),
            )

        output = eval_input.model_output

        # Extract numeric/code values if configured (but not for expression mode on expected)
        if self.config.extract_from_text:
            output = self._extract_value(output)
            # For expression mode, don't extract from expected - we want to evaluate it
            if self.config.mode != ComputeMode.EXPRESSION:
                expected = self._extract_value(expected)

        # Perform comparison based on mode
        try:
            if self.config.mode == ComputeMode.NUMERIC:
                passed, score, details = self._numeric_comparison(output, expected)
            elif self.config.mode == ComputeMode.EXACT:
                passed, score, details = self._exact_comparison(output, expected)
            elif self.config.mode == ComputeMode.EXPRESSION:
                passed, score, details = self._expression_comparison(output, expected)
            elif self.config.mode == ComputeMode.CODE_OUTPUT:
                passed, score, details = self._code_output_comparison(output, expected)
            else:
                return EvalResult.error_result(
                    eval_name=self.eval_name,
                    error=ValueError(f"Unknown compute mode: {self.config.mode}"),
                )
        except Exception as e:
            return EvalResult.error_result(
                eval_name=self.eval_name,
                error=e,
            )

        duration_ms = (time.time() - start_time) * 1000

        # Build result
        status = EvalStatus.PASSED if passed else EvalStatus.FAILED
        message = details.get("message", "")

        return EvalResult(
            status=status,
            score=score,
            eval_name=self.eval_name,
            message=message,
            details=details,
            duration_ms=duration_ms,
        )

    def _extract_value(self, text: str) -> str:
        """Extract numeric or code value from text.

        Args:
            text: Text potentially containing numeric/code values.

        Returns:
            Extracted value or original text.
        """
        # Try to extract number (with optional decimal, scientific notation)
        number_pattern = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
        matches = re.findall(number_pattern, text)
        if matches:
            # Return the last number found (often the answer)
            return matches[-1]

        # Try to extract code block
        code_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        code_matches = re.findall(code_pattern, text, re.DOTALL)
        if code_matches:
            return code_matches[0].strip()

        return text.strip()

    def _numeric_comparison(
        self,
        output: Union[str, float, int],
        expected: Union[str, float, int],
    ) -> tuple[bool, float, dict]:
        """Compare numeric values with tolerance.

        Args:
            output: Model output value.
            expected: Expected value.

        Returns:
            Tuple of (passed, score, details).
        """
        try:
            # Convert to float
            output_num = float(output)
            expected_num = float(expected)

            # Handle special values
            if math.isnan(output_num) and math.isnan(expected_num):
                return (
                    True,
                    1.0,
                    {
                        "message": "Both values are NaN",
                        "output": output_num,
                        "expected": expected_num,
                    },
                )

            if math.isinf(output_num) and math.isinf(expected_num):
                if output_num == expected_num:  # Same sign infinity
                    return (
                        True,
                        1.0,
                        {
                            "message": "Both values are infinity with same sign",
                            "output": output_num,
                            "expected": expected_num,
                        },
                    )

            # Calculate difference
            diff = abs(output_num - expected_num)

            # Apply tolerance
            if self.config.relative_tolerance:
                tolerance = abs(expected_num) * self.config.tolerance
            else:
                tolerance = self.config.tolerance

            passed = diff <= tolerance

            # Calculate score (inverse of normalized difference)
            if expected_num != 0:
                normalized_diff = diff / abs(expected_num)
            else:
                normalized_diff = diff

            score = max(0.0, 1.0 - normalized_diff)

            return (
                passed,
                score,
                {
                    "message": f"Difference: {diff:.6e} (tolerance: {tolerance:.6e})",
                    "output": output_num,
                    "expected": expected_num,
                    "difference": diff,
                    "tolerance": tolerance,
                    "relative_tolerance": self.config.relative_tolerance,
                },
            )

        except (ValueError, TypeError) as e:
            return (
                False,
                0.0,
                {
                    "message": f"Failed to convert to numeric: {e}",
                    "output": output,
                    "expected": expected,
                    "error": str(e),
                },
            )

    def _exact_comparison(self, output: Any, expected: Any) -> tuple[bool, float, dict]:
        """Exact comparison of values.

        Args:
            output: Model output.
            expected: Expected value.

        Returns:
            Tuple of (passed, score, details).
        """
        # Convert to string and normalize whitespace if strings
        if isinstance(output, str) and isinstance(expected, str):
            output = output.strip()
            expected = expected.strip()

        passed = output == expected
        score = 1.0 if passed else 0.0

        return (
            passed,
            score,
            {
                "message": "Exact match" if passed else "Values do not match exactly",
                "output": output,
                "expected": expected,
            },
        )

    def _expression_comparison(
        self,
        output: Union[str, float, int],
        expected: Union[str, float, int],
    ) -> tuple[bool, float, dict]:
        """Evaluate mathematical expressions and compare.

        Args:
            output: Mathematical expression or value.
            expected: Expected expression or value.

        Returns:
            Tuple of (passed, score, details).
        """
        try:
            # Evaluate both expressions
            output_value = self._safe_eval_expression(str(output))
            expected_value = self._safe_eval_expression(str(expected))

            # Use numeric comparison
            return self._numeric_comparison(output_value, expected_value)

        except Exception as e:
            return (
                False,
                0.0,
                {
                    "message": f"Failed to evaluate expression: {e}",
                    "output": output,
                    "expected": expected,
                    "error": str(e),
                },
            )

    def _safe_eval_expression(self, expr: str) -> float:
        """Safely evaluate a mathematical expression.

        Args:
            expr: Mathematical expression string.

        Returns:
            Evaluated numeric result.

        Raises:
            ValueError: If expression is invalid or unsafe.
        """
        # Try direct conversion first
        try:
            return float(expr)
        except ValueError:
            pass

        # Parse and validate AST
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")

        # Only allow safe operations
        allowed_nodes = (
            ast.Expression,
            ast.Constant,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.FloorDiv,
            ast.Call,
            ast.Name,
            ast.Load,  # Allow function calls
        )

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(
                    f"Unsafe operation in expression: {node.__class__.__name__}"
                )

        # Safe evaluation with limited namespace
        safe_namespace = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "sum": sum,
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
        }

        try:
            result = eval(
                compile(tree, "<string>", "eval"), {"__builtins__": {}}, safe_namespace
            )
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")

    def _code_output_comparison(
        self,
        output: str,
        expected: str,
    ) -> tuple[bool, float, dict]:
        """Execute code and compare outputs.

        Note: This is a simplified implementation. Full sandboxing
        would require containerization or external services.

        Args:
            output: Code to execute.
            expected: Expected output or code.

        Returns:
            Tuple of (passed, score, details).
        """
        # For safety, this is a placeholder implementation
        # Production use should involve proper sandboxing
        return (
            False,
            0.0,
            {
                "message": "Code execution not implemented for safety reasons",
                "note": "Use external sandbox services for code evaluation",
                "output": output[:100],
                "expected": expected[:100],
            },
        )
