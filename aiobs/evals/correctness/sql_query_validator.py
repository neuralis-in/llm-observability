"""SQL query validation evaluator."""

from __future__ import annotations

from typing import Any, List, Optional, Type

from ..base import BaseEval
from ..models import (
    EvalInput,
    EvalResult,
    EvalStatus,
    SQLQueryValidatorConfig,
    AssertionDetail,
)


class SQLQueryValidator(BaseEval):
    """Evaluator that validates SQL query syntax.
    
    This evaluator checks if the model output is a valid SQL query
    using sqlglot for parsing and validation. Supports multiple SQL
    dialects including PostgreSQL, MySQL, SQLite, and more.
    
    Example:
        evaluator = SQLQueryValidator()
        
        result = evaluator.evaluate(
            EvalInput(
                user_input="Get all users",
                model_output="SELECT * FROM users"
            )
        )
        assert result.passed
        
        # With specific dialect
        config = SQLQueryValidatorConfig(dialect="postgres")
        evaluator = SQLQueryValidator(config)
        
        result = evaluator.evaluate(
            EvalInput(
                user_input="Get users with limit",
                model_output="SELECT * FROM users LIMIT 10"
            )
        )
        assert result.passed
    """
    
    name: str = "sql_query_validator"
    description: str = "Validates that output is a valid SQL query"
    config_class: Type[SQLQueryValidatorConfig] = SQLQueryValidatorConfig
    
    _sqlglot_available: Optional[bool] = None
    
    def __init__(self, config: Optional[SQLQueryValidatorConfig] = None) -> None:
        """Initialize the SQL query validator.
        
        Args:
            config: Configuration for SQL validation.
        """
        super().__init__(config)
        self.config: SQLQueryValidatorConfig = self.config
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if sqlglot is installed.
        
        Returns:
            True if sqlglot is available, False otherwise.
        """
        if cls._sqlglot_available is None:
            try:
                import sqlglot  # noqa: F401
                cls._sqlglot_available = True
            except ImportError:
                cls._sqlglot_available = False
        return cls._sqlglot_available
    
    def evaluate(self, eval_input: EvalInput, **kwargs: Any) -> EvalResult:
        """Evaluate if model output is valid SQL.
        
        Args:
            eval_input: Input containing model_output to validate.
            **kwargs: Additional arguments (unused).
            
        Returns:
            EvalResult indicating pass/fail with details about SQL validity.
            
        Raises:
            ImportError: If sqlglot is not installed.
        """
        if not self.is_available():
            raise ImportError(
                "sqlglot is required for SQLQueryValidator. "
                "Install it with: pip install 'aiobs[sql]'"
            )
        
        import sqlglot
        from sqlglot.errors import ParseError
        
        output = eval_input.model_output.strip()
        assertions: List[AssertionDetail] = []
        
        # Attempt to parse the SQL query
        try:
            # Parse with specified dialect if provided
            parsed = sqlglot.parse_one(output, dialect=self.config.dialect)
            
            # Additional validation: ensure something was parsed
            if parsed is None:
                assertions.append(AssertionDetail(
                    name="sql_parse",
                    passed=False,
                    expected="Valid SQL query",
                    actual=output[:100] + "..." if len(output) > 100 else output,
                    message="Parser returned None - possibly empty or invalid SQL",
                ))
                return EvalResult(
                    status=EvalStatus.FAILED,
                    score=0.0,
                    eval_name=self.eval_name,
                    message="Invalid SQL: Parser returned None",
                    assertions=assertions,
                )
            
            # Successfully parsed
            dialect_info = f" (dialect: {self.config.dialect})" if self.config.dialect else ""
            assertions.append(AssertionDetail(
                name="sql_parse",
                passed=True,
                expected="Valid SQL query",
                actual="Parsed successfully",
                message=f"Output is valid SQL{dialect_info}",
            ))
            
            return EvalResult(
                status=EvalStatus.PASSED,
                score=1.0,
                eval_name=self.eval_name,
                message=f"Valid SQL query{dialect_info}",
                assertions=assertions,
            )
            
        except ParseError as e:
            # SQL parsing failed
            error_msg = str(e)
            assertions.append(AssertionDetail(
                name="sql_parse",
                passed=False,
                expected="Valid SQL query",
                actual=output[:100] + "..." if len(output) > 100 else output,
                message=error_msg,
            ))
            return EvalResult(
                status=EvalStatus.FAILED,
                score=0.0,
                eval_name=self.eval_name,
                message=f"Invalid SQL: {error_msg}",
                assertions=assertions,
            )
        except Exception as e:
            # Unexpected error during parsing
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            assertions.append(AssertionDetail(
                name="sql_parse",
                passed=False,
                expected="Valid SQL query",
                actual=output[:100] + "..." if len(output) > 100 else output,
                message=error_msg,
            ))
            return EvalResult(
                status=EvalStatus.FAILED,
                score=0.0,
                eval_name=self.eval_name,
                message=error_msg,
                assertions=assertions,
            )
