"""Minimal example: SQLQueryValidator evaluator.

Validates that model output is a valid SQL query using sqlglot.

Requirements:
    pip install 'aiobs[sql]'
"""

from aiobs.evals import EvalInput, SQLQueryValidator, SQLQueryValidatorConfig

# --- Example 1: Basic SQL validation ---
print("--- Example 1: Basic SQL Validation ---")

evaluator = SQLQueryValidator()

# Test with valid SQL
valid_input = EvalInput(
    user_input="Get all users from the database",
    model_output="SELECT * FROM users",
)

result = evaluator(valid_input)
print(f"Valid SQL - Status: {result.status.value}, Score: {result.score}")
print(f"Message: {result.message}\n")

# Test with invalid SQL
invalid_input = EvalInput(
    user_input="Get all users",
    model_output="SELECT FROM WHERE",  # Invalid SQL structure
)

result = evaluator(invalid_input)
print(f"Invalid SQL - Status: {result.status.value}, Score: {result.score}")
print(f"Message: {result.message}\n")


# --- Example 2: Dialect-specific validation ---
print("--- Example 2: PostgreSQL Dialect ---")

postgres_config = SQLQueryValidatorConfig(dialect="postgres")
pg_evaluator = SQLQueryValidator(postgres_config)

# PostgreSQL-specific query
pg_input = EvalInput(
    user_input="Get first 10 users",
    model_output="SELECT * FROM users LIMIT 10",
)

result = pg_evaluator(pg_input)
print(f"PostgreSQL Query - Status: {result.status.value}")
print(f"Message: {result.message}\n")


# --- Example 3: Complex query validation ---
print("--- Example 3: Complex Query Validation ---")

complex_input = EvalInput(
    user_input="Get users who registered in the last 30 days with their order counts",
    model_output="""
        SELECT 
            u.id,
            u.name,
            u.email,
            COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at >= NOW() - INTERVAL '30 days'
        GROUP BY u.id, u.name, u.email
        ORDER BY order_count DESC
    """,
)

result = evaluator(complex_input)
print(f"Complex Query - Status: {result.status.value}")
print(f"Message: {result.message}\n")


# --- Example 4: Dialect-Specific Query ---
print("--- Example 4: PostgreSQL-Specific Query ---")

# PostgreSQL-specific syntax with INTERVAL
postgres_config = SQLQueryValidatorConfig(dialect="postgres")
postgres_evaluator = SQLQueryValidator(postgres_config)

postgres_specific_input = EvalInput(
    user_input="Get users who registered in the last week",
    model_output="SELECT * FROM users WHERE created_at > NOW() - INTERVAL '7 days'",
)

result = postgres_evaluator(postgres_specific_input)
print(f"PostgreSQL-specific syntax: {result.status.value}")
print(f"Message: {result.message}\n")

# Try the same query without dialect
generic_evaluator = SQLQueryValidator()
result_generic = generic_evaluator(postgres_specific_input)
print(f"Same query without dialect: {result_generic.status.value}")


# --- Example 5: Validating LLM-Generated SQL ---
print("\n--- Example 5: Validating LLM-Generated SQL ---")

# Simulate an LLM that might generate invalid SQL
test_cases = [
    ("SELECT * FROM products WHERE", "Incomplete WHERE clause"),
    ("UPDATE users SET status = 'active' WHERE id = 1", "Valid UPDATE statement"),
    ("DELETE FROM orders WHERE order_date < '2023-01-01'", "Valid DELETE statement"),
    ("INSERT INTO users (name, email) VALUES ('John', 'john@example.com')", "Valid INSERT statement"),
]

for sql_query, description in test_cases:
    result = evaluator(EvalInput(
        user_input="Execute query",
        model_output=sql_query,
    ))
    status_symbol = "✓" if result.passed else "✗"
    print(f"{status_symbol} {description}: {result.status.value}")
    if result.failed and result.assertions:
        print(f"   Error: {result.assertions[0].message[:80]}...")
