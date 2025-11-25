"""
Standalone example demonstrating the @observe decorator for function tracing.

This example shows how to trace regular Python functions (sync and async)
without requiring any external API keys. Run with:

    python example/observe-decorator/main.py

Or from repo root:

    python -m example.observe-decorator.main
"""

import asyncio
import os
import sys
import time

# Ensure the package in repo root is importable when running this example directly
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from aiobs import observer, observe  # noqa: E402


# =============================================================================
# Example 1: Basic function tracing
# =============================================================================

@observe
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number (iterative)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# =============================================================================
# Example 2: Custom name for the trace
# =============================================================================

@observe(name="data_transform")
def transform_data(data: list) -> list:
    """Transform a list by doubling each element."""
    return [x * 2 for x in data]


# =============================================================================
# Example 3: Async function tracing
# =============================================================================

@observe
async def fetch_simulated(url: str, delay: float = 0.1) -> dict:
    """Simulate an async HTTP fetch with artificial delay."""
    await asyncio.sleep(delay)
    return {"url": url, "status": 200, "data": f"Response from {url}"}


# =============================================================================
# Example 4: Hiding sensitive arguments
# =============================================================================

@observe(capture_args=False)
def authenticate(username: str, password: str) -> bool:
    """Authenticate a user (args won't be captured for privacy)."""
    # Simulated authentication
    return username == "admin" and password == "secret"


# =============================================================================
# Example 5: Hiding large return values
# =============================================================================

@observe(capture_result=False)
def generate_large_data(size: int) -> list:
    """Generate a large list (result won't be captured to save space)."""
    return list(range(size))


# =============================================================================
# Example 6: Pipeline of traced functions
# =============================================================================

@observe(name="pipeline_fetch")
def fetch_items() -> list:
    """Step 1: Fetch raw items."""
    time.sleep(0.05)  # Simulate work
    return [1, 2, 3, 4, 5]


@observe(name="pipeline_process")
def process_items(items: list) -> list:
    """Step 2: Process items."""
    time.sleep(0.03)  # Simulate work
    return [x ** 2 for x in items]


@observe(name="pipeline_aggregate")
def aggregate_results(items: list) -> dict:
    """Step 3: Aggregate results."""
    time.sleep(0.02)  # Simulate work
    return {"sum": sum(items), "count": len(items), "items": items}


@observe(name="full_pipeline")
def run_pipeline() -> dict:
    """Run the complete pipeline (traces nested calls)."""
    items = fetch_items()
    processed = process_items(items)
    return aggregate_results(processed)


# =============================================================================
# Example 7: Error handling
# =============================================================================

@observe
def divide(a: float, b: float) -> float:
    """Divide a by b (will capture error if b is zero)."""
    return a / b


# =============================================================================
# Main
# =============================================================================

@observe
def main() -> None:
    print("=" * 60)
    print("@observe Decorator Example")
    print("=" * 60)

    # Example 1: Basic tracing
    print("\n1. Basic function tracing (Fibonacci):")
    result = fibonacci(10)
    print(f"   fibonacci(10) = {result}")

    # Example 2: Custom name
    print("\n2. Custom trace name:")
    result = transform_data([1, 2, 3, 4, 5])
    print(f"   transform_data([1,2,3,4,5]) = {result}")

    # Example 3: Async function
    print("\n3. Async function tracing:")
    result = asyncio.run(fetch_simulated("https://api.example.com/data"))
    print(f"   fetch_simulated() = {result}")

    # Example 4: Hidden args (for sensitive data)
    print("\n4. Hidden arguments (password not captured):")
    result = authenticate("admin", "secret")
    print(f"   authenticate() = {result}")

    # Example 5: Hidden result (for large data)
    print("\n5. Hidden result (large list not captured):")
    result = generate_large_data(10000)
    print(f"   generate_large_data(10000) returned {len(result)} items")

    # Example 6: Pipeline with nested traces
    print("\n6. Pipeline (nested function traces):")
    result = run_pipeline()
    print(f"   run_pipeline() = {result}")

    # Example 7: Error handling
    print("\n7. Error handling (division by zero):")
    try:
        divide(10, 0)
    except ZeroDivisionError:
        print("   Caught ZeroDivisionError (error is captured in trace)")



if __name__ == "__main__":
    # Start observability session
    observer.observe(session_name="observe-decorator-demo")
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # End session and flush to JSON
        observer.end()
        out_path = observer.flush()

    print("\n" + "=" * 60)
    print(f"Observability data written to: {out_path}")
    print("=" * 60)

    # Show a preview of what was captured
    import json
    with open(out_path) as f:
        data = json.load(f)

    print(f"\nSession: {data['sessions'][0]['name']}")
    print(f"Total function events: {len(data['function_events'])}")
    print("\nCaptured traces:")
    for ev in data["function_events"]:
        error_marker = " [ERROR]" if ev.get("error") else ""
        args_preview = str(ev.get("args", []))[:40]
        print(f"  - {ev['name']}: {ev['duration_ms']:.2f}ms{error_marker}")
        if ev.get("args"):
            print(f"    args: {args_preview}...")


