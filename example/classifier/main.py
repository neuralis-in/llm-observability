"""Minimal example demonstrating the OpenAI classifier.

This example shows how to use the classifier to evaluate whether
a model's response is good, bad, or uncertain.
"""

import os
import sys
from dotenv import load_dotenv, find_dotenv

# Ensure the package in repo root is importable when running this example directly
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from aiobs.classifier import OpenAIClassifier, ClassificationVerdict  # noqa: E402


def main() -> None:
    # Load env from nearest .env (searching upward)
    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in environment (.env)")
        sys.exit(1)

    # Create classifier
    classifier = OpenAIClassifier(api_key=api_key)

    # Example 1: Good response
    print("=" * 50)
    print("Example 1: Evaluating a GOOD response")
    print("=" * 50)
    
    result = classifier.classify(
        system_prompt="You are a helpful math tutor.",
        user_input="What is 2 + 2?",
        model_output="2 + 2 equals 4."
    )
    
    print(f"Verdict: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print()

    # Example 2: Bad response (incorrect answer)
    print("=" * 50)
    print("Example 2: Evaluating a BAD response")
    print("=" * 50)
    
    result = classifier.classify(
        system_prompt="You are a helpful geography assistant.",
        user_input="What is the capital of France?",
        model_output="The capital of France is London."
    )
    
    print(f"Verdict: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    if result.categories:
        print(f"Categories: {', '.join(result.categories)}")
    print()

    # Example 3: Off-topic response
    print("=" * 50)
    print("Example 3: Evaluating an OFF-TOPIC response")
    print("=" * 50)
    
    result = classifier.classify(
        system_prompt="You are a cooking assistant.",
        user_input="How do I make pasta?",
        model_output="The weather today is sunny with a high of 75°F."
    )
    
    print(f"Verdict: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    if result.categories:
        print(f"Categories: {', '.join(result.categories)}")


if __name__ == "__main__":
    main()

