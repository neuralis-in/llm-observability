"""Async example demonstrating batch classification.

This example shows how to use the classifier asynchronously
to evaluate multiple responses in parallel.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Ensure the package in repo root is importable when running this example directly
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from aiobs.classifier import (  # noqa: E402
    OpenAIClassifier,
    ClassificationInput,
    ClassificationVerdict,
)


async def main() -> None:
    # Load env from nearest .env (searching upward)
    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in environment (.env)")
        sys.exit(1)

    # Create classifier
    classifier = OpenAIClassifier(api_key=api_key)

    # Prepare batch of inputs to classify
    inputs = [
        ClassificationInput(
            system_prompt="You are a helpful assistant.",
            user_input="What is Python?",
            model_output="Python is a high-level programming language known for its readability and versatility."
        ),
        ClassificationInput(
            system_prompt="You are a math tutor.",
            user_input="What is 10 * 5?",
            model_output="10 multiplied by 5 equals 60."  # Wrong answer
        ),
        ClassificationInput(
            system_prompt="You are a helpful assistant.",
            user_input="Tell me a joke.",
            model_output="Why did the scarecrow win an award? Because he was outstanding in his field!"
        ),
    ]

    print("Classifying batch of responses in parallel...")
    print("=" * 50)
    
    # Classify all inputs in parallel
    results = await classifier.classify_batch_async(inputs)

    for i, (inp, result) in enumerate(zip(inputs, results), 1):
        print(f"\n--- Response {i} ---")
        print(f"User: {inp.user_input}")
        print(f"Model: {inp.model_output[:50]}...")
        print(f"Verdict: {result.verdict.value}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning[:100]}...")

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    good = sum(1 for r in results if r.verdict == ClassificationVerdict.GOOD)
    bad = sum(1 for r in results if r.verdict == ClassificationVerdict.BAD)
    uncertain = sum(1 for r in results if r.verdict == ClassificationVerdict.UNCERTAIN)
    print(f"  Good: {good}, Bad: {bad}, Uncertain: {uncertain}")


if __name__ == "__main__":
    asyncio.run(main())

