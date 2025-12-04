"""Example: Hallucination Detection Evaluation

This example demonstrates how to use the HallucinationDetectionEval
to detect hallucinations in LLM outputs using an LLM-as-judge approach.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import OpenAI

from aiobs.evals import (
    HallucinationDetectionEval,
    HallucinationDetectionConfig,
    EvalInput,
)


def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create the hallucination detection evaluator
    evaluator = HallucinationDetectionEval(
        client=client,
        model="gpt-4o-mini",  # Judge model
    )
    
    print("=" * 60)
    print("Hallucination Detection Evaluation Examples")
    print("=" * 60)
    
    # Example 1: No hallucination (grounded response)
    print("\n--- Example 1: Grounded Response (No Hallucination) ---")
    result1 = evaluator.evaluate(
        EvalInput(
            user_input="What is the capital of France?",
            model_output="The capital of France is Paris.",
            context={
                "documents": [
                    "Paris is the capital and largest city of France.",
                    "France is a country in Western Europe.",
                ]
            }
        )
    )
    print(f"Status: {result1.status}")
    print(f"Score: {result1.score}")
    print(f"Message: {result1.message}")
    if result1.details:
        print(f"Analysis: {result1.details.get('analysis', 'N/A')}")
    
    # Example 2: Contains hallucination
    print("\n--- Example 2: Response with Hallucination ---")
    result2 = evaluator.evaluate(
        EvalInput(
            user_input="What is the capital of France?",
            model_output="The capital of France is Paris. It was founded in 250 BC by Julius Caesar and has a population of exactly 15 million people.",
            context={
                "documents": [
                    "Paris is the capital and largest city of France.",
                    "The Paris metropolitan area has a population of around 12 million.",
                ]
            }
        )
    )
    print(f"Status: {result2.status}")
    print(f"Score: {result2.score}")
    print(f"Message: {result2.message}")
    if result2.details:
        print(f"Analysis: {result2.details.get('analysis', 'N/A')}")
        if result2.details.get("hallucinations"):
            print("Detected hallucinations:")
            for h in result2.details["hallucinations"]:
                print(f"  - [{h.get('severity', 'unknown')}] {h.get('claim', 'Unknown')}")
                print(f"    Reason: {h.get('reason', 'N/A')}")
    
    # Example 3: Completely fabricated response
    print("\n--- Example 3: Completely Fabricated Response ---")
    result3 = evaluator.evaluate(
        EvalInput(
            user_input="Who wrote Romeo and Juliet?",
            model_output="Romeo and Juliet was written by Charles Dickens in 1920. It was his first novel and won the Nobel Prize in Literature.",
            context={
                "documents": [
                    "Romeo and Juliet is a tragedy written by William Shakespeare.",
                    "The play was written between 1591 and 1596.",
                ]
            }
        )
    )
    print(f"Status: {result3.status}")
    print(f"Score: {result3.score}")
    print(f"Message: {result3.message}")
    if result3.details:
        print(f"Analysis: {result3.details.get('analysis', 'N/A')}")
    
    # Example 4: Using strict mode
    print("\n--- Example 4: Strict Mode ---")
    strict_evaluator = HallucinationDetectionEval(
        client=client,
        model="gpt-4o-mini",
        config=HallucinationDetectionConfig(
            strict=True,  # Any hallucination fails
            hallucination_threshold=0.8,  # Higher threshold
        )
    )
    result4 = strict_evaluator.evaluate(
        EvalInput(
            user_input="What year was Python created?",
            model_output="Python was created in 1991 by Guido van Rossum. It was named after Monty Python's Flying Circus and initially released on February 20, 1991.",
            context={
                "documents": [
                    "Python was conceived in the late 1980s by Guido van Rossum.",
                    "Python was first released in 1991.",
                ]
            }
        )
    )
    print(f"Status: {result4.status}")
    print(f"Score: {result4.score}")
    print(f"Message: {result4.message}")
    
    # Example 5: Without context (general factuality check)
    print("\n--- Example 5: No Context (General Factuality) ---")
    result5 = evaluator.evaluate(
        EvalInput(
            user_input="Who is the current CEO of Apple?",
            model_output="The current CEO of Apple is Tim Cook. He took over from Steve Jobs in 2011.",
        )
    )
    print(f"Status: {result5.status}")
    print(f"Score: {result5.score}")
    print(f"Message: {result5.message}")
    if result5.details:
        print(f"Analysis: {result5.details.get('analysis', 'N/A')}")


if __name__ == "__main__":
    main()

