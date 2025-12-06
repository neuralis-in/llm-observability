"""Benchmark HallucinationDetectionEval against HaluEvalQA."""

import json
import requests   #newly added dependency
import asyncio
from pathlib import Path
from typing import Dict, List
from google import genai
from tqdm import tqdm     #newly added dependency
from aiobs.evals import HallucinationDetectionEval, EvalInput
from aiobs import observer


# configurations (defaults)
JUDGE_MODEL = "gemini-2.5-flash"               # judge model: gemini-flash-2.5 as of now  
SAMPLE_SIZE = None                  # None = full dataset, or set e.g., 200
OUTPUT_FILE = "halueval_results.json"
DATA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


def load_halueval_qa(url: str) -> List[Dict]:
    """Fetch HaluEvalQA dataset from GitHub raw using HTTP GET."""
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.strip().split("\n")
    return [json.loads(line) for line in lines]


def run_benchmark(
    evaluator: HallucinationDetectionEval,
    data: List[Dict],
    sample_size: int = None,
) -> Dict:
    """Run benchmark and return metrics."""
    results = {
        "hallucinated_detected": 0,  # True positives
        "hallucinated_missed": 0,    # False negatives
        "correct_passed": 0,         # True negatives
        "correct_flagged": 0,        # False positives
    }
    
    samples = data[:sample_size] if sample_size else data
    
    for sample in tqdm(samples, desc="Checking on benchmarks"):
        knowledge = sample["knowledge"]
        question = sample["question"]
        
        # Test hallucinated answer (should fail)
        result_hallucinated = evaluator.evaluate(EvalInput(
            user_input=question,
            model_output=sample["hallucinated_answer"],
            context={"documents": [knowledge]},
        ))
        
        if result_hallucinated.failed:
            results["hallucinated_detected"] += 1
        else:
            results["hallucinated_missed"] += 1
        
        # Test correct answer (should pass)
        result_correct = evaluator.evaluate(EvalInput(
            user_input=question,
            model_output=sample["right_answer"],
            context={"documents": [knowledge]},
        ))
        
        if result_correct.passed:
            results["correct_passed"] += 1
        else:
            results["correct_flagged"] += 1
    
    # Calculate metrics
    total = len(samples) * 2
    accuracy = (results["hallucinated_detected"] + results["correct_passed"]) / total
    
    precision = results["hallucinated_detected"] / (
        results["hallucinated_detected"] + results["correct_flagged"]
    ) if (results["hallucinated_detected"] + results["correct_flagged"]) > 0 else 0
    
    recall = results["hallucinated_detected"] / (
        results["hallucinated_detected"] + results["hallucinated_missed"]
    ) if (results["hallucinated_detected"] + results["hallucinated_missed"]) > 0 else 0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "raw_counts": results,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": len(samples),
    }

def main():
    observer.observe("Hallucination Benchmarking")
    
    dataset = load_halueval_qa(DATA_URL)

    client = genai.Client()

    evaluation_model = HallucinationDetectionEval.with_gemini(client=client , model=JUDGE_MODEL)

    metrics = run_benchmark(evaluator=evaluation_model , data=dataset , sample_size=SAMPLE_SIZE)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(metrics,f,indent=2)

    observer.end()
    observer.flush(OUTPUT_FILE)


#===================TESTING ONLY=============

if(__name__ == "__main__"):
    import os
    from dotenv import load_dotenv
    from getpass import getpass
    load_dotenv()
    SAMPLE_SIZE = 10
    if not (gemini_api_key := os.getenv("GEMINI_API_KEY")):
        gemini_api_key = getpass("Enter your Gemini API key: ")
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    main()
