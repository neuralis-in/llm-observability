"""Benchmark for ToxicityDetectionEval using WikiToxic dataset."""

from datasets import load_dataset
from openai import OpenAI
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from aiobs.evals import ToxicityDetectionEval, EvalInput

# Load the WikiToxic dataset
print("Loading WikiToxic dataset...")
dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="train")
print(f"Loaded {len(dataset)} examples")

# Take a sample for benchmarking (1000 examples)
sample_size = 1000
dataset = dataset.shuffle(seed=42).select(range(sample_size))

# Initialize OpenAI client
client = OpenAI()

# Initialize evaluator
evaluator = ToxicityDetectionEval(
    client=client,
    model="gpt-4o",
)

# Run benchmark
print(f"Running toxicity detection on {sample_size} examples...")
start_time = time.time()

predictions = []
scores = []
ground_truth = []

for i, example in enumerate(dataset):
    if i % 50 == 0:
        print(f"Processing example {i+1}/{sample_size}")
    
    # Create eval input
    eval_input = EvalInput(
        user_input="",
        model_output=example["comment_text"]
    )
    
    # Run evaluation
    result = evaluator.evaluate(eval_input)
    
    # Store results
    toxicity_score = 1 - result.score  # Convert to toxicity score (0=non-toxic, 1=toxic)
    predictions.append(1 if toxicity_score >= evaluator.config.toxicity_threshold else 0)
    scores.append(toxicity_score)
    ground_truth.append(1 if example["toxic"] else 0)

# Calculate metrics
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)
accuracy = accuracy_score(ground_truth, predictions)

# Calculate inference speed
total_time = time.time() - start_time
speed = sample_size / total_time

# Print results
print("\nBenchmark Results:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Total Time: {total_time:.2f} seconds")
print(f"Speed: {speed:.2f} examples/second")

# Save detailed results
with open("toxicity_benchmark_results.csv", "w") as f:
    f.write("comment,ground_truth,prediction,score\n")
    for i, example in enumerate(dataset):
        f.write(f"\"{example['comment_text']}\","
                f"{ground_truth[i]},{predictions[i]},{scores[i]}\n")

print("Detailed results saved to toxicity_benchmark_results.csv")