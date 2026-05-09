# eval/evaluate.py
# This checks how good our AI model is — like grading a test!

import json
import os
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()


def load_eval_data(path):
    """Load our test questions"""
    print(f"📂 Loading eval data from {path}...")
    samples = []
    with open(path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"✅ Loaded {len(samples)} eval samples")
    return samples


def prepare_ragas_dataset(samples):
    """
    Format data for RAGAS evaluation.
    RAGAS needs: question, answer, contexts, ground_truth
    """
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for sample in samples:
        ragas_data["question"].append(sample["instruction"])
        ragas_data["answer"].append(sample["response"])
        ragas_data["contexts"].append([sample.get("context", "")])
        ragas_data["ground_truth"].append(sample["response"])

    return Dataset.from_dict(ragas_data)


def run_evaluation(eval_path="./data/eval.jsonl"):
    """Run the full evaluation — like grading all the test papers"""
    
    # Load eval data
    samples = load_eval_data(eval_path)
    
    # Only use first 50 samples for speed
    samples = samples[:50]
    
    # Prepare dataset
    print("📊 Preparing evaluation dataset...")
    dataset = prepare_ragas_dataset(samples)
    
    # Run RAGAS evaluation
    print("🧪 Running RAGAS evaluation...")
    print("This checks:")
    print("  ✅ Faithfulness — does the answer match the context?")
    print("  ✅ Answer Relevancy — is the answer relevant to the question?")
    print("  ✅ Context Precision — is the retrieved context useful?")
    print("  ✅ Context Recall — did we retrieve enough context?")
    
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )
    
    # Print results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"Faithfulness:      {results['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['context_precision']:.3f}")
    print(f"Context Recall:    {results['context_recall']:.3f}")
    print("="*60)
    
    # Save results to file
    results_path = "./eval/results.json"
    with open(results_path, "w") as f:
        json.dump(dict(results), f, indent=2)
    print(f"💾 Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    run_evaluation()