# data/collect_data.py
# This file collects coding Q&A pairs to teach our AI model

import json
import os
from datasets import load_dataset
from tqdm import tqdm

def collect_stackoverflow_data(num_samples=5000):
    print("📚 Collecting Stack Overflow coding data...")
    
    dataset = load_dataset(
        "koutch/stackoverflow_python",
        split="train",
        trust_remote_code=True
    )
    
    samples = []
    for item in tqdm(dataset.select(range(min(num_samples, len(dataset))))):
        sample = {
            "instruction": item.get("question_body", ""),
            "context": f"Title: {item.get('question_title', '')}\nTags: {item.get('tags', '')}",
            "response": item.get("answer_body", "")
        }
        if sample["instruction"] and sample["response"]:
            samples.append(sample)
    
    print(f"✅ Collected {len(samples)} samples!")
    return samples


def collect_synthetic_rag_data():
    print("🧪 Creating synthetic RAG training examples...")
    
    rag_examples = [
        {
            "instruction": "How do I create a FastAPI endpoint with dependency injection?",
            "context": "[Retrieved Doc]: FastAPI supports dependency injection via the Depends() function. You declare dependencies as function parameters. FastAPI handles the injection automatically at request time.",
            "response": "Based on the documentation, you can use FastAPI's `Depends()` for dependency injection:\n\n```python\nfrom fastapi import FastAPI, Depends\n\napp = FastAPI()\n\ndef get_db():\n    db = DatabaseSession()\n    try:\n        yield db\n    finally:\n        db.close()\n\n@app.get('/users')\ndef get_users(db = Depends(get_db)):\n    return db.query(User).all()\n```"
        },
        {
            "instruction": "My PyTorch model is using too much GPU memory. How do I fix it?",
            "context": "[Retrieved Doc]: Common causes of GPU memory issues in PyTorch: 1) Not clearing gradients with optimizer.zero_grad() 2) Keeping computation graphs alive 3) Large batch sizes. Use torch.cuda.empty_cache() to free unused memory.",
            "response": "Based on the retrieved context, here are the fixes:\n\n```python\n# Fix 1: Always zero gradients\noptimizer.zero_grad()\nloss.backward()\noptimizer.step()\n\n# Fix 2: Use no_grad for inference\nwith torch.no_grad():\n    output = model(input)\n\n# Fix 3: Clear cache\ntorch.cuda.empty_cache()\n```"
        },
    ]
    
    print(f"✅ Created {len(rag_examples)} RAG training examples!")
    return rag_examples


def save_dataset(samples, filename):
    filepath = os.path.join("data", filename)
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"💾 Saved {len(samples)} samples to {filepath}")


if __name__ == "__main__":
    so_data = collect_stackoverflow_data(num_samples=2000)
    rag_data = collect_synthetic_rag_data()
    all_data = so_data + rag_data
    
    split_point = int(len(all_data) * 0.9)
    train_data = all_data[:split_point]
    eval_data = all_data[split_point:]
    
    save_dataset(train_data, "train.jsonl")
    save_dataset(eval_data, "eval.jsonl")
    
    print(f"\n🎉 Done! {len(train_data)} training samples, {len(eval_data)} eval samples")