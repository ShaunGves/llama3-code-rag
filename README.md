# 🦙 Llama 3 Code RAG — Fine-Tuned Domain Model

A production-grade RAG-grounded code assistant built on Llama 3-8B, 
fine-tuned with QLoRA on 50K+ coding samples with a hybrid dense-sparse retrieval pipeline.

## 🏗️ Architecture
- **Base Model**: Meta-Llama-3-8B-Instruct
- **Fine-Tuning**: QLoRA (4-bit, LoRA r=16)
- **Retrieval**: Hybrid (Qdrant dense + BM25 sparse)
- **Evaluation**: RAGAS faithfulness + code correctness

## 📁 Project Structure
## ⚙️ Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## 🚀 Usage
```python
from inference.pipeline import CodeRAGPipeline
rag = CodeRAGPipeline()
result = rag.answer("How do I fix memory leaks in PyTorch?")
print(result["answer"])
```

## 📊 Evaluation
```bash
python eval/evaluate.py
```

## 👤 Author
Built as a portfolio project demonstrating fine-tuning and RAG pipelines.
