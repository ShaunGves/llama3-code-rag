# inference/pipeline.py
# This ties everything together: retrieve → generate → answer

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from retrieval.retriever import HybridRetriever
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "./outputs/llama3-code-rag"


class CodeRAGPipeline:
    def __init__(self):
        print("🚀 Loading CodeRAG Pipeline...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            token=os.getenv("HF_TOKEN")
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()
        
        self.retriever = HybridRetriever()
        
        print("✅ Pipeline ready!")
    
    def answer(self, question: str, top_k: int = 3) -> dict:
        print(f"🔍 Retrieving context for: '{question[:50]}...'")
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        context = self.retriever.format_context(retrieved_docs)
        
        prompt = f"""### Instruction:
{question}

### Retrieved Context:
{context}

### Response:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output.split("### Response:")[-1].strip()
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs
        }


if __name__ == "__main__":
    rag = CodeRAGPipeline()
    result = rag.answer("How do I handle async database calls in FastAPI?")
    
    print("\n" + "="*60)
    print("QUESTION:", result["question"])
    print("="*60)
    print("ANSWER:", result["answer"])
    print("="*60)
    print("SOURCES USED:", len(result["sources"]))