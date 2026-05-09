# train/train.py
# This is where we actually TEACH our AI model

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./outputs/llama3-code-rag"
DATASET_PATH = "./data/train.jsonl"

def load_training_data(path):
    print(f"📂 Loading training data from {path}...")
    samples = []
    with open(path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"✅ Loaded {len(samples)} training samples")
    return samples


def format_prompt(sample):
    context_section = ""
    if sample.get("context"):
        context_section = f"\n\n### Retrieved Context:\n{sample['context']}"
    
    return f"""### Instruction:
{sample['instruction']}{context_section}

### Response:
{sample['response']}"""


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def setup_model_and_tokenizer():
    print(f"🦙 Loading {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=os.getenv("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HF_TOKEN")
    )
    
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def add_lora_adapters(model):
    print("🔧 Adding LoRA adapters...")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train():
    raw_data = load_training_data(DATASET_PATH)
    formatted = [{"text": format_prompt(s)} for s in raw_data]
    dataset = Dataset.from_list(formatted)
    
    model, tokenizer = setup_model_and_tokenizer()
    model = add_lora_adapters(model)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        report_to="wandb",
        run_name="llama3-code-rag-v1",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )
    
    print("🚀 Starting training!")
    trainer.train()
    
    print(f"💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Training complete!")


if __name__ == "__main__":
    train()