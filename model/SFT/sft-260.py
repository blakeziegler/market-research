import pandas as pd
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

def convert_csv_to_chatml():
    """Convert CSV to ChatML format"""
    print("Loading combined CSV...")
    df = pd.read_csv("synthetic-data/combined_sft-260.csv")
    
    def convert_to_chatml(row):
        return {
            "type": "chatml",
            "messages": [
                {"role": "system", "content": "You are a helpful financial analysis assistant."},
                {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": row["assistant"]}
            ]
        }
    
    print(f"Converting {len(df)} rows to ChatML format...")
    with open("sft-260_chatml.jsonl", "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(convert_to_chatml(row), ensure_ascii=False) + "\n")
    
    print("Saved as sft-260_chatml.jsonl")

def train_model():
    """Perform supervised fine-tuning"""
    print("Loading model and tokenizer...")
    model_name = "blakeziegler/qwen3_4b_dapt-700k_v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading dataset...")
    dataset = load_dataset("json", data_files="sft-260_chatml.jsonl", split="train")
    
    def format_chatml(example):
        messages = example["messages"]
        conversation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": conversation}
    
    dataset = dataset.map(format_chatml)
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            padding=False,
            truncation=True,
            max_length=16384,
            return_tensors="pt",
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])
    
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir="./qwen3_4b_dapt-700k_SFT-260_v1",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")

if __name__ == "__main__":
    convert_csv_to_chatml()
    train_model()
