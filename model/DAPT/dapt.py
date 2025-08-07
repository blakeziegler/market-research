from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import torch
from transformers.trainer_callback import EarlyStoppingCallback

# Config
model_name = "Dev9124/qwen3-finance-model"
output_dir = "v4"

# Load tokenizer and FP16 base model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

dataset = load_dataset("text", data_files={"train": "data/raw-text/*.txt"})["train"]
dataset = dataset.filter(lambda x: bool(x["text"].strip()), num_proc=4)

# Tokenization
MAX_LENGTH = 8192
STRIDE = 4096

def chunk_with_tokenizer(batch):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_attention_mask=True,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

tokenized_dataset = dataset.map(
    chunk_with_tokenizer,
    batched=True,
    remove_columns=["text"],
    num_proc=4
).flatten_indices()

# Validate
def check_token_types(dataset, num_batches=5):
    for i, example in enumerate(dataset):
        if i >= num_batches:
            break
        assert all(isinstance(x, int) for x in example["input_ids"])
        assert all(isinstance(x, int) for x in example["attention_mask"])
    print("✅ Token types valid")

check_token_types(tokenized_dataset)

split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=1e-5,
    fp16=True,
    eval_strategy="steps",
    eval_delay=500,
    save_strategy="steps",
    eval_steps=500,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    metric_for_best_model="eval_loss",
    save_steps=500,
    logging_steps=10,
    save_total_limit=2,
    gradient_checkpointing=True,
)

# Data Collator
class SafeDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["input_ids"] = batch["input_ids"].long()
        batch["attention_mask"] = batch["attention_mask"].long()
        return batch

data_collator = SafeDataCollator(tokenizer=tokenizer, mlm=False)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()