from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt
import torch

# === Model Configuration ===
model_name = "tarun7r/Finance-Llama-8B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Load and Clean Dataset ===
dataset = load_dataset("text", data_files={"train": "data/raw-text/*.txt"})["train"]
dataset = dataset.filter(lambda x: bool(x["text"].strip()), num_proc=4)

# === Tokenization and Chunking ===
MAX_LENGTH = 4096
STRIDE = 256

def chunk_with_tokenizer(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_attention_mask=True
    )
    return [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"])
    ]

# Apply tokenization and flatten
tokenized_dataset = dataset.map(
    chunk_with_tokenizer,
    batched=False,
    remove_columns=["text"],
    num_proc=4
).flatten_indices()

# === Validate Token Types ===
def check_token_types(dataset, num_batches=5):
    print("Checking token types...")
    for i, example in enumerate(dataset):
        if i >= num_batches:
            break
        if not all(isinstance(x, int) for x in example["input_ids"]):
            raise TypeError(f"input_ids not all ints at index {i}")
        if not all(isinstance(x, int) for x in example["attention_mask"]):
            raise TypeError(f"attention_mask not all ints at index {i}")
    print("✅ All token types are valid")

check_token_types(tokenized_dataset)

# === Plot Histogram of Sequence Lengths ===
lengths = [len(x["input_ids"]) for x in tokenized_dataset]
plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="v2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=5e-5,
    fp16=True,
    save_steps=500,
    logging_steps=10,
    save_total_limit=2,
)

# === Safe Data Collator ===
class SafeDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["input_ids"] = batch["input_ids"].long()
        batch["attention_mask"] = batch["attention_mask"].long()
        return batch

data_collator = SafeDataCollator(
    tokenizer=tokenizer,
    mlm=False
)

# === Trainer Initialization ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Start Training ===
trainer.train()