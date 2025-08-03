from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = "tarun7r/Finance-Llama-8B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("text", data_files={"train": "data/raw-text/*.txt"})["train"]

dataset = dataset.filter(lambda x: bool(x["text"].strip()), num_proc=4)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=32768)

def filter_empty(example):
    return len(example["input_ids"]) > 0

tokenized_dataset = (
    dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
           .filter(filter_empty, num_proc=4)
)

def check_token_types(dataset, num_batches=5):
    print("Checking token types...")
    for i, example in enumerate(dataset):
        if i >= num_batches:
            break
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        if not all(isinstance(x, int) for x in input_ids):
            raise TypeError(f"Non-integer token found in input_ids at index {i}")
        if not all(isinstance(x, int) for x in attention_mask):
            raise TypeError(f"Non-integer token found in attention_mask at index {i}")
    print("✅ All token types are valid (int)")

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
check_token_types(tokenized_dataset)

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

class SafeDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["input_ids"] = batch["input_ids"].long()
        batch["attention_mask"] = batch["attention_mask"].long()
        return batch

data_collator = SafeDataCollator(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()