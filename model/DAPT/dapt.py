from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

model_name = "Dev9124/qwen3-finance-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("text", data_files={"train": "data/raw-text/*.txt"})["train"]

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=32768)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)

training_args = TrainingArguments(
    output_dir="model/DAPT/dapt-models",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=5e-5,
    fp16=True,
    save_steps=500,
    logging_steps=10,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(
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











