from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
import os
import torch

# Create directories
os.makedirs("results/sentiment", exist_ok=True)
os.makedirs("logs/sentiment", exist_ok=True)

# Load data
dataset = load_from_disk("data/processed_data")  # Fixed path
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = examples["label"]
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

# Training arguments optimized for M1 Pro
training_args = TrainingArguments(
    output_dir="results/sentiment",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="logs/sentiment",
    gradient_accumulation_steps=2,
    save_strategy="epoch",
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train and save
trainer.train()
model.save_pretrained("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")