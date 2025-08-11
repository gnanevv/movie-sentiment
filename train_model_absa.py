from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
import os
import torch

# Create directories
os.makedirs("results/absa", exist_ok=True)
os.makedirs("logs/absa", exist_ok=True)

# Load data
dataset = load_from_disk("data/yelp_absa")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = examples["label"]  # Rename 'label' to 'labels'
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

# Training arguments optimized for speed
training_args = TrainingArguments(
    output_dir="results/absa",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir="logs/absa",
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=False,
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
model.save_pretrained("models/absa_model")
tokenizer.save_pretrained("models/absa_model")