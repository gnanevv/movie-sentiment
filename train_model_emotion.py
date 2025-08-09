import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
import os

# --- 1. Explicitly set the device ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU.")

# Create directories
os.makedirs("results/emotion", exist_ok=True)
os.makedirs("logs/emotion", exist_ok=True)

# Load data
dataset = load_from_disk("data/go_emotions")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    # --- 2. Reduced max_length ---
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128) # Reduced from 512
    tokenized["labels"] = examples["label"]
    return tokenized

# Tokenize and set format
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# Load model and move to the correct device
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=28).to(device)

# --- 3. Optimized Training Arguments ---
training_args = TrainingArguments(
    output_dir="results/emotion",
    eval_strategy="epoch",
    per_device_train_batch_size=8,  # Can potentially be slightly increased with optimizations
    per_device_eval_batch_size=8,
    num_train_epochs=2,  # Reduced from 3
    logging_dir="logs/emotion",
    gradient_accumulation_steps=2,
    bf16=True if device.type == 'mps' else False,  # Use bfloat16 for MPS mixed precision
    save_strategy="epoch",
    save_total_limit=2,
    # Use less frequent logging to reduce overhead
    logging_steps=100,
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
model.save_pretrained("models/emotion_model")
tokenizer.save_pretrained("models/emotion_model")