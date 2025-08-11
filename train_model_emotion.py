from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
import os
import torch

# Create directories
os.makedirs("results/emotion", exist_ok=True)
os.makedirs("logs/emotion", exist_ok=True)

# Step 1: Train on GoEmotions
dataset = load_from_disk("data/go_emotions")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = examples["label"]
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=28)
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

training_args = TrainingArguments(
    output_dir="results/emotion",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="logs/emotion",
    gradient_accumulation_steps=2,
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
model.save_pretrained("models/emotion_model_goemotions")
tokenizer.save_pretrained("models/emotion_model_goemotions")

# Step 2: Fine-tune on IMDB
dataset = load_from_disk("data/processed_data")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

model = DistilBertForSequenceClassification.from_pretrained("models/emotion_model_goemotions", num_labels=2, ignore_mismatched_sizes=True)
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

training_args = TrainingArguments(
    output_dir="results/emotion_finetuned",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,  # Fewer epochs for fine-tuning
    logging_dir="logs/emotion_finetuned",
    gradient_accumulation_steps=2,
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
model.save_pretrained("models/emotion_model_finetuned")
tokenizer.save_pretrained("models/emotion_model_finetuned")