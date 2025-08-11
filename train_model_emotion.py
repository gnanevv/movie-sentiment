from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
import os
import torch

# Create directories
os.makedirs("results/emotion", exist_ok=True)
os.makedirs("logs/emotion", exist_ok=True)

# Load IMDB data (small subset)
dataset = load_from_disk("data/processed_data")
# Use a small subset for speed
dataset = dataset.shuffle(seed=42)
train_dataset = dataset["train"].select(range(1000))  # 1000 samples
eval_dataset = dataset["test"].select(range(200))    # 200 samples

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)  # Shorter max_length
    tokenized["labels"] = examples["label"]
    return tokenized

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

# Training arguments optimized for speed
training_args = TrainingArguments(
    output_dir="results/emotion",
    eval_strategy="epoch",
    per_device_train_batch_size=8,  # Larger batch size
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # Single epoch
    logging_dir="logs/emotion",
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=False,  # Disable mixed precision if MPS has issues
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Train and save
trainer.train()
model.save_pretrained("models/emotion_model")
tokenizer.save_pretrained("models/emotion_model")