from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
import os

# Create directories
os.makedirs("results/emotion", exist_ok=True)
os.makedirs("logs/emotion", exist_ok=True)

# Load data
dataset = load_from_disk("data/go_emotions")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = examples["label"]  # Ensure single integer label
    return tokenized

# Tokenize and set format
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# Load model for emotion classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)  # GoEmotions has 28 labels

# Training arguments
training_args = TrainingArguments(
    output_dir="results/emotion",
    eval_strategy="epoch",  # Updated from evaluation_strategy
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="logs/emotion",
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