from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

# Load data
dataset = load_from_disk("data/sst5_data")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# Load model for 5-class classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Training arguments
training_args = TrainingArguments(
    output_dir="results/multi_class",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="logs/multi_class",
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
model.save_pretrained("models/sentiment_model_multi")
tokenizer.save_pretrained("models/sentiment_model_multi")