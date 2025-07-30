import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

try:
    # Check if processed data exists
    data_path = "data/processed_data.pt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at {data_path}. Run data_preparation.py first.")

    # Load preprocessed data
    print("Loading preprocessed data...")
    data = torch.load(data_path)

    # Create a custom dataset class for PyTorch
    class IMDBDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    # Prepare dataset
    print("Preparing dataset...")
    dataset = IMDBDataset({"input_ids": data["input_ids"], "attention_mask": data["attention_mask"]}, data["labels"])

    # Load pre-trained BERT model
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,  # Short training for beginners
        per_device_train_batch_size=8,  # Small batch to save memory
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize trainer
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("models/sentiment_model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("models/sentiment_model")
    print("Model training complete! Model saved to models/sentiment_model")

except Exception as e:
    print(f"Error during training: {e}")
    exit(1)