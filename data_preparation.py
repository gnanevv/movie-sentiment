import os
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer
import torch

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

try:
    # Load a small subset of the IMDB dataset (1000 samples for beginners)
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", split="train[:1000]")
    df = pd.DataFrame({"text": dataset["text"], "label": dataset["label"]})  # 0 = negative, 1 = positive

    # Initialize BERT tokenizer
    print("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize the text
    def tokenize_data(texts):
        return tokenizer(
            texts,
            padding=True,  # Pad shorter texts to same length
            truncation=True,  # Cut longer texts to fit
            max_length=128,  # Limit length to save memory
            return_tensors="pt"  # Return PyTorch tensors
        )

    # Apply tokenization
    print("Tokenizing data...")
    encoded_data = tokenize_data(df["text"].tolist())

    # Save tokenized data and labels
    output_path = "data/processed_data.pt"
    torch.save({
        "input_ids": encoded_data["input_ids"],
        "attention_mask": encoded_data["attention_mask"],
        "labels": torch.tensor(df["label"].tolist())
    }, output_path)
    print(f"Data preparation complete! Saved to {output_path}")

except Exception as e:
    print(f"Error during data preparation: {e}")
    exit(1)