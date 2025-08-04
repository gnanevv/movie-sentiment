from datasets import load_dataset
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load SetFit/sst5 dataset for 5-class sentiment
try:
    dataset = load_dataset("SetFit/sst5")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Preprocess data
def preprocess(example):
    return {"text": example["text"], "label": example["label"]}

processed_dataset = dataset.map(preprocess)
processed_dataset.save_to_disk("data/sst5_data")