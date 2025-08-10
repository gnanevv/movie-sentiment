from datasets import load_dataset
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# Load SST-5 dataset
try:
    dataset = load_dataset("SetFit/sst5")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Preprocess
def preprocess(example):
    return {"text": example["text"], "label": example["label"]}

processed_dataset = dataset.map(preprocess)
processed_dataset.save_to_disk("data/sst5_data")