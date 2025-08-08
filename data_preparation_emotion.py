from datasets import load_dataset
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# Load GoEmotions dataset
try:
    dataset = load_dataset("go_emotions")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Preprocess: Ensure single label (first emotion)
def preprocess(example):
    # Take the first label if multiple exist, or handle single label
    label = example["labels"][0] if isinstance(example["labels"], list) else example["labels"]
    return {"text": example["text"], "label": label}

processed_dataset = dataset.map(preprocess)
processed_dataset.save_to_disk("data/go_emotions")

# Verify dataset structure
print("Sample from train split:", processed_dataset["train"][0])
print("Dataset splits:", processed_dataset.keys())