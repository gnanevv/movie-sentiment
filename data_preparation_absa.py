from datasets import load_dataset
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# Load a sample ABSA dataset (replace with SemEval if available)
try:
    dataset = load_dataset("yelp_review_full")  # Fallback dataset
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Preprocess (simplify for demo; adapt for actual ABSA dataset)
def preprocess(example):
    return {"text": example["text"], "label": example["label"]}

processed_dataset = dataset.map(preprocess)
processed_dataset.save_to_disk("data/yelp_absa")