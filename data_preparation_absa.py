from datasets import load_dataset, DatasetDict
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# Load Yelp dataset
dataset = load_dataset("yelp_review_full")

# Map 5 labels to 3: 0,1->Negative (0), 2->Neutral (1), 3,4->Positive (2)
def preprocess(example):
    label = example["label"]
    if label in [0, 1]:
        new_label = 0  # Negative
    elif label == 2:
        new_label = 1  # Neutral
    else:
        new_label = 2  # Positive
    return {"text": example["text"], "label": new_label}

# Use small subset for speed
dataset = dataset.shuffle(seed=42)
small_dataset = DatasetDict({
    "train": dataset["train"].select(range(1000)),  # 1000 samples
    "test": dataset["test"].select(range(200))      # 200 samples
})

# Process the dataset
processed_dataset = small_dataset.map(preprocess)

# Save to disk
processed_dataset.save_to_disk("data/yelp_absa")