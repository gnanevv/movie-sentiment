from datasets import load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import torch

# Load IMDB data and emotion model
dataset = load_from_disk("data/processed_data")  # Original IMDB data
tokenizer = BertTokenizer.from_pretrained("models/emotion_model")
model = BertForSequenceClassification.from_pretrained("models/emotion_model", num_labels=2)

# Evaluate
model.eval()
predictions, labels = [], []
for example in dataset["test"]:
    inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions.append(torch.argmax(outputs.logits, dim=1).item())
    labels.append(example["label"])

accuracy = accuracy_score(labels, predictions)
print(f"Transfer Learning Accuracy: {accuracy:.4f}")