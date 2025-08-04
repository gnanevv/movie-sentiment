from datasets import load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load data and model
dataset = load_from_disk("data/sst5_data")
tokenizer = BertTokenizer.from_pretrained("models/sentiment_model_multi")
model = BertForSequenceClassification.from_pretrained("models/sentiment_model_multi")
test_dataset = dataset["test"]

# Evaluate
model.eval()
predictions, labels = [], []
for example in test_dataset:
    inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions.append(torch.argmax(outputs.logits, dim=1).item())
    labels.append(example["label"])

accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average="weighted")
print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")