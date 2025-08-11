from datasets import load_from_disk
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load data and model
dataset = load_from_disk("data/yelp_absa")
tokenizer = DistilBertTokenizer.from_pretrained("models/absa_model")
model = DistilBertForSequenceClassification.from_pretrained("models/absa_model", num_labels=3)
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")
test_dataset = dataset["test"]

# Evaluate
model.eval()
predictions, labels = [], []
for example in test_dataset:
    inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions.append(torch.argmax(outputs.logits, dim=1).item())
    labels.append(example["label"])

accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average="weighted")
print(f"ABSA Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")