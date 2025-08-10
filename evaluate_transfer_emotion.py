from datasets import load_from_disk
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score
import torch

# Load IMDB data and emotion model
dataset = load_from_disk("data/processed_data")
tokenizer = DistilBertTokenizer.from_pretrained("models/emotion_model")
model = DistilBertForSequenceClassification.from_pretrained("models/emotion_model", num_labels=2)
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

# Evaluate
model.eval()
predictions, labels = [], []
for example in dataset["test"]:
    inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions.append(torch.argmax(outputs.logits, dim=1).item())
    labels.append(example["label"])

accuracy = accuracy_score(labels, predictions)
print(f"Transfer Learning Accuracy: {accuracy:.4f}")