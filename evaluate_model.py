import os
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score

try:
    # Check if model exists
    model_path = "models/sentiment_model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset("imdb", split="test[:100]")
    texts = test_dataset["text"]
    labels = test_dataset["label"]

    # Load trained model and tokenizer
    print("Loading model and tokenizer...")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Tokenize test data
    print("Tokenizing test data...")
    encoded_data = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # Predict
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_data)
        predictions = torch.argmax(outputs.logits, dim=1).numpy()

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"Test Accuracy: {accuracy:.2f}")

except Exception as e:
    print(f"Error during evaluation: {e}")
    exit(1)