import os
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch

try:
    # Check if model exists
    model_path = "models/sentiment_model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Streamlit app
    st.title("Movie Review Sentiment Analysis")
    st.write("Enter a movie review to predict if it's positive or negative.")

    # Text input
    user_input = st.text_area("Your Review:", "Type your movie review here...")

    if st.button("Predict"):
        if not user_input.strip():
            st.error("Please enter a review.")
        else:
            # Tokenize input
            inputs = tokenizer(user_input, padding=True, truncation=True, max_length=128, return_tensors="pt")
            
            # Predict
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
            
            # Display result
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.write(f"Sentiment: **{sentiment}**")

except Exception as e:
    st.error(f"Error in app: {e}")