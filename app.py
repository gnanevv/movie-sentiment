import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# Load model and tokenizer (assuming paths are set)
tokenizer = BertTokenizer.from_pretrained("models/sentiment_model")
model = BertForSequenceClassification.from_pretrained("models/sentiment_model")

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
    return sentiment, probs

st.title("Movie Sentiment Analyzer")
review = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    if review:
        sentiment, probs = predict_sentiment(review, tokenizer, model)
        st.success(f"Sentiment: **{sentiment}**")

        # Word cloud
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(review)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # Sentiment confidence bar chart
        st.subheader("Sentiment Confidence")
        confidence_scores = probs[0].detach().numpy() * 100  # Convert to percentages
        fig = px.bar(
            x=["Negative", "Positive"],
            y=confidence_scores,
            labels={"x": "Sentiment", "y": "Confidence (%)"},
            title="Model Confidence Scores",
            color=["Negative", "Positive"],
            color_discrete_map={"Negative": "#FF6B6B", "Positive": "#4CAF50"}
        )
        fig.update_layout(width=600, height=400)
        st.plotly_chart(fig)
    else:
        st.error("Please enter a review!")