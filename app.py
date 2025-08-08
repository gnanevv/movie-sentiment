import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import os

# Load binary sentiment model
binary_model_path = "/Users/georginanev/Documents/movie-sentiment/models/sentiment_model"  # Update with your actual path
try:
    tokenizer = BertTokenizer.from_pretrained(binary_model_path)
    model = BertForSequenceClassification.from_pretrained(binary_model_path)
    sentiment_pipeline = pipeline("sentiment-analysis", model=binary_model_path, tokenizer=binary_model_path)
except OSError as e:
    st.error(f"Error loading binary model: {e}")
    st.stop()

# Load multi-class model
multi_model_path = "/Users/georginanev/Documents/movie-sentiment/models/sentiment_model_multi"
try:
    multi_tokenizer = BertTokenizer.from_pretrained(multi_model_path)
    multi_model = BertForSequenceClassification.from_pretrained(multi_model_path)
except OSError as e:
    st.error(f"Error loading multi-class model: {e}")
    st.stop()

# Create SHAP explainer for binary model
explainer = shap.Explainer(sentiment_pipeline)

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
    return sentiment, probs

# Streamlit app
st.set_page_config(page_title="Movie Sentiment Analyzer", layout="wide")
st.title("🎬 Movie Sentiment Analyzer")
st.markdown("Analyze movie review sentiments with BERT and explore word importance.")

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Explainability", "Multi-Class Sentiment", "Transfer Learning"])

# Tab 1: Sentiment Analysis (Binary)
with tab1:
    st.header("Analyze Sentiment")
    review = st.text_area("Enter a movie review:", height=150, key="sentiment_input")
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
            confidence_scores = probs[0].detach().numpy() * 100
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

# Tab 2: Explainability
with tab2:
    st.header("Explain Sentiment")
    review = st.text_area("Enter a movie review for explanation:", height=150, key="explain_input")
    if st.button("Explain Sentiment"):
        if review:
            prediction = sentiment_pipeline(review)[0]
            st.success(f"Sentiment: **{prediction['label']}** (Score: {prediction['score']:.2f})")
            
            # SHAP explanation
            st.subheader("Word Importance")
            shap_values = explainer([review])
            plt.figure(figsize=(10, 5))
            shap.plots.text(shap_values, display=False)
            st.pyplot(plt)
        else:
            st.error("Please enter a review!")

# Tab 3: Multi-Class Sentiment
with tab3:
    st.header("Multi-Class Sentiment Analysis")
    review = st.text_area("Enter a movie review for multi-class prediction:", height=150, key="multiclass_input")
    if st.button("Analyze Multi-Class Sentiment"):
        if review:
            inputs = multi_tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = multi_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
            predicted_label = labels[torch.argmax(probs).item()]
            st.success(f"Sentiment: **{predicted_label}**")
            
            # Confidence bar chart
            st.subheader("Multi-Class Confidence")
            confidence_scores = probs[0].detach().numpy() * 100
            fig = px.bar(
                x=labels,
                y=confidence_scores,
                labels={"x": "Sentiment", "y": "Confidence (%)"},
                title="Multi-Class Confidence Scores",
                color=labels,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig)
        else:
            st.error("Please enter a review!")
# Tab 4: Transfer Learning (Placeholder)
# Tab 4: Transfer Learning
with tab4:
    st.header("Transfer Learning Sentiment Analysis")
    review = st.text_area("Enter a movie review for transfer learning:", height=150, key="transfer_input")
    if st.button("Analyze Transfer Sentiment"):
        if review:
            transfer_tokenizer = BertTokenizer.from_pretrained("/Users/georginanev/Documents/movie-sentiment/models/emotion_model")
            transfer_model = BertForSequenceClassification.from_pretrained("/Users/georginanev/Documents/movie-sentiment/models/emotion_model", num_labels=2)
            inputs = transfer_tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = transfer_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
            st.success(f"Sentiment (Transfer Learning): **{sentiment}**")
            
            # Confidence bar chart
            st.subheader("Transfer Learning Confidence")
            confidence_scores = probs[0].detach().numpy() * 100
            fig = px.bar(
                x=["Negative", "Positive"],
                y=confidence_scores,
                labels={"x": "Sentiment", "y": "Confidence (%)"},
                title="Transfer Learning Confidence Scores",
                color=["Negative", "Positive"],
                color_discrete_map={"Negative": "#FF6B6B", "Positive": "#4CAF50"}
            )
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig)
        else:
            st.error("Please enter a review!")