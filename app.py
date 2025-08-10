import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import os
from torchvision import transforms
from PIL import Image

# Device setup for M1 Pro
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load binary sentiment model
binary_model_path = "/Users/georginanev/Documents/movie-sentiment/models/sentiment_model"
try:
    tokenizer = DistilBertTokenizer.from_pretrained(binary_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(binary_model_path)
    model = model.to(device)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
except (OSError, ValueError) as e:
    st.error(f"Error loading binary model: {e}. Ensure the model is DistilBERT-based and properly saved.")
    st.stop()

# Load multi-class model
multi_model_path = "/Users/georginanev/Documents/movie-sentiment/models/sentiment_model_multi"
try:
    multi_tokenizer = DistilBertTokenizer.from_pretrained(multi_model_path)
    multi_model = DistilBertForSequenceClassification.from_pretrained(multi_model_path)
    multi_model = multi_model.to(device)
except (OSError, ValueError) as e:
    st.error(f"Error loading multi-class model: {e}. Ensure the model is DistilBERT-based and properly saved.")
    st.stop()

# Load emotion model (Transfer Learning)
emotion_model_path = "/Users/georginanev/Documents/movie-sentiment/models/emotion_model"
try:
    emotion_tokenizer = DistilBertTokenizer.from_pretrained(emotion_model_path)
    emotion_model = DistilBertForSequenceClassification.from_pretrained(emotion_model_path, num_labels=2)
    emotion_model = emotion_model.to(device)
except (OSError, ValueError) as e:
    st.error(f"Error loading emotion model: {e}. Ensure the model is DistilBERT-based and properly saved.")
    st.stop()

# Load ABSA model
absa_model_path = "/Users/georginanev/Documents/movie-sentiment/models/absa_model"
try:
    absa_tokenizer = DistilBertTokenizer.from_pretrained(absa_model_path)
    absa_model = DistilBertForSequenceClassification.from_pretrained(absa_model_path)
    absa_model = absa_model.to(device)
except (OSError, ValueError) as e:
    st.error(f"Error loading ABSA model: {e}. Ensure the model is DistilBERT-based and properly saved.")
    st.stop()

# Load multimodal model (Placeholder)
multimodal_model_path = "/Users/georginanev/Documents/movie-sentiment/models/multimodal_model"
try:
    multimodal_tokenizer = DistilBertTokenizer.from_pretrained(multimodal_model_path)
    multimodal_model = DistilBertForSequenceClassification.from_pretrained(multimodal_model_path)
    multimodal_model = multimodal_model.to(device)
except (OSError, ValueError) as e:
    st.warning(f"Multimodal model not loaded: {e}. Using text-only placeholder.")

# Create SHAP explainer for binary model
try:
    explainer = shap.Explainer(sentiment_pipeline)
except Exception as e:
    st.error(f"Error creating SHAP explainer: {e}")
    st.stop()

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
    return sentiment, probs

# Streamlit app
st.set_page_config(page_title="Movie Sentiment Analyzer", layout="wide")
st.title("🎬 Movie Sentiment Analyzer")
st.markdown("Analyze movie review sentiments with DistilBERT and explore word importance.")

# Tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sentiment Analysis",
    "Explainability",
    "Multi-Class Sentiment",
    "Transfer Learning",
    "Aspect-Based Sentiment",
    "Multimodal Sentiment"
])

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
            confidence_scores = probs[0].detach().cpu().numpy() * 100
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
            inputs = multi_tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = multi_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
            predicted_label = labels[torch.argmax(probs).item()]
            st.success(f"Sentiment: **{predicted_label}**")
            
            # Confidence bar chart
            st.subheader("Multi-Class Confidence")
            confidence_scores = probs[0].detach().cpu().numpy() * 100
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

# Tab 4: Transfer Learning
with tab4:
    st.header("Transfer Learning Sentiment Analysis")
    review = st.text_area("Enter a movie review for transfer learning:", height=150, key="transfer_input")
    if st.button("Analyze Transfer Sentiment"):
        if review:
            inputs = emotion_tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = emotion_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
            st.success(f"Sentiment (Transfer Learning): **{sentiment}**")
            
            # Confidence bar chart
            st.subheader("Transfer Learning Confidence")
            confidence_scores = probs[0].detach().cpu().numpy() * 100
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

# Tab 5: Aspect-Based Sentiment
with tab5:
    st.header("Aspect-Based Sentiment Analysis")
    review = st.text_area("Enter a movie review:", height=150, key="absa_review")
    aspect = st.text_input("Enter an aspect (e.g., acting, plot):", key="absa_aspect")
    if st.button("Analyze Aspect Sentiment"):
        if review and aspect:
            inputs = absa_tokenizer(f"{review} [SEP] {aspect}", return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = absa_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = ["Negative", "Neutral", "Positive"]
            predicted_label = labels[torch.argmax(probs).item() % len(labels)]
            st.success(f"Aspect '{aspect}' Sentiment: **{predicted_label}**")
            
            # Confidence bar chart
            st.subheader("Aspect Confidence")
            confidence_scores = probs[0].detach().cpu().numpy()[:3] * 100
            fig = px.bar(
                x=labels,
                y=confidence_scores,
                labels={"x": "Sentiment", "y": "Confidence (%)"},
                title=f"Confidence Scores for Aspect '{aspect}'",
                color=labels,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig)
        else:
            st.error("Please enter a review and an aspect!")

# Tab 6: Multimodal Sentiment (Placeholder)
with tab6:
    st.header("Multimodal Sentiment Analysis")
    st.warning("Multimodal analysis is a placeholder due to lack of paired text-image dataset.")
    review = st.text_area("Enter a movie review:", height=150, key="multimodal_review")
    image = st.file_uploader("Upload a movie poster (optional):", type=["jpg", "png"], key="multimodal_image")
    if st.button("Analyze Multimodal Sentiment"):
        if review:
            inputs = multimodal_tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = multimodal_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
            st.success(f"Sentiment (Text-Only, Placeholder): **{sentiment}**")
            
            # Confidence bar chart
            st.subheader("Multimodal Confidence (Text-Only)")
            confidence_scores = probs[0].detach().cpu().numpy() * 100
            fig = px.bar(
                x=["Negative", "Positive"],
                y=confidence_scores,
                labels={"x": "Sentiment", "y": "Confidence (%)"},
                title="Text-Only Confidence Scores (Placeholder)",
                color=["Negative", "Positive"],
                color_discrete_map={"Negative": "#FF6B6B", "Positive": "#4CAF50"}
            )
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig)
            
            if image:
                st.image(image, caption="Uploaded Movie Poster", use_column_width=True)
                st.info("Image processing not implemented due to missing multimodal model.")
        else:
            st.error("Please enter a review!")