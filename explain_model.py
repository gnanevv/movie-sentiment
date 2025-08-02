import shap
from transformers import pipeline
import streamlit as st
import matplotlib.pyplot as plt

# Load your BERT model as a pipeline
model_path = "models/sentiment_model"  # Update with your actual model path
sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# Create SHAP explainer
explainer = shap.Explainer(sentiment_pipeline)

# Streamlit app for explainability
st.title("BERT Sentiment Explainability")
st.markdown("See which words in your movie review drive the sentiment prediction.")

review = st.text_area("Enter a movie review:", value="The acting was great but the plot was boring.")
if st.button("Explain Sentiment"):
    if review:
        # Get SHAP values
        shap_values = explainer([review])
        
        # Display sentiment prediction
        prediction = sentiment_pipeline(review)[0]
        st.success(f"Sentiment: **{prediction['label']}** (Score: {prediction['score']:.2f})")
        
        # Visualize SHAP explanation
        st.subheader("Word Importance")
        plt.figure(figsize=(10, 5))
        shap.plots.text(shap_values, display=False)
        st.pyplot(plt)
    else:
        st.error("Please enter a review!")