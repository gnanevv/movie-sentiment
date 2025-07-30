Sentiment Analysis with BERT
This project uses a pre-trained BERT model to classify movie reviews as positive or negative. It includes data preprocessing, model training, evaluation, and an interactive web app built with Streamlit.
Features

Fine-tunes BERT for sentiment analysis on the IMDB dataset.
Achieves high accuracy with minimal training.
Interactive web app for real-time predictions.

Setup

Clone the repository:git clone https://github.com/gnanevv/movie-sentiment

Create a virtual environment and install dependencies:python -m venv sentiment_env
source sentiment_env/bin/activate # On Windows: sentiment_env\Scripts\activate
pip install -r requirements.txt

Run the scripts in order:
python data_preparation.py
python train_model.py
python evaluate_model.py
streamlit run app.py

Requirements
See requirements.txt for dependencies.
Dataset
Uses a subset of the IMDB Movie Reviews dataset.
Results

Test accuracy: ~85% (varies based on training).
Interactive app for user input.

License
MIT License
