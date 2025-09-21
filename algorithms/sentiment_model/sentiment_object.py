
from algorithms.sentiment_model.logistic_model import LogisticSentimentModel
from algorithms.sentiment_model.format_youtube_data import format_youtube_data
from algorithms.sentiment_model.train_sentiment_model import train_sentiment_model
import torch
import os


class SentimentModel:
    def __init__(self):
        self.model = LogisticSentimentModel(input_dim=300)  # Example input dimension


    def format_data(self):
        self.features_tensor, self.vectorizer, _ = format_youtube_data(
            raw_path="data/youtube_data.raw",
            output_features_path="data/youtube_features.npy",
            output_vectorizer_path="data/youtube_vectorizer.pkl",
            max_features=300,
            use_existing_vectorizer=False
        )
        return None

    def train_model(self, label_path=None, epochs=20, lr=0.01):
        """
        Train the sentiment model using train_sentiment_model utility.
        Updates self.model with trained weights.
        """
        base_dir = os.path.dirname(__file__)
        model = train_sentiment_model(
            raw_path='data/youtube_data.raw',
            vectorizer_path='data/youtube_vectorizer.pkl',
            features_path='data/youtube_features.npy',
            model_save_path='data/sentiment_logistic_model.pt',
            label_path=label_path or 'data/youtube_labels.npy',
            max_features=300,
            epochs=epochs,
            lr=lr
        )
        self.model.load_state_dict(model.state_dict())
        self.model.eval()


    def load_model(self, model_path: str):
        # Load the sentiment analysis model from the specified path
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_sentiment(self, text: str) -> float:
        """
        Predict the sentiment score of the given text using the trained model and fitted vectorizer.
        Returns a float in [0, 1] (probability of positive sentiment).
        """
        if not hasattr(self, 'vectorizer') or self.vectorizer is None:
            raise ValueError("Vectorizer not loaded. Run format_data() first or load vectorizer.")
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not loaded. Run train_model() or load_model() first.")

        # Clean and vectorize input text
        import re
        def clean_text(t):
            t = re.sub(r'&[a-z]+;', ' ', t)
            t = re.sub(r'\s+', ' ', t)
            return t.strip()
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned]).toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor)
            prob = torch.sigmoid(logits).item()
        return prob