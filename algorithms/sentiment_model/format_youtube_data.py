
import pickle
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import torch

def format_youtube_data(
    raw_path="data/youtube_data.raw",
    output_features_path="data/youtube_features.npy",
    output_vectorizer_path=None,
    vectorizer_path=None,
    max_features=300,
    use_existing_vectorizer=True
):
    """
    Reads raw YouTube data, cleans and vectorizes it, and optionally saves features/vectorizer.
    If use_existing_vectorizer is True and output_vectorizer_path exists, loads and uses it for transform only (guaranteeing consistent feature size).
    Otherwise, fits a new vectorizer and saves it if output_vectorizer_path is given.
    Returns (features_tensor, vectorizer, samples).
    """
    def clean_text(text):
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    with open(raw_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    samples = [clean_text(s) for s in re.split(r'\n{2,}|(?<=\.)\s{2,}', raw) if s.strip()]

    vectorizer = None
    loaded_existing = False
    if use_existing_vectorizer and vectorizer_path and os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        loaded_existing = True
        features = vectorizer.transform(samples).toarray()
    else:
        # Fit a new vectorizer and optionally save it
        vectorizer = TfidfVectorizer(max_features=max_features)
        features = vectorizer.fit_transform(samples).toarray()
        if output_vectorizer_path:
            with open(output_vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"Saved vectorizer to {output_vectorizer_path}")

    features_tensor = torch.tensor(features, dtype=torch.float32)

    if output_features_path:
        np.save(output_features_path, features)

    print(f"Processed {len(samples)} samples. Features shape: {features_tensor.shape}")
    if output_features_path:
        print(f"Saved features to {output_features_path}")
    if vectorizer_path and loaded_existing:
        print(f"Loaded vectorizer from {vectorizer_path}")

    return features_tensor, vectorizer, samples

if __name__ == "__main__":
    # Example: Format YouTube data using Sentiment140 vectorizer for inference
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, '../../data'))
    format_youtube_data(
        raw_path=os.path.join(data_dir, 'youtube_data.raw'),
        output_features_path=os.path.join(data_dir, 'youtube_features.npy'),
        vectorizer_path=os.path.join(data_dir, 'twitter_vectorizer.pkl'),
        max_features=300,
        use_existing_vectorizer=True
    )