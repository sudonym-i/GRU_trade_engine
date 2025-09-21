
import pickle
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import torch

def format_youtube_data(
        raw_path="data/youtube_data.raw",
        output_features_path="data/youtube_features.pt",
        output_vectorizer_path="data/youtube_vectorizer.pkl",
        max_features=1000,
        use_existing_vectorizer=False
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
    if use_existing_vectorizer and output_vectorizer_path and os.path.exists(output_vectorizer_path):
        with open(output_vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        loaded_existing = True
        features = vectorizer.transform(samples).toarray()
    else:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        features = vectorizer.fit_transform(samples).toarray()
        if output_vectorizer_path:
            with open(output_vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)

    features_tensor = torch.tensor(features, dtype=torch.float32)

    if output_features_path:
        np.save(output_features_path, features)

    print(f"Processed {len(samples)} samples. Features shape: {features_tensor.shape}")
    if output_features_path:
        print(f"Saved features to {output_features_path}")
    if output_vectorizer_path:
        if loaded_existing:
            print(f"Loaded vectorizer from {output_vectorizer_path}")
        else:
            print(f"Saved vectorizer to {output_vectorizer_path}")

    return features_tensor, vectorizer, samples

if __name__ == "__main__":
    # for testing
    format_youtube_data(
        raw_path="data/youtube_data.raw",
        output_features_path="data/youtube_features.pt",
        output_vectorizer_path="data/youtube_vectorizer.pkl",
        max_features=1000
    )