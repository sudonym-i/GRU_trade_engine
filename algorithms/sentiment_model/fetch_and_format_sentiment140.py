# fetch_and_format_sentiment140.py
"""
Downloads the Sentiment140 dataset, preprocesses it, fits a TF-IDF vectorizer, and saves features, labels, and the vectorizer for use in training.
Ensures consistent feature dimension for all future sentiment data.
"""
import os
import requests
import zipfile
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
RAW_ZIP_PATH = os.path.join(DATA_DIR, 'sentiment140.zip')
CSV_PATH = os.path.join(DATA_DIR, 'training.1600000.processed.noemoticon.csv')
FEATURES_PATH = os.path.join(DATA_DIR, 'twitter_features.npy')
LABELS_PATH = os.path.join(DATA_DIR, 'twitter_labels.npy')
VECTORIZER_PATH = os.path.join(DATA_DIR, 'twitter_vectorizer.pkl')

MAX_FEATURES = 300  # Set to match your pipeline


def download_and_extract():
    if not os.path.exists(CSV_PATH):
        print("Downloading Sentiment140 dataset...")
        r = requests.get(DATA_URL, stream=True)
        with open(RAW_ZIP_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Extracting...")
        with zipfile.ZipFile(RAW_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Done.")
    else:
        print("Sentiment140 CSV already exists.")

def clean_text(text):
    import re
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_and_extract()
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, encoding='latin-1', header=None)
    # Sentiment140: 0=negative, 4=positive
    df = df[[0, 5]]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({0: 0, 4: 1})
    df = df[df['label'].isin([0, 1])]
    print(f"Loaded {len(df)} samples.")
    # Clean text
    df['text'] = df['text'].astype(str).map(clean_text)
    # Fit vectorizer
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
    features = vectorizer.fit_transform(df['text']).toarray()
    labels = df['label'].values.astype(np.float32)
    # Save
    np.save(FEATURES_PATH, features)
    np.save(LABELS_PATH, labels)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Saved features to {FEATURES_PATH}")
    print(f"Saved labels to {LABELS_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Feature shape: {features.shape}, Labels shape: {labels.shape}")

if __name__ == "__main__":
    main()
