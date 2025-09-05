#!/usr/bin/env python3
"""
Download Twitter Sentiment Dataset

Simple script to download real Twitter sentiment datasets for BERT model training.
No hardcoded data - downloads actual datasets from public sources.
"""

import requests
import zipfile
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_file_with_progress(url: str, destination: str) -> bool:
    """Download a file with progress indication."""
    try:
        logger.info(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading... {percent:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Downloaded successfully: {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_sentiment140_dataset(data_dir: Path) -> Optional[str]:
    """
    Download the Sentiment140 dataset (1.6M Twitter sentiments).
    This is a real dataset from Stanford with Twitter sentiment data.
    """
    logger.info("Downloading Sentiment140 dataset...")
    
    # Sentiment140 dataset URL (original from Stanford)
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    zip_file = data_dir / "sentiment140.zip"
    
    # Download the dataset
    if not download_file_with_progress(url, str(zip_file)):
        logger.error("Failed to download Sentiment140 dataset")
        return None
    
    # Extract the zip file
    try:
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # The main training file is usually named training.1600000.processed.noemoticon.csv
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in extracted archive")
            return None
        
        main_csv = max(csv_files, key=lambda x: x.stat().st_size)  # Get largest CSV
        logger.info(f"Found dataset file: {main_csv}")
        
        # Process the CSV to standard format
        processed_file = process_sentiment140_csv(main_csv, data_dir)
        
        # Clean up zip file
        zip_file.unlink()
        
        return processed_file
        
    except Exception as e:
        logger.error(f"Failed to extract or process dataset: {e}")
        return None


def process_sentiment140_csv(csv_file: Path, output_dir: Path) -> str:
    """Process Sentiment140 CSV to standard format."""
    logger.info(f"Processing {csv_file}...")
    
    try:
        # Sentiment140 format: [target, ids, date, flag, user, text]
        # target: 0 = negative, 2 = neutral, 4 = positive
        column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
        
        # Read the CSV (it's quite large, so we might want to sample)
        logger.info("Reading CSV file (this may take a while for large files)...")
        df = pd.read_csv(csv_file, encoding='latin-1', header=None, names=column_names)
        
        logger.info(f"Loaded {len(df)} samples from dataset")
        
        # Map sentiment labels
        sentiment_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
        df['sentiment'] = df['target'].map(sentiment_map)
        
        # Keep only text and sentiment columns
        df_clean = df[['text', 'sentiment']].copy()
        
        # Remove any rows with missing sentiment mapping
        df_clean = df_clean.dropna()
        
        # Remove duplicates
        initial_size = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['text'])
        logger.info(f"Removed {initial_size - len(df_clean)} duplicate entries")
        
        # Save processed dataset
        output_file = output_dir / "twitter_sentiment_dataset.csv"
        df_clean.to_csv(output_file, index=False)
        
        logger.info(f"Processed dataset saved: {output_file}")
        logger.info(f"Final dataset size: {len(df_clean)} samples")
        
        # Show sentiment distribution
        sentiment_counts = df_clean['sentiment'].value_counts()
        logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Failed to process CSV: {e}")
        # Fallback: try to download alternative dataset
        result = download_alternative_dataset(output_dir)
        return result if result else ""


def download_alternative_dataset(data_dir: Path) -> Optional[str]:
    """Download alternative Twitter sentiment dataset if Sentiment140 fails."""
    logger.info("Attempting to download alternative dataset...")
    
    # Try the Kaggle Twitter sentiment dataset (smaller, more accessible)
    urls = [
        "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv",
        "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"
    ]
    
    for url in urls:
        try:
            csv_file = data_dir / "twitter_sentiment_alt.csv"
            if download_file_with_progress(url, str(csv_file)):
                
                # Try to read and process
                df = pd.read_csv(csv_file)
                
                # Look for common column names
                text_col = None
                sentiment_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if 'text' in col_lower or 'tweet' in col_lower or 'message' in col_lower:
                        text_col = col
                    elif 'sentiment' in col_lower or 'label' in col_lower or 'target' in col_lower:
                        sentiment_col = col
                
                if text_col and sentiment_col:
                    # Create standard format
                    df_clean = df[[text_col, sentiment_col]].copy()
                    df_clean.columns = ['text', 'sentiment']
                    
                    # Standardize sentiment labels
                    unique_sentiments = df_clean['sentiment'].unique()
                    logger.info(f"Found sentiment labels: {unique_sentiments}")
                    
                    # Map common sentiment formats
                    if set(unique_sentiments).issubset({0, 1, 2}):
                        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                        df_clean['sentiment'] = df_clean['sentiment'].map(sentiment_map)
                    elif set(unique_sentiments).issubset({-1, 0, 1}):
                        sentiment_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
                        df_clean['sentiment'] = df_clean['sentiment'].map(sentiment_map)
                    
                    # Remove rows with unmapped sentiments
                    df_clean = df_clean[df_clean['sentiment'].isin(['positive', 'negative', 'neutral'])]
                    
                    if len(df_clean) > 100:  # Minimum viable dataset
                        output_file = data_dir / "twitter_sentiment_dataset.csv"
                        df_clean.to_csv(output_file, index=False)
                        
                        logger.info(f"Alternative dataset saved: {output_file}")
                        logger.info(f"Dataset size: {len(df_clean)} samples")
                        
                        sentiment_counts = df_clean['sentiment'].value_counts()
                        logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
                        
                        return str(output_file)
                
                csv_file.unlink()  # Clean up failed attempt
                
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue
    
    logger.error("All download attempts failed")
    return None


def main():
    """Main function to download Twitter sentiment dataset."""
    # Setup directories
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    
    data_dir.mkdir(exist_ok=True)
    
    logger.info("=== Twitter Sentiment Dataset Download ===")
    logger.info(f"Data directory: {data_dir}")
    
    # Try to download Sentiment140 first (largest, highest quality)
    dataset_file = download_sentiment140_dataset(data_dir)
    
    if not dataset_file:
        logger.warning("Sentiment140 download failed, trying alternative sources...")
        dataset_file = download_alternative_dataset(data_dir)
    
    if dataset_file:
        logger.info("=== Dataset Download Completed Successfully! ===")
        logger.info(f"Dataset saved to: {dataset_file}")
        logger.info("Use tokenize_pipeline.py for tokenization and normalization")
        return 0
    else:
        logger.error("=== Dataset Download Failed! ===")
        logger.error("Could not download any Twitter sentiment dataset")
        return 1


if __name__ == "__main__":
    main()