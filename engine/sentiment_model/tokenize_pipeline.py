#!/usr/bin/env python3
"""
Tokenization Pipeline for YouTube Sentiment Analysis Data

This script processes raw YouTube transcript data scraped for NVIDIA stock sentiment analysis.
It converts unstructured text into tokenized sequences ready for PyTorch model training.

Author: ML Trading Bot Project
Purpose: Sentiment analysis data preparation for financial trading decisions
"""

import os
import re
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, logging as transformers_logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Suppress transformer warnings for cleaner output
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging for pipeline monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TokenizationConfig:
    """
    Configuration class for tokenization pipeline parameters.
    
    This dataclass centralizes all configuration options for easy modification
    and ensures type safety across the pipeline.
    """
    # File paths
    raw_data_path: str = "youtube.raw"
    config_path: str = "../../../config.json"
    output_dir: str = "processed_data/"
    
    # Tokenization parameters
    tokenizer_model: str = "bert-base-uncased"  # Pre-trained tokenizer for financial text
    max_sequence_length: int = 128  # Maximum tokens per sequence (BERT limit is 512)
    overlap_size: int = 32  # Overlap between sliding windows for context preservation
    
    # Data processing parameters
    train_split: float = 0.8  # 80% for training
    validation_split: float = 0.1  # 10% for validation (remaining 10% for test)
    min_text_length: int = 50  # Minimum characters per text segment
    
    # PyTorch parameters
    batch_size: int = 16  # Batch size for data loading
    shuffle_data: bool = True  # Shuffle training data
    random_seed: int = 42  # For reproducible results
    
    def __post_init__(self):
        """Load configuration values from config.json file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Extract sentiment model tokenization config
            tokenization_config = config_data.get('sentiment_model', {}).get('tokenization', {})
            
            # Update values from config file if they exist
            if 'tokenizer_model' in tokenization_config:
                self.tokenizer_model = tokenization_config['tokenizer_model']
            if 'train_split' in tokenization_config:
                self.train_split = tokenization_config['train_split']
            if 'validation_split' in tokenization_config:
                self.validation_split = tokenization_config['validation_split']
            if 'batch_size' in tokenization_config:
                self.batch_size = tokenization_config['batch_size']
            if 'shuffle_data' in tokenization_config:
                self.shuffle_data = tokenization_config['shuffle_data']
            if 'random_seed' in tokenization_config:
                self.random_seed = tokenization_config['random_seed']
                
            logger.info("Configuration loaded from config.json")
            
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using default values")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file {self.config_path}, using default values")
        except Exception as e:
            logger.error(f"Error loading config file {self.config_path}: {e}, using default values")


class TextPreprocessor:
    """
    Handles preprocessing of raw YouTube transcript data for sentiment analysis.
    
    This class cleans and normalizes text data, removing noise and standardizing
    format for optimal tokenization performance.
    """
    
    def __init__(self):
        """Initialize the preprocessor with cleaning patterns."""
        # Compile regex patterns for efficient text cleaning
        self.cleaning_patterns = {
            # Remove extra whitespace and normalize spaces
            'whitespace': re.compile(r'\s+'),
            
            # Remove line numbers and arrows from scraped data (e.g., "1→", "2→")
            'line_numbers': re.compile(r'\d+→\s*'),
            
            # Clean up percentage signs and financial symbols
            'percentages': re.compile(r'(\d+)\s*%'),
            
            # Normalize stock tickers (e.g., "NVIDIA", "Nvidia", "nvidia" -> "NVIDIA")
            'nvidia_ticker': re.compile(r'\bnvidia\b', re.IGNORECASE),
            
            # Remove excessive punctuation but preserve sentence structure
            'punct_cleanup': re.compile(r'([.!?]){2,}'),
            
            # Clean up common transcript artifacts
            'transcript_noise': re.compile(r'\b(um|uh|like|you know|right\?)\b', re.IGNORECASE),
        }
        
        logger.info("TextPreprocessor initialized with cleaning patterns")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text for tokenization.
        
        Args:
            text (str): Raw text from YouTube transcripts
            
        Returns:
            str: Cleaned and normalized text
        """
        if not isinstance(text, str):
            logger.warning(f"Invalid text type: {type(text)}, converting to string")
            text = str(text)
        
        # Remove line numbers and formatting artifacts
        text = self.cleaning_patterns['line_numbers'].sub('', text)
        
        # Normalize whitespace
        text = self.cleaning_patterns['whitespace'].sub(' ', text)
        
        # Standardize percentage formatting
        text = self.cleaning_patterns['percentages'].sub(r'\1%', text)
        
        # Normalize NVIDIA ticker mentions
        text = self.cleaning_patterns['nvidia_ticker'].sub('NVIDIA', text)
        
        # Clean up excessive punctuation
        text = self.cleaning_patterns['punct_cleanup'].sub(r'\1', text)
        
        # Remove common transcript filler words (optional - might contain sentiment)
        # text = self.cleaning_patterns['transcript_noise'].sub('', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def segment_text(self, text: str, min_length: int = 50) -> List[str]:
        """
        Split long text into manageable segments for tokenization.
        
        This method attempts to split text at natural boundaries (sentences)
        while maintaining minimum length requirements for meaningful analysis.
        
        Args:
            text (str): Input text to segment
            min_length (int): Minimum characters per segment
            
        Returns:
            List[str]: List of text segments
        """
        if len(text) < min_length:
            logger.debug(f"Text too short ({len(text)} chars), skipping")
            return []
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add sentence to current segment
            potential_segment = f"{current_segment} {sentence}".strip()
            
            # If segment is getting too long, finalize current and start new
            if len(potential_segment) > min_length * 3:  # 3x min_length as soft max
                if len(current_segment) >= min_length:
                    segments.append(current_segment)
                current_segment = sentence
            else:
                current_segment = potential_segment
        
        # Add final segment if it meets minimum length
        if len(current_segment) >= min_length:
            segments.append(current_segment)
        
        logger.debug(f"Segmented text into {len(segments)} segments")
        return segments


class SentimentDataset(Dataset):
    """
    PyTorch Dataset class for sentiment analysis data.
    
    This dataset handles tokenized sequences and provides the interface
    needed for PyTorch DataLoader integration.
    """
    
    def __init__(self, tokenized_texts: List[Dict], labels: Optional[List[int]] = None):
        """
        Initialize dataset with tokenized texts.
        
        Args:
            tokenized_texts (List[Dict]): List of tokenized text dictionaries
            labels (Optional[List[int]]): Sentiment labels (if available)
        """
        self.tokenized_texts = tokenized_texts
        self.labels = labels
        
        logger.info(f"SentimentDataset initialized with {len(self.tokenized_texts)} samples")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing tokenized data
        """
        item = {
            'input_ids': torch.tensor(self.tokenized_texts[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.tokenized_texts[idx]['attention_mask'], dtype=torch.long)
        }
        
        # Add labels if available (for supervised learning)
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class TokenizationPipeline:
    """
    Main tokenization pipeline for processing YouTube sentiment data.
    
    This class orchestrates the entire tokenization process from raw text
    to PyTorch-ready data loaders.
    """
    
    def __init__(self, config: TokenizationConfig):
        """
        Initialize the tokenization pipeline.
        
        Args:
            config (TokenizationConfig): Configuration object with pipeline parameters
        """
        self.config = config
        self.preprocessor = TextPreprocessor()
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {config.tokenizer_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_model,
            do_lower_case=True,  # Normalize case for consistency
            padding_side='right'  # Pad on right side for BERT-style models
        )
        
        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("TokenizationPipeline initialized successfully")
    
    def load_raw_data(self) -> str:
        """
        Load raw YouTube transcript data from file.
        
        Returns:
            str: Raw text data
            
        Raises:
            FileNotFoundError: If raw data file doesn't exist
            IOError: If file cannot be read
        """
        try:
            logger.info(f"Loading raw data from: {self.config.raw_data_path}")
            
            with open(self.config.raw_data_path, 'r', encoding='utf-8') as file:
                raw_data = file.read()
            
            logger.info(f"Successfully loaded {len(raw_data)} characters of raw data")
            return raw_data
            
        except FileNotFoundError:
            logger.error(f"Raw data file not found: {self.config.raw_data_path}")
            raise
        except IOError as e:
            logger.error(f"Error reading raw data file: {e}")
            raise
    
    def preprocess_data(self, raw_data: str) -> List[str]:
        """
        Preprocess raw text data into clean segments.
        
        Args:
            raw_data (str): Raw text from YouTube transcripts
            
        Returns:
            List[str]: List of cleaned text segments
        """
        logger.info("Starting data preprocessing")
        
        # Clean the entire text first
        cleaned_text = self.preprocessor.clean_text(raw_data)
        
        # Segment into manageable pieces
        text_segments = self.preprocessor.segment_text(
            cleaned_text, 
            min_length=self.config.min_text_length
        )
        
        logger.info(f"Preprocessing complete: {len(text_segments)} segments created")
        return text_segments
    
    def tokenize_segments(self, text_segments: List[str]) -> List[Dict]:
        """
        Tokenize text segments using the loaded tokenizer.
        
        Args:
            text_segments (List[str]): List of preprocessed text segments
            
        Returns:
            List[Dict]: List of tokenized sequences with input_ids and attention_masks
        """
        logger.info(f"Tokenizing {len(text_segments)} text segments")
        
        tokenized_data = []
        
        for i, segment in enumerate(text_segments):
            # Tokenize with padding and truncation
            encoded = self.tokenizer(
                segment,
                add_special_tokens=True,  # Add [CLS] and [SEP] tokens
                max_length=self.config.max_sequence_length,
                padding='max_length',  # Pad to max_length
                truncation=True,  # Truncate if longer than max_length
                return_attention_mask=True,  # Return attention masks
                return_tensors='pt'  # Return as PyTorch tensors
            )
            
            # Convert to lists for storage (tensors can't be easily serialized)
            tokenized_data.append({
                'input_ids': encoded['input_ids'].squeeze().tolist(),
                'attention_mask': encoded['attention_mask'].squeeze().tolist(),
                'original_text': segment  # Keep reference to original for debugging
            })
            
            # Log progress periodically
            if (i + 1) % 100 == 0:
                logger.debug(f"Tokenized {i + 1}/{len(text_segments)} segments")
        
        logger.info(f"Tokenization complete: {len(tokenized_data)} sequences created")
        return tokenized_data
    
    def create_sliding_windows(self, tokenized_data: List[Dict]) -> List[Dict]:
        """
        Create overlapping sliding windows for better context preservation.
        
        This method is useful for very long texts where important context
        might be lost due to sequence length limits.
        
        Args:
            tokenized_data (List[Dict]): Original tokenized sequences
            
        Returns:
            List[Dict]: Extended list with sliding window sequences
        """
        if self.config.overlap_size <= 0:
            logger.info("No sliding window overlap specified, returning original data")
            return tokenized_data
        
        logger.info("Creating sliding windows with overlap")
        windowed_data = []
        
        for token_dict in tokenized_data:
            input_ids = token_dict['input_ids']
            attention_mask = token_dict['attention_mask']
            
            # Find actual sequence length (excluding padding)
            actual_length = sum(attention_mask)
            
            if actual_length <= self.config.max_sequence_length:
                # Sequence fits in one window, keep as-is
                windowed_data.append(token_dict)
            else:
                # Create overlapping windows
                step_size = self.config.max_sequence_length - self.config.overlap_size
                
                for start_idx in range(0, actual_length - self.config.overlap_size, step_size):
                    end_idx = min(start_idx + self.config.max_sequence_length, actual_length)
                    
                    # Create windowed sequence
                    windowed_input = input_ids[start_idx:end_idx]
                    windowed_attention = attention_mask[start_idx:end_idx]
                    
                    # Pad if necessary
                    if len(windowed_input) < self.config.max_sequence_length:
                        pad_length = self.config.max_sequence_length - len(windowed_input)
                        windowed_input.extend([self.tokenizer.pad_token_id] * pad_length)
                        windowed_attention.extend([0] * pad_length)
                    
                    windowed_data.append({
                        'input_ids': windowed_input,
                        'attention_mask': windowed_attention,
                        'original_text': token_dict['original_text']  # Reference to source
                    })
        
        logger.info(f"Sliding windows created: {len(windowed_data)} total sequences")
        return windowed_data
    
    def split_data(self, tokenized_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split tokenized data into training, validation, and test sets.
        
        Args:
            tokenized_data (List[Dict]): Complete tokenized dataset
            
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Train, validation, and test splits
        """
        logger.info("Splitting data into train/validation/test sets")
        
        # Set random seed for reproducible splits
        np.random.seed(self.config.random_seed)
        
        # Calculate split sizes
        total_size = len(tokenized_data)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.validation_split)
        # test_size is the remainder
        
        # Shuffle data before splitting
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create splits
        train_data = [tokenized_data[i] for i in train_indices]
        val_data = [tokenized_data[i] for i in val_indices]
        test_data = [tokenized_data[i] for i in test_indices]
        
        logger.info(f"Data split complete: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def create_datasets_and_loaders(self, train_data: List[Dict], 
                                  val_data: List[Dict], 
                                  test_data: List[Dict]) -> Dict[str, DataLoader]:
        """
        Create PyTorch datasets and data loaders.
        
        Args:
            train_data (List[Dict]): Training data
            val_data (List[Dict]): Validation data
            test_data (List[Dict]): Test data
            
        Returns:
            Dict[str, DataLoader]: Dictionary containing train/val/test data loaders
        """
        logger.info("Creating PyTorch datasets and data loaders")
        
        # Create datasets
        train_dataset = SentimentDataset(train_data)
        val_dataset = SentimentDataset(val_data)
        test_dataset = SentimentDataset(test_data)
        
        # Create data loaders
        data_loaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_data,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=True if torch.cuda.is_available() else False
            ),
            'validation': DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,  # Don't shuffle validation data
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,  # Don't shuffle test data
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
        }
        
        logger.info("Data loaders created successfully")
        return data_loaders
    
    def save_processed_data(self, train_data: List[Dict], 
                          val_data: List[Dict], 
                          test_data: List[Dict]) -> None:
        """
        Save processed data to disk for future use.
        
        Args:
            train_data (List[Dict]): Training data to save
            val_data (List[Dict]): Validation data to save
            test_data (List[Dict]): Test data to save
        """
        logger.info("Saving processed data to disk")
        
        # Save each split separately
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, data in splits.items():
            output_path = Path(self.config.output_dir) / f"{split_name}_tokenized.json"
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {split_name} data: {len(data)} samples to {output_path}")
                
            except IOError as e:
                logger.error(f"Error saving {split_name} data: {e}")
                raise
        
        # Save tokenizer configuration for model compatibility
        tokenizer_config_path = Path(self.config.output_dir) / "tokenizer_config.json"
        tokenizer_info = {
            'model_name': self.config.tokenizer_model,
            'max_sequence_length': self.config.max_sequence_length,
            'vocab_size': self.tokenizer.vocab_size,
            'pad_token_id': self.tokenizer.pad_token_id,
            'special_tokens': self.tokenizer.special_tokens_map
        }
        
        with open(tokenizer_config_path, 'w') as f:
            json.dump(tokenizer_info, f, indent=2)
        
        logger.info(f"Saved tokenizer configuration to {tokenizer_config_path}")
    
    def generate_data_statistics(self, train_data: List[Dict], 
                               val_data: List[Dict], 
                               test_data: List[Dict]) -> Dict:
        """
        Generate and save statistics about the processed data.
        
        Args:
            train_data (List[Dict]): Training data
            val_data (List[Dict]): Validation data  
            test_data (List[Dict]): Test data
            
        Returns:
            Dict: Dictionary containing data statistics
        """
        logger.info("Generating data statistics")
        
        def analyze_split(data: List[Dict]) -> Dict:
            """Analyze a single data split."""
            if not data:
                return {'count': 0}
            
            # Calculate sequence length statistics
            seq_lengths = [sum(item['attention_mask']) for item in data]
            
            # Token frequency analysis (first 1000 samples to avoid memory issues)
            sample_data = data[:1000] if len(data) > 1000 else data
            all_tokens = []
            for item in sample_data:
                # Only count non-padded tokens
                valid_tokens = [token_id for token_id, mask in 
                              zip(item['input_ids'], item['attention_mask']) if mask == 1]
                all_tokens.extend(valid_tokens)
            
            token_counter = Counter(all_tokens)
            
            return {
                'count': len(data),
                'sequence_length': {
                    'min': min(seq_lengths),
                    'max': max(seq_lengths),
                    'mean': np.mean(seq_lengths),
                    'std': np.std(seq_lengths)
                },
                'vocabulary': {
                    'unique_tokens': len(token_counter),
                    'total_tokens': len(all_tokens),
                    'most_common_tokens': token_counter.most_common(10)
                }
            }
        
        # Analyze each split
        statistics = {
            'train': analyze_split(train_data),
            'validation': analyze_split(val_data),
            'test': analyze_split(test_data),
            'total_samples': len(train_data) + len(val_data) + len(test_data),
            'tokenizer_info': {
                'model': self.config.tokenizer_model,
                'vocab_size': self.tokenizer.vocab_size,
                'max_length': self.config.max_sequence_length
            }
        }
        
        # Save statistics
        stats_path = Path(self.config.output_dir) / "data_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2, default=str)  # default=str handles numpy types
        
        logger.info(f"Data statistics saved to {stats_path}")
        return statistics
    
    def run_pipeline(self) -> Dict[str, DataLoader]:
        """
        Execute the complete tokenization pipeline.
        
        Returns:
            Dict[str, DataLoader]: Dictionary containing train/val/test data loaders
        """
        logger.info("=== Starting Tokenization Pipeline ===")
        
        try:
            # Step 1: Load raw data
            raw_data = self.load_raw_data()
            
            # Step 2: Preprocess data
            text_segments = self.preprocess_data(raw_data)
            
            # Step 3: Tokenize segments
            tokenized_data = self.tokenize_segments(text_segments)
            
            # Step 4: Create sliding windows (if configured)
            if self.config.overlap_size > 0:
                tokenized_data = self.create_sliding_windows(tokenized_data)
            
            # Step 5: Split data
            train_data, val_data, test_data = self.split_data(tokenized_data)
            
            # Step 6: Create datasets and loaders
            data_loaders = self.create_datasets_and_loaders(train_data, val_data, test_data)
            
            # Step 7: Save processed data
            self.save_processed_data(train_data, val_data, test_data)
            
            # Step 8: Generate statistics
            statistics = self.generate_data_statistics(train_data, val_data, test_data)
            
            logger.info("=== Tokenization Pipeline Complete ===")
            logger.info(f"Total samples: {statistics['total_samples']}")
            logger.info(f"Train: {statistics['train']['count']}, Val: {statistics['validation']['count']}, Test: {statistics['test']['count']}")
            
            return data_loaders
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """
    Main function to run the tokenization pipeline.
    
    This function serves as the entry point when the script is run directly.
    It initializes the pipeline with default configuration and processes the data.
    """
    # Initialize configuration
    config = TokenizationConfig()
    
    # Log configuration details
    logger.info("Tokenization Pipeline Configuration:")
    logger.info(f"  Raw data path: {config.raw_data_path}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Tokenizer model: {config.tokenizer_model}")
    logger.info(f"  Max sequence length: {config.max_sequence_length}")
    logger.info(f"  Batch size: {config.batch_size}")
    
    # Initialize and run pipeline
    pipeline = TokenizationPipeline(config)
    data_loaders = pipeline.run_pipeline()
    
    # Demonstrate data loader usage
    logger.info("=== Data Loader Demo ===")
    train_loader = data_loaders['train']
    
    # Get first batch for demonstration
    first_batch = next(iter(train_loader))
    logger.info(f"First batch shape - Input IDs: {first_batch['input_ids'].shape}")
    logger.info(f"First batch shape - Attention Mask: {first_batch['attention_mask'].shape}")
    
    # Decode first sequence for verification
    first_sequence = first_batch['input_ids'][0]
    decoded_text = pipeline.tokenizer.decode(first_sequence, skip_special_tokens=True)
    logger.info(f"First sequence (decoded): {decoded_text[:100]}...")
    
    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()