#!/usr/bin/env python3
"""
Train BERT Sentiment Model with Properly Labeled Financial Data

This script trains the BERT sentiment model using properly labeled financial sentiment data
instead of pseudo-labels from keyword matching.

Usage:
    python train_with_labeled_data.py [--epochs 5] [--batch_size 16] [--learning_rate 2e-5]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Import the existing model classes
from model import BertSentimentModel, ModelConfig
from tokenize_pipeline import TokenizationPipeline, TokenizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabeledSentimentDataset(Dataset):
    """Dataset class for properly labeled sentiment analysis training."""
    
    def __init__(self, tokenized_data_file: str):
        """
        Initialize dataset with tokenized data file containing proper labels.
        
        Args:
            tokenized_data_file: Path to JSON file with tokenized data and labels
        """
        with open(tokenized_data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {tokenized_data_file}")
        
        # Log label distribution
        label_counts = {}
        for item in self.data:
            label = item.get('labels', item.get('sentiment', 'unknown'))
            if isinstance(label, str):
                # Convert string labels to numbers
                label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
                label = label_map.get(label, 1)  # default to neutral
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"Label distribution: {label_counts}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Handle both numeric and string labels
        label = sample.get('labels', sample.get('sentiment', 1))
        if isinstance(label, str):
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            label = label_map.get(label, 1)
        
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ImprovedSentimentTrainer:
    """Improved trainer with proper data handling and metrics."""
    
    def __init__(self, model: BertSentimentModel, config: ModelConfig):
        """
        Initialize the trainer.
        
        Args:
            model (BertSentimentModel): The model to train
            config (ModelConfig): Configuration object
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with proper label handling."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Calculate accuracy
            predictions, _ = self.model.get_predictions(logits)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch with proper label handling."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions, _ = self.model.get_predictions(logits)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
                
                # Update progress bar
                current_acc = correct_predictions / total_samples
                progress_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_acc': f'{current_acc:.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Train the model for the specified number of epochs."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            
            logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            
            logger.info(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.save_model("best_model_labeled.pt")
                self.save_weights_only("model_weights_labeled.pt")
                logger.info(f"New best validation accuracy: {val_acc:.4f}")
        
        logger.info("Training completed!")
        return self.training_history
    
    def save_model(self, filename: str) -> None:
        """Save the model state dict and configuration."""
        save_path = Path(self.config.model_save_path) / filename
        
        # Save model state dict and config
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def save_weights_only(self, filename: str) -> None:
        """Save only the model weights for easier loading."""
        save_path = Path(self.config.model_save_path) / filename
        
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model weights saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train BERT Sentiment Model with Labeled Data')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--train_file', default='processed_data/train_tokenized_labeled.json', 
                       help='Training data file')
    parser.add_argument('--val_file', default='processed_data/val_tokenized_labeled.json', 
                       help='Validation data file')
    
    args = parser.parse_args()
    
    logger.info("=== Starting BERT Sentiment Model Training with Labeled Data ===")
    
    # Set up paths
    current_dir = Path(__file__).parent
    train_file = current_dir / args.train_file
    val_file = current_dir / args.val_file
    
    # Check if files exist
    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        logger.info("Please run: python download_financial_sentiment_data.py --dataset sample")
        return 1
    
    if not val_file.exists():
        logger.error(f"Validation data not found: {val_file}")
        logger.info("Please run: python download_financial_sentiment_data.py --dataset sample")
        return 1
    
    # Create datasets
    train_dataset = LabeledSentimentDataset(str(train_file))
    val_dataset = LabeledSentimentDataset(str(val_file))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model configuration
    config = ModelConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info(f"Training configuration:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    # Create model
    model = BertSentimentModel(config)
    logger.info(f"Created model with {model.num_parameters():,} parameters")
    
    # Create trainer
    trainer = ImprovedSentimentTrainer(model, config)
    
    # Train the model
    try:
        logger.info("Starting training...")
        training_history = trainer.train(train_loader, val_loader)
        
        # Save final model
        trainer.save_model("final_model_labeled.pt")
        trainer.save_weights_only("final_weights_labeled.pt")
        
        # Save training history
        history_file = Path(config.model_save_path) / "training_history_labeled.json"
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info("=== Training completed successfully! ===")
        
        # Print final results
        if training_history['val_accuracy']:
            final_val_acc = training_history['val_accuracy'][-1]
            best_val_acc = max(training_history['val_accuracy'])
            logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        logger.info(f"Best model saved to: {Path(config.model_save_path) / 'best_model_labeled.pt'}")
        logger.info(f"Weights saved to: {Path(config.model_save_path) / 'model_weights_labeled.pt'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)