#!/usr/bin/env python3
"""
BERT-based Sentiment Analysis Model for Neural Trade Engine

This module implements a BERT model for sentiment analysis of YouTube transcript data
to support financial trading decisions. The model is designed to work with the 
tokenization pipeline and provide robust sentiment predictions.

Author: ML Trading Bot Project
Purpose: Sentiment analysis for financial trading decisions
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    BertModel, 
    BertConfig, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging as transformers_logging
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Suppress transformer warnings for cleaner output
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration class for BERT sentiment model parameters.
    
    Centralizes model hyperparameters and training configuration.
    """
    # Model architecture
    bert_model_name: str = "bert-base-uncased"
    num_classes: int = 3  # negative, neutral, positive
    dropout_rate: float = 0.3
    hidden_dim: int = 768  # BERT base hidden dimension
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data parameters
    max_sequence_length: int = 128
    
    # Paths
    model_save_path: str = "saved_models/"
    tokenizer_config_path: str = "processed_data/tokenizer_config.json"
    config_path: str = "../../../config.json"
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Load configuration values from config.json if available."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Extract sentiment model config
            model_config = config_data.get('sentiment_model', {}).get('model', {})
            
            # Update values from config file if they exist
            for key, value in model_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info("Model configuration loaded from config.json")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load config file: {e}, using default values")
        
        # Ensure model save directory exists
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)


class BertSentimentModel(nn.Module):
    """
    BERT-based sentiment analysis model.
    
    This model uses a pre-trained BERT encoder with a classification head
    for sentiment prediction on financial text data.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the BERT sentiment model.
        
        Args:
            config (ModelConfig): Model configuration object
        """
        super(BertSentimentModel, self).__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        
        # Load pre-trained BERT model
        logger.info(f"Loading BERT model: {config.bert_model_name}")
        self.bert = BertModel.from_pretrained(
            config.bert_model_name,
            return_dict=True
        )
        
        # Freeze BERT embeddings for stability (optional)
        # self.bert.embeddings.requires_grad_(False)
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        logger.info(f"Model initialized with {self.num_parameters()} parameters")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Logits for each class [batch_size, num_classes]
        """
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation for classification
        pooled_output = bert_outputs.pooler_output  # [batch_size, hidden_dim]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        return logits
    
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_predictions(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits to predictions and probabilities.
        
        Args:
            logits (torch.Tensor): Model logits [batch_size, num_classes]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predictions and probabilities
        """
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        return predictions, probabilities


class SentimentTrainer:
    """
    Trainer class for the BERT sentiment model.
    
    Handles training, validation, and model persistence.
    """
    
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
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            Tuple[float, float]: Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # For demonstration, create mock labels (replace with actual labels)
            labels = torch.randint(0, self.config.num_classes, (input_ids.size(0),)).to(self.device)
            
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
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Tuple[float, float]: Average loss and accuracy for validation
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # For demonstration, create mock labels (replace with actual labels)
                labels = torch.randint(0, self.config.num_classes, (input_ids.size(0),)).to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions, _ = self.model.get_predictions(logits)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (Optional[DataLoader]): Validation data loader
            
        Returns:
            Dict: Training history with losses and accuracies
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Initialize learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            
            logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
                
                logger.info(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    self.save_model("best_model.pt")
                    logger.info(f"New best validation accuracy: {val_acc:.4f}")
            
            # Update learning rate
            scheduler.step()
            
            # Log current learning rate
            current_lr = scheduler.get_last_lr()[0]
            logger.debug(f"Learning rate: {current_lr:.2e}")
        
        logger.info("Training completed!")
        return self.training_history
    
    def save_model(self, filename: str) -> None:
        """
        Save the model state dict and configuration.
        
        Args:
            filename (str): Filename to save the model
        """
        save_path = Path(self.config.model_save_path) / filename
        
        # Save model state dict and config
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str) -> None:
        """
        Load a saved model.
        
        Args:
            filename (str): Filename of the saved model
        """
        load_path = Path(self.config.model_save_path) / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Model loaded from {load_path}")


class SentimentPredictor:
    """
    Predictor class for inference with the trained sentiment model.
    """
    
    def __init__(self, model_path: str, config: ModelConfig):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model
            config (ModelConfig): Model configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model
        self.model = BertSentimentModel(config)
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
        
        # Load trained weights
        self.load_model(model_path)
        
        # Sentiment labels
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        logger.info("SentimentPredictor initialized")
    
    def load_model(self, model_path: str) -> None:
        """Load the trained model weights."""
        try:
            # First try to load as weights-only (state_dict directly)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Model weights loaded from {model_path} (weights-only)")
        except Exception as e1:
            try:
                # Try loading as checkpoint with model_state_dict
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                logger.info(f"Model weights loaded from {model_path} (checkpoint)")
            except Exception as e2:
                logger.error(f"Failed to load model: {e1}, {e2}")
                raise e2
    
    def predict_text(self, text: str) -> Dict[str, Union[str, float, List[float]]]:
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            Dict: Prediction results with sentiment, confidence, and probabilities
        """
        # Tokenize text
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            predictions, probabilities = self.model.get_predictions(logits)
        
        # Convert to numpy for easier handling
        prediction = predictions.cpu().item()
        probs = probabilities.cpu().numpy().flatten()
        
        return {
            'text': text,
            'sentiment': self.sentiment_labels[prediction],
            'confidence': float(probs[prediction]),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(self.sentiment_labels, probs)
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts (List[str]): List of texts for sentiment analysis
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for text in texts:
            result = self.predict_text(text)
            results.append(result)
        
        return results


def create_model_from_tokenizer_config(tokenizer_config_path: str) -> BertSentimentModel:
    """
    Create a BERT model based on tokenizer configuration.
    
    Args:
        tokenizer_config_path (str): Path to tokenizer configuration file
        
    Returns:
        BertSentimentModel: Initialized BERT sentiment model
    """
    # Load tokenizer configuration
    with open(tokenizer_config_path, 'r') as f:
        tokenizer_config = json.load(f)
    
    # Create model config based on tokenizer config
    model_config = ModelConfig(
        bert_model_name=tokenizer_config['model_name'],
        max_sequence_length=tokenizer_config['max_sequence_length']
    )
    
    # Create and return model
    model = BertSentimentModel(model_config)
    
    logger.info(f"Model created based on tokenizer config: {tokenizer_config_path}")
    return model


def main():
    """
    Main function demonstrating model usage.
    """
    logger.info("=== BERT Sentiment Model Demo ===")
    
    # Initialize configuration
    config = ModelConfig()
    
    # Create model
    model = BertSentimentModel(config)
    
    logger.info(f"Model created with {model.num_parameters()} parameters")
    logger.info(f"Model architecture: {model}")
    
    # Test forward pass with dummy data
    batch_size = 2
    seq_length = config.max_sequence_length
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length)
    
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)
        predictions, probabilities = model.get_predictions(logits)
    
    logger.info(f"Output shape: {logits.shape}")
    logger.info(f"Predictions: {predictions}")
    logger.info(f"Probabilities shape: {probabilities.shape}")
    
    logger.info("Model demo completed successfully!")


if __name__ == "__main__":
    main()