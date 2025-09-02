import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
from datetime import datetime

from .models import TSRStockPredictor, AdaptiveTSRPredictor
from .data_pipelines.tsr_pipeline import TSRDataPipeline

# Simple visualization function
def plot_training_loss(train_losses, val_losses=None):
    """Plot training and validation losses."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        if val_losses is not None:
            plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSRTrainer:
    """
    Trainer for the TSR (Technical Stock Return) prediction model.
    """
    
    def __init__(self, model: nn.Module, device: str = "auto"):
        """
        Initialize the trainer.
        
        Args:
            model: The TSR model to train
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model = model
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() 
            else device if device != "auto" else "cpu"
        )
        self.model.to(self.device)
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                   criterion: nn.Module, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, dataset, epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 1e-3, validation_split: float = 0.2,
              save_best: bool = True, save_path: str = "tsr_model.pth") -> Dict:
        """
        Train the TSR model.
        
        Args:
            dataset: PyTorch dataset with price features
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            save_best: Whether to save the best model
            save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Dataset size: {len(dataset)} samples")
        logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Split dataset
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_best:
                    self.save_model(save_path, epoch, val_loss)
                    logger.info(f"  New best model saved! Val Loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs (patience: {patience})")
                break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return self.training_history
    
    def save_model(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'model_config': {
                'price_features': getattr(self.model, 'price_features', 4),
                'hidden_dim': getattr(self.model, 'hidden_dim', 128),
                'use_attention': getattr(self.model, 'use_attention', True)
            },
            'epoch': epoch,
            'val_loss': val_loss,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Val Loss: {checkpoint.get('val_loss', 'unknown')}")
        
        return checkpoint
    
    def evaluate_predictions(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate model predictions and calculate metrics.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(y_batch.cpu())
        
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Direction accuracy (did we predict the right direction?)
        pred_direction = np.sign(predictions.flatten())
        true_direction = np.sign(targets.flatten())
        direction_accuracy = np.mean(pred_direction == true_direction)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'num_samples': len(predictions)
        }


def train_tsr_model(tickers: List[str], start_date: str, end_date: str, 
                       model_type: str = "standard", interval: str = "2h", **kwargs) -> TSRTrainer:
    """
    Main function to train a TSR stock prediction model.
    
    Args:
        tickers: List of stock symbols to train on
        start_date: Start date for data
        end_date: End date for data  
        model_type: 'standard' or 'adaptive'
        **kwargs: Additional training parameters
        
    Returns:
        Trained model instance
    """
    # Create data pipeline
    pipeline = TSRDataPipeline()
    
    # Create dataset
    logger.info("Creating TSR dataset...")
    dataset = pipeline.create_tsr_dataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        seq_length=kwargs.get('seq_length', 60),
        interval=interval,
        normalize=kwargs.get('normalize', True)
    )
    
    # Get sample to determine feature dimensions
    X_sample, _ = dataset[0]
    price_features = X_sample.shape[1]  # Should be 4: Close, SMA_14, RSI_14, MACD
    
    logger.info(f"Price features: {price_features} (Close, SMA_14, RSI_14, MACD)")
    
    # Create model
    if model_type == "adaptive":
        model = AdaptiveTSRPredictor(
            price_features=price_features,
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 3),
            use_attention=kwargs.get('use_attention', True)
        )
    else:
        model = TSRStockPredictor(
            price_features=price_features,
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 3),
            use_attention=kwargs.get('use_attention', True)
        )
    
    # Create trainer
    trainer = TSRTrainer(model)
    
    # Train model
    history = trainer.train(
        dataset=dataset,
        epochs=kwargs.get('epochs', 50),
        batch_size=kwargs.get('batch_size', 32),
        learning_rate=kwargs.get('learning_rate', 1e-3),
        validation_split=kwargs.get('validation_split', 0.2),
        save_path=f"models/tsr_{model_type}_model.pth"
    )
    
    # Plot training history
    if kwargs.get('plot_loss', True):
        plot_training_loss(history['train_loss'], history['val_loss'])
    
    return trainer


