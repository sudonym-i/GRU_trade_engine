"""
TSR Model API

Main API functions for the TSR (Technical Stock Return) prediction model.
Provides simple interfaces for training models and making predictions.
"""

import torch
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .training import train_tsr_model
from .models import TSRStockPredictor, AdaptiveTSRPredictor
from .data_pipelines.tsr_pipeline import TSRDataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(tickers: List[str], start_date: str, end_date: str,
                model_type: str = "standard", save_path: Optional[str] = None,
                interval: str = "2h", **kwargs) -> str:
    """
    Train a TSR stock prediction model.
    
    Args:
        tickers: List of stock symbols to train on (e.g., ["AAPL", "MSFT"])
        start_date: Training data start date (YYYY-MM-DD)
        end_date: Training data end date (YYYY-MM-DD)
        model_type: "standard" or "adaptive"
        save_path: Where to save the trained model (optional)
        interval: Data interval ("2h", "1h", "1d", etc.)
        **kwargs: Additional training parameters
        
    Returns:
        Path to the saved model file
        
    Example:
        model_path = train_model(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date="2020-01-01",
            end_date="2024-01-01",
            epochs=50,
            batch_size=32
        )
    """
    logger.info(f"Training {model_type} TSR model on {len(tickers)} tickers")
    logger.info(f"Training data: {start_date} to {end_date}")
    
    # Set default save path
    if save_path is None:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/tsr_{model_type}_model_{timestamp}.pth"
    
    # Default training parameters
    default_params = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'seq_length': 60,
        'hidden_dim': 128,
        'use_attention': True,
        'plot_loss': False
    }
    default_params.update(kwargs)
    
    try:
        # Train the model
        trainer = train_tsr_model(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
            save_path=save_path,
            interval=interval,
            **default_params
        )
        
        # Get training summary
        history = trainer.training_history
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Final train loss: {final_train_loss:.6f}")
        logger.info(f"Final validation loss: {final_val_loss:.6f}")
        logger.info(f"Model saved to: {save_path}")
        
        return save_path
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def predict_price(ticker: str, model_path: str, 
                 prediction_date: Optional[str] = None,
                 days_history: int = 30,
                 interval: str = "2h",
                 include_confidence: bool = True) -> Dict[str, Any]:
    """
    Predict the next stock price using a trained TSR model.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        model_path: Path to trained model file
        prediction_date: Date to predict from (YYYY-MM-DD). If None, uses latest data.
        days_history: Number of days of historical data to use
        interval: Data interval ("2h", "1h", "1d", etc.)
        include_confidence: Whether to include confidence intervals
        
    Returns:
        Dictionary with prediction results
        
    Example:
        result = predict_price(
            ticker="AAPL",
            model_path="models/tsr_model.pth",
            include_confidence=True
        )
        print(f"Predicted price: ${result['predicted_price']:.2f}")
        print(f"Confidence: {result['confidence_interval']}")
    """
    logger.info(f"Predicting price for {ticker} using model: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load model checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint['model_config'].copy()
        
        # Remove financial_features if present (for backward compatibility)
        if 'financial_features' in model_config:
            del model_config['financial_features']
        
        # Create model instance (handle old model class names for backward compatibility)
        model_class = checkpoint.get('model_class', 'TSRStockPredictor')
        if model_class in ['AdaptiveTSRPredictor', 'AdaptiveUnifiedPredictor']:
            model = AdaptiveTSRPredictor(**model_config)
        else:
            model = TSRStockPredictor(**model_config)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Prepare prediction date and data range
        if prediction_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = prediction_date
        
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days_history + 120)).strftime('%Y-%m-%d')
        
        # Create data pipeline and get recent data
        pipeline = TSRDataPipeline()
        
        # First try with the model's expected sequence length
        model_seq_length = 60  # Default model sequence length
        try:
            dataset = pipeline.create_tsr_dataset(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                seq_length=model_seq_length,
                interval=interval,
                normalize=True
            )
        except ValueError as e:
            if "must be greater than sequence length" in str(e):
                # Not enough data for 60-day sequences, try with less data
                logger.warning(f"Not enough data for {model_seq_length}-day sequences, trying with available data")
                
                # Get raw data to see how much we have
                from .data_pipelines.price_data import TSRDataLoader, add_technical_indicators
                tsr_loader = TSRDataLoader(ticker, start_date, end_date, interval)
                raw_data = tsr_loader.fetch_data()
                if raw_data is not None and not raw_data.empty:
                    price_data = add_technical_indicators(raw_data)
                    available_days = len(price_data)
                    # Use 80% of available data for sequence length
                    adapted_seq_length = max(10, int(available_days * 0.8))
                    logger.info(f"Using adapted sequence length: {adapted_seq_length} (from {available_days} available days)")
                    
                    dataset = pipeline.create_tsr_dataset(
                        tickers=[ticker],
                        start_date=start_date,
                        end_date=end_date,
                        seq_length=adapted_seq_length,
                        interval=interval,
                        normalize=True
                    )
                else:
                    raise ValueError(f"No price data available for {ticker}")
            else:
                raise e
        
        if len(dataset) == 0:
            raise ValueError(f"No data available for {ticker} in the specified range")
        
        # Get the most recent sequence
        X_recent, _ = dataset[-1]  # Most recent sequence
        
        # If sequence length doesn't match model expectation, adapt it
        current_seq_len = X_recent.size(0)
        if current_seq_len != model_seq_length:
            logger.warning(f"Adapting sequence length from {current_seq_len} to {model_seq_length}")
            if current_seq_len < model_seq_length:
                # Pad with the first available values (repeat early data)
                pad_size = model_seq_length - current_seq_len
                padding = X_recent[0].unsqueeze(0).repeat(pad_size, 1)
                X_recent = torch.cat([padding, X_recent], dim=0)
            else:
                # Truncate to match model expectation (take most recent data)
                X_recent = X_recent[-model_seq_length:]
        
        X_batch = X_recent.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get the normalization parameters to denormalize the prediction
        # We need to get the raw close prices to calculate the normalization params
        from .data_pipelines.price_data import TSRDataLoader, add_technical_indicators
        tsr_loader = TSRDataLoader(ticker, start_date, end_date, interval)
        raw_data = tsr_loader.fetch_data()
        if raw_data is None or raw_data.empty:
            raise ValueError(f"No price data available for {ticker}")
        price_data_unnorm = add_technical_indicators(raw_data)
        
        # Get current price (most recent close price)
        current_price = float(price_data_unnorm['Close'].iloc[-1])
        
        # Calculate normalization parameters for Close price (same as in training)
        close_mean = price_data_unnorm['Close'].mean()
        close_std = price_data_unnorm['Close'].std() + 1e-8
        
        # Make prediction
        with torch.no_grad():
            if include_confidence:
                result = model.predict_with_confidence(X_batch, num_samples=10)
                
                # Denormalize predictions
                predicted_price = float(result['prediction'].item()) * close_std + close_mean
                ci_lower = float(result['ci_lower'].item()) * close_std + close_mean
                ci_upper = float(result['ci_upper'].item()) * close_std + close_mean
                
                prediction_result = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'confidence_std': float(result['std'].item()) * close_std,  # Scale std too
                    'confidence_interval': [ci_lower, ci_upper],
                    'uncertainty': float(result['uncertainty'].item()),
                    'prediction_date': end_date,
                    'model_path': model_path
                }
            else:
                prediction = model(X_batch)
                # Denormalize prediction back to actual price scale
                predicted_price = float(prediction.item()) * close_std + close_mean
                
                prediction_result = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'prediction_date': end_date,
                    'model_path': model_path
                }
        
        logger.info(f"Prediction for {ticker}: ${prediction_result['predicted_price']:.2f}")
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a trained model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary with model information
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    info = {
        'model_class': checkpoint.get('model_class', 'Unknown'),
        'model_config': checkpoint.get('model_config', {}),
        'training_epoch': checkpoint.get('epoch', 'Unknown'),
        'validation_loss': checkpoint.get('val_loss', 'Unknown'),
        'timestamp': checkpoint.get('timestamp', 'Unknown'),
        'file_size_mb': os.path.getsize(model_path) / (1024 * 1024)
    }
    
    if 'training_history' in checkpoint:
        history = checkpoint['training_history']
        info['training_history'] = {
            'epochs_trained': len(history.get('train_loss', [])),
            'final_train_loss': history.get('train_loss', [])[-1] if history.get('train_loss') else None,
            'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
            'best_val_loss': min(history.get('val_loss', [])) if history.get('val_loss') else None
        }
    
    return info


def list_available_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """
    List all available trained models.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        List of model information dictionaries
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    model_files = list(models_path.glob("*.pth"))
    models_info = []
    
    for model_file in model_files:
        try:
            info = get_model_info(str(model_file))
            info['file_name'] = model_file.name
            info['file_path'] = str(model_file)
            models_info.append(info)
        except Exception as e:
            logger.warning(f"Could not load model info for {model_file}: {e}")
    
    # Sort by timestamp (newest first)
    models_info.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return models_info


