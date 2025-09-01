"""
Unified Model API

Main API functions for the unified stock prediction model.
Provides simple interfaces for training models and making predictions.
"""

import torch
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .train import train_unified_model, UnifiedTrainer
from .integrated_model import UnifiedStockPredictor, AdaptiveUnifiedPredictor
from .data_pipelines.integrated_data_pipeline import UnifiedDataPipeline
from .data_pipelines.stock_pipeline import get_price_features
from .data_pipelines.financial_pipeline import get_financial_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(tickers: List[str], start_date: str, end_date: str,
                model_type: str = "standard", save_path: Optional[str] = None,
                **kwargs) -> str:
    """
    Train a unified stock prediction model.
    
    Args:
        tickers: List of stock symbols to train on (e.g., ["AAPL", "MSFT"])
        start_date: Training data start date (YYYY-MM-DD)
        end_date: Training data end date (YYYY-MM-DD)
        model_type: "standard" or "adaptive"
        save_path: Where to save the trained model (optional)
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
    logger.info(f"Training {model_type} unified model on {len(tickers)} tickers")
    logger.info(f"Training data: {start_date} to {end_date}")
    
    # Set default save path
    if save_path is None:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/unified_{model_type}_model_{timestamp}.pth"
    
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
        trainer = train_unified_model(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
            save_path=save_path,
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
                 days_history: int = 60, 
                 include_confidence: bool = True) -> Dict[str, Any]:
    """
    Predict the next stock price using a trained unified model.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        model_path: Path to trained model file
        prediction_date: Date to predict from (YYYY-MM-DD). If None, uses latest data.
        days_history: Number of days of historical data to use
        include_confidence: Whether to include confidence intervals
        
    Returns:
        Dictionary with prediction results
        
    Example:
        result = predict_price(
            ticker="AAPL",
            model_path="models/unified_model.pth",
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
        model_config = checkpoint['model_config']
        
        # Create model instance
        if checkpoint['model_class'] == 'AdaptiveUnifiedPredictor':
            model = AdaptiveUnifiedPredictor(**model_config)
        else:
            model = UnifiedStockPredictor(**model_config)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Prepare prediction date and data range
        if prediction_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = prediction_date
        
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days_history + 30)).strftime('%Y-%m-%d')
        
        # Create data pipeline and get recent data
        pipeline = UnifiedDataPipeline()
        dataset = pipeline.create_unified_dataset(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            seq_length=days_history,
            normalize=True
        )
        
        if len(dataset) == 0:
            raise ValueError(f"No data available for {ticker} in the specified range")
        
        # Get the most recent sequence
        X_recent, _ = dataset[-1]  # Most recent sequence
        X_batch = X_recent.unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            if include_confidence:
                result = model.predict_with_confidence(X_batch, num_samples=10)
                
                prediction_result = {
                    'ticker': ticker,
                    'predicted_price': float(result['prediction'].item()),
                    'confidence_std': float(result['std'].item()),
                    'confidence_interval': [
                        float(result['ci_lower'].item()),
                        float(result['ci_upper'].item())
                    ],
                    'uncertainty': float(result['uncertainty'].item()),
                    'prediction_date': end_date,
                    'model_path': model_path
                }
            else:
                prediction = model(X_batch)
                prediction_result = {
                    'ticker': ticker,
                    'predicted_price': float(prediction.item()),
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


if __name__ == "__main__":
    # Example usage
    print("=== Unified Model API Demo ===")
    
    # Check for API key
    if not os.getenv('FMP_API_KEY'):
        print("❌ FMP_API_KEY not set. Please set your API key:")
        print("export FMP_API_KEY=your_api_key_here")
        exit(1)
    
    try:
        # Demo training (uncomment to run)
        # print("Training model...")
        # model_path = train_model(
        #     tickers=["AAPL"],
        #     start_date="2023-01-01",
        #     end_date="2024-01-01",
        #     epochs=5,  # Short for demo
        #     batch_size=16
        # )
        
        # Demo prediction (requires trained model)
        models = list_available_models()
        if models:
            print(f"Found {len(models)} trained models:")
            for model in models[:3]:  # Show first 3
                print(f"  - {model['file_name']} ({model['model_class']})")
            
            # Use most recent model for prediction demo
            latest_model = models[0]['file_path']
            print(f"\nTesting prediction with: {latest_model}")
            
            # result = predict_price("AAPL", latest_model)
            # print(f"Predicted price: ${result['predicted_price']:.2f}")
        else:
            print("No trained models found. Train a model first.")
            
    except Exception as e:
        print(f"Demo failed: {e}")