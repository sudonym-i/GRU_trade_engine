"""
TSR Stock Prediction Model

This module provides technical stock return (TSR) prediction using price patterns 
and technical indicators.

Main API Functions:
- train_model: Train a TSR prediction model
- predict_price: Predict next stock price using trained model
- get_model_info: Get information about a trained model
- list_available_models: List all trained models

Internal Components:
- TSRDataPipeline: Processes historical price data and technical indicators
- TSRStockPredictor: Neural network for price pattern recognition
- AdaptiveTSRPredictor: Enhanced version with adaptive attention weighting
- TSRTrainer: Training pipeline for the TSR models

Usage:
    # Main API (recommended)
    from engine.tsr_model import train_model, predict_price
    
    model_path = train_model(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")
    prediction = predict_price("AAPL", model_path)
    
    # Advanced usage
    from engine.tsr_model import TSRTrainer, train_tsr_model
    trainer = train_tsr_model(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")
"""

# Main API functions (recommended for external use)
from .api import train_model, predict_price, get_model_info, list_available_models

# Internal components (for advanced usage)
from .data_pipelines.tsr_pipeline import TSRDataPipeline
from .models import TSRStockPredictor, AdaptiveTSRPredictor  
from .training import TSRTrainer, train_tsr_model

__all__ = [
    # Main API
    'train_model',
    'predict_price', 
    'get_model_info',
    'list_available_models',
    
    # Internal components
    'TSRDataPipeline',
    'TSRStockPredictor',
    'AdaptiveTSRPredictor',
    'TSRTrainer', 
    'train_tsr_model'
]