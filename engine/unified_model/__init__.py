"""
Unified Stock Prediction Model

This module combines technical price analysis (TSR) with financial fundamentals 
to create a more comprehensive stock prediction system.

Main API Functions:
- train_model: Train a unified prediction model
- predict_price: Predict next stock price using trained model
- get_model_info: Get information about a trained model
- list_available_models: List all trained models

Internal Components:
- UnifiedDataPipeline: Combines historical price and financial data
- UnifiedStockPredictor: Neural network with separate encoders for price/fundamentals
- AdaptiveUnifiedPredictor: Enhanced version with adaptive feature weighting
- UnifiedTrainer: Training pipeline for the combined models

Usage:
    # Main API (recommended)
    from engine.unified_model import train_model, predict_price
    
    model_path = train_model(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")
    prediction = predict_price("AAPL", model_path)
    
    # Advanced usage
    from engine.unified_model import UnifiedTrainer, train_unified_model
    trainer = train_unified_model(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")
"""

# Main API functions (recommended for external use)
from .api import train_model, predict_price, get_model_info, list_available_models

# Internal components (for advanced usage)
from .data_pipelines.integrated_data_pipeline import UnifiedDataPipeline
from .integrated_model import UnifiedStockPredictor, AdaptiveUnifiedPredictor  
from .train import UnifiedTrainer, train_unified_model

__all__ = [
    # Main API
    'train_model',
    'predict_price', 
    'get_model_info',
    'list_available_models',
    
    # Internal components
    'UnifiedDataPipeline',
    'UnifiedStockPredictor',
    'AdaptiveUnifiedPredictor',
    'UnifiedTrainer', 
    'train_unified_model'
]