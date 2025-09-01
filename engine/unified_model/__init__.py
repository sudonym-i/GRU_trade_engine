"""
Unified Stock Prediction Model

This module combines technical price analysis (TSR) with financial fundamentals 
to create a more comprehensive stock prediction system.

Key Components:
- UnifiedDataPipeline: Combines historical price and financial data
- UnifiedStockPredictor: Neural network with separate encoders for price/fundamentals
- AdaptiveUnifiedPredictor: Enhanced version with adaptive feature weighting
- UnifiedTrainer: Training pipeline for the combined models

Usage:
    from unified_model import train_unified_model
    
    trainer = train_unified_model(
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2024-01-01"
    )
"""

from .data_pipelines.integrated_data_pipeline import UnifiedDataPipeline
from .integrated_model import UnifiedStockPredictor, AdaptiveUnifiedPredictor  
from .train import UnifiedTrainer, train_unified_model

__all__ = [
    'UnifiedDataPipeline',
    'UnifiedStockPredictor',
    'AdaptiveUnifiedPredictor',
    'UnifiedTrainer', 
    'train_unified_model'
]