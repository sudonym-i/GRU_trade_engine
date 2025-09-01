"""
TSR (Time Series Regression) Model Package

A comprehensive time series regression model for stock price prediction 
and automated trading using GRU neural networks and Financial Modeling Prep API.
"""

__version__ = "2.0.0"
__author__ = "Neural Trade Engine Project"

# Core components
from .model import GRUPredictor
from .data_pipeline import DataLoader, make_dataset
from .utils import add_technical_indicators, create_sequences
from .train import train_gru_predictor
from .trade import TradingSimulator, simulate_trading
from .route import train_model, test_run_model

# Main exports
__all__ = [
    'GRUPredictor',
    'DataLoader', 
    'make_dataset',
    'add_technical_indicators',
    'create_sequences',
    'train_gru_predictor',
    'TradingSimulator',
    'simulate_trading',
    'train_model',
    'test_run_model'
]