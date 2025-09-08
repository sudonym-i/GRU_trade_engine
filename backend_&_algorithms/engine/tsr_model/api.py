#!/usr/bin/env python3

import logging
from typing import Dict
from .simple_tsr import predict_price, train_model, get_model_info

logger = logging.getLogger(__name__)

def train_tsr_model(ticker: str, days: int = 252) -> Dict:
    """Train TSR model for ticker."""
    return train_model(ticker, days)

def make_prediction(ticker: str, days_ahead: int = 1) -> Dict:
    """Make price prediction."""
    return predict_price(ticker, days_ahead)

def get_tsr_info(ticker: str = None) -> Dict:
    """Get model information."""
    return get_model_info(ticker)