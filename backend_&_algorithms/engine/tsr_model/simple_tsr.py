#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def get_stock_data(ticker: str, days: int = 100) -> Optional[pd.DataFrame]:
    """Get stock data using Yahoo Finance."""
    try:
        import yfinance as yf
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"No data for {ticker}")
            return None
            
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple technical indicators."""
    df = df.copy()
    
    # Simple moving averages
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

def predict_price(ticker: str, days_ahead: int = 1) -> Dict:
    """Simple price prediction using linear trend."""
    try:
        df = get_stock_data(ticker)
        if df is None or len(df) < 20:
            return {"error": "Insufficient data"}
        
        df = add_indicators(df)
        
        # Simple linear trend prediction
        recent_prices = df['Close'].tail(10).values
        trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)
        
        current_price = df['Close'].iloc[-1]
        predicted_price = trend[0] * days_ahead + current_price
        
        # Get latest indicators
        latest_rsi = df['RSI'].iloc[-1]
        latest_sma_10 = df['SMA_10'].iloc[-1]
        latest_sma_20 = df['SMA_20'].iloc[-1]
        
        # Simple signal
        if latest_sma_10 > latest_sma_20 and latest_rsi < 70:
            signal = "BUY"
        elif latest_sma_10 < latest_sma_20 and latest_rsi > 30:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return {
            "ticker": ticker,
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "days_ahead": days_ahead,
            "signal": signal,
            "rsi": float(latest_rsi),
            "sma_10": float(latest_sma_10),
            "sma_20": float(latest_sma_20),
            "trend_slope": float(trend[0])
        }
        
    except Exception as e:
        return {"error": str(e)}

def train_model(ticker: str, days: int = 252) -> Dict:
    """Mock training - just return data stats."""
    try:
        df = get_stock_data(ticker, days)
        if df is None:
            return {"error": "No training data"}
        
        df = add_indicators(df)
        
        return {
            "ticker": ticker,
            "training_days": len(df),
            "price_range": [float(df['Close'].min()), float(df['Close'].max())],
            "avg_volume": float(df['Volume'].mean()),
            "volatility": float(df['Close'].pct_change().std() * np.sqrt(252)),
            "status": "trained"
        }
    except Exception as e:
        return {"error": str(e)}

def get_model_info(ticker: str = None) -> Dict:
    """Get model information."""
    return {
        "model_type": "Simple TSR",
        "version": "1.0",
        "features": ["SMA_10", "SMA_20", "RSI"],
        "supported_tickers": "Any",
        "prediction_horizon": "1-30 days"
    }