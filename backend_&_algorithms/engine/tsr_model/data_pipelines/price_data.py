"""
TSR (Technical Stock Research) Data Pipeline

Lightweight data pipeline for fetching and processing historical stock price data
with technical indicators. Used by the unified model for price/technical features.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .config_utils import get_time_interval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSRDataLoader:
    """Data loader for stock price data from Financial Modeling Prep API."""
    
    def __init__(self, ticker: str, start_date: str, end_date: str, 
                 interval: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize TSR data loader.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data frequency ('1d', '1h', '5m', etc.). If None, reads from config.json
            api_key: FMP API key
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        # Use config interval if not provided
        self.interval = interval if interval is not None else get_time_interval()
        # Yahoo Finance doesn't need an API key
        self.api_key = api_key  # Keep for compatibility but not required
        
        logger.info(f"TSRDataLoader initialized with interval: {self.interval}")
    
    def _get_interval_mapping(self, interval: str) -> str:
        """Map interval to FMP API format."""
        mapping = {
            '1d': 'daily',
            '1h': '1hour',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min'
        }
        return mapping.get(interval, 'daily')
    
    def fetch_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch stock price data using alternative sources.
        
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        try:
            # Use the smart fetcher
            from .stock_data_sources import get_stock_data_smart
            
            df = get_stock_data_smart(self.ticker, self.start_date, self.end_date, self.interval)
            
            if df is None or df.empty:
                logger.warning(f"No data found for {self.ticker}")
                return None
            
            logger.info(f"Fetched {len(df)} records for {self.ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.ticker}: {e}")
            return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to price data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Simple Moving Average (14-day)
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    
    # Relative Strength Index (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)  # Avoid division by zero
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # Remove rows with NaN values created by indicators
    df = df.dropna()
    
    return df


def create_sequences(data: pd.DataFrame, seq_length: int):
    """
    Convert DataFrame to sequences for time series prediction.
    
    Args:
        data: DataFrame with features
        seq_length: Length of input sequences
        
    Returns:
        Tuple of (X, y) where X is sequences and y is targets
    """
    if len(data) <= seq_length:
        raise ValueError(f"Data length ({len(data)}) must be greater than sequence length ({seq_length})")
    
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        # Input sequence
        x = data.iloc[i:(i + seq_length)].values
        # Target (next Close price)
        y = data.iloc[i + seq_length]["Close"]
        
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)


def get_price_features(ticker: str, start_date: str, end_date: str, 
                      interval: Optional[str] = None, normalize: bool = False) -> pd.DataFrame:
    """
    Get price data with technical indicators for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data frequency. If None, reads from config.json
        normalize: Whether to normalize features
        
    Returns:
        DataFrame with price and technical features
    """
    # Fetch data
    loader = TSRDataLoader(ticker, start_date, end_date, interval)
    price_data = loader.fetch_data()
    
    if price_data is None or price_data.empty:
        raise ValueError(f"No price data available for {ticker}")
    
    # Add technical indicators
    price_data = add_technical_indicators(price_data)
    
    # Select features for modeling
    features = price_data[['Close', 'SMA_14', 'RSI_14', 'MACD']].copy()
    
    if normalize:
        # Z-score normalization
        features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features


