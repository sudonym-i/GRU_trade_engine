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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSRDataLoader:
    """Data loader for stock price data from Financial Modeling Prep API."""
    
    def __init__(self, ticker: str, start_date: str, end_date: str, 
                 interval: str = "1d", api_key: Optional[str] = None):
        """
        Initialize TSR data loader.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data frequency ('1d', '1h', '5m', etc.)
            api_key: FMP API key
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        
        if not self.api_key:
            raise ValueError("API key required. Set FMP_API_KEY environment variable.")
    
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
        Fetch stock price data from FMP API.
        
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        try:
            fmp_interval = self._get_interval_mapping(self.interval)
            
            if fmp_interval == 'daily':
                # Daily data endpoint
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{self.ticker}"
                params = {
                    'from': self.start_date,
                    'to': self.end_date,
                    'apikey': self.api_key
                }
            else:
                # Intraday data endpoint
                url = f"https://financialmodelingprep.com/api/v3/historical-chart/{fmp_interval}/{self.ticker}"
                params = {
                    'from': self.start_date,
                    'to': self.end_date,
                    'apikey': self.api_key
                }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract historical data
            if fmp_interval == 'daily':
                historical_data = data.get('historical', [])
            else:
                historical_data = data
            
            if not historical_data:
                logger.warning(f"No data found for {self.ticker}")
                return None
            
            # Convert to DataFrame
            df_data = []
            for item in historical_data:
                row = {
                    'Open': item.get('open'),
                    'High': item.get('high'),
                    'Low': item.get('low'),
                    'Close': item.get('close'),
                    'Volume': item.get('volume', 0)
                }
                date_str = item.get('date')
                if date_str:
                    df_data.append((date_str, row))
            
            if not df_data:
                return None
            
            # Create DataFrame with date index
            dates, rows = zip(*df_data)
            df = pd.DataFrame(rows, index=pd.to_datetime(dates))
            df.index.name = 'Date'
            df = df.sort_index()  # Sort by date (oldest first)
            
            # Remove any rows with missing data
            df = df.dropna()
            
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
                      interval: str = "1d", normalize: bool = False) -> pd.DataFrame:
    """
    Get price data with technical indicators for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data frequency
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


if __name__ == "__main__":
    # Test the TSR pipeline
    print("Testing TSR Pipeline...")
    
    try:
        # Test data loading
        loader = TSRDataLoader("AAPL", "2023-01-01", "2024-01-01")
        data = loader.fetch_data()
        
        if data is not None:
            print(f"Loaded {len(data)} records for AAPL")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            # Test technical indicators
            data_with_indicators = add_technical_indicators(data)
            print(f"Data shape with indicators: {data_with_indicators.shape}")
            print(f"Columns: {list(data_with_indicators.columns)}")
            
            # Test sequence creation
            features = data_with_indicators[['Close', 'SMA_14', 'RSI_14', 'MACD']]
            X, y = create_sequences(features, seq_length=30)
            print(f"Sequences: X={X.shape}, y={y.shape}")
            
            # Test get_price_features function
            price_features = get_price_features("AAPL", "2023-01-01", "2024-01-01")
            print(f"Price features shape: {price_features.shape}")
            
            print("SUCCESS: TSR Pipeline test successful!")
        else:
            print("ERROR: Failed to load data")
            
    except Exception as e:
        print(f"ERROR: TSR Pipeline test failed: {e}")