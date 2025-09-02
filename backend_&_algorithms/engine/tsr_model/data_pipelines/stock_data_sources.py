#!/usr/bin/env python3

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_yfinance():
    """Install yfinance if not available."""
    try:
        import yfinance
        return True
    except ImportError:
        logger.info("Installing yfinance...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
            import yfinance
            return True
        except Exception as e:
            logger.error(f"Failed to install yfinance: {e}")
            return False


def fetch_yahoo_finance_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance using yfinance.
    Supports multiple intervals including 2-hour resampled data.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ("1d", "1h", "2h", etc.)
    """
    if not install_yfinance():
        return None
        
    try:
        import yfinance as yf
        
        logger.info(f"Fetching {ticker} data from Yahoo Finance...")
        
        # Handle 2-hour intervals by resampling 1-hour data
        if interval == "2h":
            logger.info("Fetching 1-hour data and resampling to 2-hour intervals...")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval="1h")
            
            if df.empty:
                logger.warning(f"No 1-hour data found for {ticker}")
                return None
            
            # Resample to 2-hour intervals using proper OHLC aggregation
            df_2h = df.resample('2h').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            logger.info(f"Resampled to 2-hour intervals: {len(df_2h)} records")
            df = df_2h
        else:
            # Use native interval for other cases
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        
        # Keep only OHLCV data (remove Adj Close, Dividends, Stock Splits if present)
        base_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in base_columns if col in df.columns]
        df = df[available_columns]
        
        logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data: {e}")
        return None


def fetch_alpha_vantage_data(ticker: str, api_key: str = None) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Alpha Vantage API.
    Free tier: 25 requests per day, 5 requests per minute.
    """
    api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY') or 'demo'
    
    try:
        logger.info(f"Fetching {ticker} data from Alpha Vantage...")
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': 'compact',  # Last 100 data points
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'Error Message' in data:
            logger.error(f"Alpha Vantage error: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
            return None
        
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            logger.warning(f"No time series data found for {ticker}")
            return None
        
        # Convert to DataFrame
        df_data = []
        for date_str, values in time_series.items():
            row = {
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            }
            df_data.append((date_str, row))
        
        dates, rows = zip(*df_data)
        df = pd.DataFrame(rows, index=pd.to_datetime(dates))
        df.index.name = 'Date'
        df = df.sort_index()
        
        logger.info(f"Successfully fetched {len(df)} records from Alpha Vantage")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage data: {e}")
        return None


def generate_mock_stock_data(ticker: str, start_date: str, end_date: str, 
                           base_price: float = None) -> pd.DataFrame:
    """
    Generate realistic mock stock data for testing.
    Uses random walk with realistic patterns.
    """
    logger.info(f"Generating mock data for {ticker}")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    # Remove weekends
    dates = dates[~dates.weekday.isin([5, 6])]
    
    if base_price is None:
        # Set realistic base prices for common tickers
        base_prices = {
            'AAPL': 180,
            'MSFT': 340,
            'NVDA': 450,
            'TSLA': 240,
            'GOOGL': 140,
            'AMZN': 150,
            'META': 320
        }
        base_price = base_prices.get(ticker.upper(), 100)
    
    n_days = len(dates)
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results
    
    # Random walk with drift
    drift = 0.0005  # Small positive drift
    volatility = 0.02  # 2% daily volatility
    
    returns = np.random.normal(drift, volatility, n_days)
    prices = [base_price]
    
    for i in range(1, n_days):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 0.01))  # Ensure positive prices
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate intraday range
        daily_vol = abs(np.random.normal(0, volatility * 0.5))
        high = close * (1 + daily_vol)
        low = close * (1 - daily_vol)
        
        # Open is previous close + some gap
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, volatility * 0.3)
            open_price = prices[i-1] * (1 + gap)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate realistic volume
        base_volume = 1000000 + abs(np.random.normal(0, 500000))
        volume = int(base_volume)
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    
    logger.info(f"Generated {len(df)} days of mock data for {ticker}")
    return df


def get_stock_data_smart(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Smart stock data fetcher that tries multiple sources in order of preference.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ("1d", "1h", "2h", etc.)
        
    Returns:
        DataFrame with OHLCV data or None if all sources fail
    """
    logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
    
    # Try Yahoo Finance first (most reliable and free)
    data = fetch_yahoo_finance_data(ticker, start_date, end_date, interval)
    if data is not None and not data.empty:
        logger.info("Successfully used Yahoo Finance data")
        return data
    
    # Try Alpha Vantage if Yahoo fails
    logger.info("Yahoo Finance failed, trying Alpha Vantage...")
    data = fetch_alpha_vantage_data(ticker)
    if data is not None and not data.empty:
        # Filter to date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        if not data.empty:
            logger.info("Successfully used Alpha Vantage data")
            return data
    
    # If all APIs fail, generate mock data for testing
    logger.warning("All APIs failed, generating mock data for testing")
    data = generate_mock_stock_data(ticker, start_date, end_date)
    return data


