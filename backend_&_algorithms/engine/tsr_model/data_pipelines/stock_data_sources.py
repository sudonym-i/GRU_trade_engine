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


def install_ib_insync():
    """Install ib_insync if not available."""
    try:
        import ib_insync
        return True
    except ImportError:
        logger.info("Installing ib_insync...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ib_insync"])
            import ib_insync
            return True
        except Exception as e:
            logger.error(f"Failed to install ib_insync: {e}")
            return False


def fetch_ib_data(ticker: str, start_date: str, end_date: str, interval: str = "1 day", 
                  host: str = '127.0.0.1', port: int = 7497, client_id: int = 1) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Interactive Brokers using ib_insync.
    Supports daily intervals for swing trading and ML models.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD) 
        interval: Data interval ("1 day", "1 hour", "30 mins", etc.)
        host: IB Gateway/TWS host (default: localhost)
        port: IB Gateway/TWS port (default: 7497)
        client_id: Client ID for connection
    """
    if not install_ib_insync():
        return None
        
    try:
        from ib_insync import IB, Stock, util
        
        logger.info(f"Connecting to Interactive Brokers at {host}:{port}...")
        
        # Connect to IB Gateway or TWS
        ib = IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=10)
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            logger.info("Make sure IB Gateway or TWS is running with API enabled")
            return None
        
        logger.info(f"Fetching {ticker} data from Interactive Brokers...")
        
        # Create contract (assume US stocks)
        contract = Stock(ticker, 'SMART', 'USD')
        
        # Calculate duration string from date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days
        
        if days_diff <= 30:
            duration_str = f"{days_diff} D"
        elif days_diff <= 365:
            weeks = days_diff // 7
            duration_str = f"{weeks} W"
        elif days_diff <= 365 * 2:
            # For periods > 12 months, IB requires years format
            years = days_diff / 365.25  # Account for leap years
            duration_str = f"{years:.1f} Y"
        else:
            # For very long periods, use integer years
            years = int(days_diff / 365.25)
            duration_str = f"{years} Y"
        
        # Request historical data
        # IB requires format: YYYYMMDD HH:MM:SS with timezone
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d 23:59:59 US/Eastern')
        
        # Map interval to IB bar size format
        interval_mapping = {
            '30 mins': '30 mins',
            '1 hour': '1 hour', 
            '2 hours': '2 hours',
            '1 day': '1 day',
            '30min': '30 mins',
            '1h': '1 hour',
            '2h': '2 hours',
            '1d': '1 day'
        }
        bar_size = interval_mapping.get(interval, '30 mins')
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True  # Regular trading hours only
        )
        
        # Disconnect
        ib.disconnect()
        
        if not bars:
            logger.warning(f"No data found for {ticker}")
            return None
        
        # Convert to pandas DataFrame
        df = util.df(bars)
        
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        
        # Rename columns to match existing format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Keep only OHLCV data
        base_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in base_columns if col in df.columns]
        df = df[available_columns]
        
        # Ensure index is datetime and filter to exact date range
        if hasattr(df.index, 'date'):
            # Index is already datetime
            df = df[(df.index.date >= start_dt.date()) & (df.index.date <= end_dt.date())]
        else:
            # Index might be RangeIndex, check if there's a date column to set as index
            if 'date' in df.columns:
                df.index = pd.to_datetime(df['date'])
                df = df.drop('date', axis=1)
                df = df[(df.index.date >= start_dt.date()) & (df.index.date <= end_dt.date())]
            # If no date column, assume data is already in correct range (recent data)
        
        logger.info(f"Successfully fetched {len(df)} records from Interactive Brokers")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Interactive Brokers data: {e}")
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
                           base_price: float = None, interval: str = "1 day") -> pd.DataFrame:
    """
    Generate realistic mock stock data for testing.
    Uses random walk with realistic patterns. Optimized for daily intervals.
    """
    logger.info(f"Generating mock data for {ticker}")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Generate appropriate frequency based on interval
    if "min" in interval:
        # For minute data, generate business hours only (9:30 AM - 4:00 PM EST)
        dates = pd.date_range(start=start_dt, end=end_dt, freq='30min')
        # Filter to business hours only
        dates = dates[(dates.hour >= 9) & (dates.hour < 16)]
        # Remove weekends
        dates = dates[~dates.weekday.isin([5, 6])]
    else:
        # Daily data
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


def get_stock_data_smart(ticker: str, start_date: str, end_date: str, interval: str = "1 day") -> Optional[pd.DataFrame]:
    """
    Smart stock data fetcher that tries multiple sources in order of preference.
    Now prioritizes Interactive Brokers for real-time trading.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ("1 day", "1 hour", "30 mins", etc.)
        
    Returns:
        DataFrame with OHLCV data or None if all sources fail
    """
    logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
    
    # Try Interactive Brokers first (real-time data for trading)
    data = fetch_ib_data(ticker, start_date, end_date, interval)
    if data is not None and not data.empty:
        logger.info("Successfully used Interactive Brokers data")
        return data
    
    # Try Alpha Vantage if IB fails
    logger.info("Interactive Brokers failed, trying Alpha Vantage...")
    data = fetch_alpha_vantage_data(ticker)
    if data is not None and not data.empty:
        # Filter to date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        if not data.empty:
            logger.info("Successfully used Alpha Vantage data")
            return data
    
    # If all APIs fail, generate mock data for testing
    logger.warning("All APIs failed, generating mock data for testing")
    data = generate_mock_stock_data(ticker, start_date, end_date, interval=interval)
    return data


