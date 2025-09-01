
import requests
import pandas as pd
import torch
from torch.utils.data import TensorDataset
try:
    from .utils import create_sequences, add_technical_indicators
    from .visualizations import plot_technical_indicators
except ImportError:
    from utils import create_sequences, add_technical_indicators
    from visualizations import plot_technical_indicators
import numpy as np
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This will load .env file from current directory or parent directories
except ImportError:
    # python-dotenv not installed, continue without it
    pass


class DataLoader:
    def __init__(self, tickers, start, end, interval="1d", api_key=None):
        """
        tickers: str or list of str
        start: start date (YYYY-MM-DD)
        end: end date (YYYY-MM-DD)
        interval: data frequency (e.g., '1d', '1h', '5m')
        api_key: Financial Modeling Prep API key
        """
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start = start
        self.end = end
        self.interval = interval
        self.api_key = os.getenv('FMP_API_KEY') or api_key
        self.data = {}
        
        if not self.api_key:
            raise ValueError("API key is required. Set FMP_API_KEY environment variable or pass api_key parameter.")

    def _get_interval_mapping(self, interval):
        """Map yfinance-style intervals to FMP intervals"""
        mapping = {
            '1d': 'daily',
            '1h': '1hour',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min'
        }
        return mapping.get(interval, 'daily')

    def _fetch_data(self, ticker):
        """Fetch data from Financial Modeling Prep API"""
        try:
            if self.interval == '1d':
                # Daily data endpoint
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
                params = {
                    'apikey': self.api_key,
                    'from': self.start,
                    'to': self.end
                }
            else:
                # Intraday data endpoint
                fmp_interval = self._get_interval_mapping(self.interval)
                url = f"https://financialmodelingprep.com/api/v3/historical-chart/{fmp_interval}/{ticker}"
                params = {
                    'apikey': self.api_key,
                    'from': self.start,
                    'to': self.end
                }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if self.interval == '1d':
                # Daily data format
                historical_data = data.get('historical', [])
            else:
                # Intraday data format
                historical_data = data
            
            if not historical_data:
                return None
                
            # Convert to DataFrame with yfinance-compatible format
            df_data = []
            for item in historical_data:
                row = {
                    'Open': item.get('open'),
                    'High': item.get('high'),
                    'Low': item.get('low'),
                    'Close': item.get('close'),
                    'Volume': item.get('volume', 0)
                }
                # Handle date format
                date_str = item.get('date')
                if date_str:
                    df_data.append((date_str, row))
            
            if not df_data:
                return None
                
            # Create DataFrame
            dates, rows = zip(*df_data)
            df = pd.DataFrame(rows, index=pd.to_datetime(dates))
            df.index.name = 'Date'
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def download(self):
        """Download data for all tickers"""
        for ticker in self.tickers:
            df = self._fetch_data(ticker)
            if df is not None and not df.empty:
                df = df.dropna()
                self.data[ticker] = df
            else:
                print(f"Warning: No data retrieved for {ticker}")
        return self.data

    def get(self, ticker):
        return self.data.get(ticker)

    def get_all(self):
        return self.data




def make_dataset(ticker, start, end, seq_length=24, interval="1d", normalize=False, plot_indicators=True):  # e.g., 24 for 24 hours or days
	
    # Accepts a list of tickers, downloads and processes each, and combines all samples
	if isinstance(ticker, str):
		tickers = [ticker]
	else:
		tickers = ticker

	all_X, all_y = [], []
      
	for t in tickers:
		print(f"[INFO] Downloading data for {t} from {start} to {end} with interval {interval}...")
		dl = DataLoader(t, start, end, interval=interval)
		data = dl.download()[t]
		print(f"[INFO] Data shape: {data.shape}")
		
        # Add technical indicators
		data = add_technical_indicators(data)
		print(f"[INFO] Data shape after indicators: {data.shape}")
		
		# Plot technical indicators if requested
		if plot_indicators:
			plot_technical_indicators(data, t)
		
		print(f"[INFO] Creating sequences with sequence length {seq_length}...")
		features = data[['Close', 'SMA_14', 'RSI_14', 'MACD']]

		if(normalize == True):
			# Normalize features (z-score)
			features_norm = (features - features.mean()) / (features.std() + 1e-8)
			X, y = create_sequences(features_norm, seq_length)
			# Normalize target (z-score)
			y_norm = (y - y.mean()) / (y.std() + 1e-8)
		else:
            # Keeping variables the same for sanity
			X, y = create_sequences(features, seq_length)
			y_norm = y

		print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
            
		all_X.append(X)
		all_y.append(y_norm)
	X = np.concatenate(all_X, axis=0)
	y = np.concatenate(all_y, axis=0)
	X = torch.tensor(X, dtype=torch.float32)
	y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
      
	if y.ndim == 3:
		y = y.squeeze(1)
            
	dataset = TensorDataset(X, y)
	print(f"[INFO] Combined dataset created with {len(dataset)} samples from {len(tickers)} tickers.")
	return dataset, X.shape[2]  # input_dim
