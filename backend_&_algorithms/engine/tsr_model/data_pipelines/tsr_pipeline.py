import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Dict, List, Tuple, Optional
import logging

# Import local pipeline components
from .price_data import TSRDataLoader, add_technical_indicators, create_sequences
from .config_utils import get_time_interval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSRDataPipeline:
    """
    TSR (Technical Stock Return) data pipeline for price-based stock prediction.
    Creates sequences from historical price patterns and technical indicators.
    """
    
    def __init__(self):
        """
        Initialize the TSR data pipeline.
        """
        pass
        
    
    def _create_price_features(self, price_data: pd.DataFrame,
                               normalize: bool = True) -> pd.DataFrame:
        """
        Create price/technical features for TSR model.
        
        Args:
            price_data: Price data with technical indicators
            normalize: Whether to normalize features
            
        Returns:
            Price feature DataFrame
        """
        # Select price/technical features
        price_features = price_data[['Close', 'SMA_14', 'RSI_14', 'MACD']].copy()
        
        if normalize:
            # Normalize each feature independently
            price_features = (price_features - price_features.mean()) / (price_features.std() + 1e-8)
        
        return price_features
    
    def create_tsr_dataset(self, tickers: List[str], start_date: str, end_date: str,
                             seq_length: int = 60, interval: Optional[str] = None, 
                             normalize: bool = True) -> TensorDataset:
        """
        Create TSR dataset from price sequences and technical indicators.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date for price data (YYYY-MM-DD)
            end_date: End date for price data (YYYY-MM-DD) 
            seq_length: Length of price sequences
            interval: Price data interval ('1hr', '5min', '1d', etc.). If None, reads from config.json
            normalize: Whether to normalize features
            
        Returns:
            TensorDataset with price features and targets
        """
        # Use config interval if not provided
        if interval is None:
            interval = get_time_interval()
            logger.info(f"Using time interval from config: {interval}")
        logger.info(f"Creating TSR dataset for {tickers} from {start_date} to {end_date}")
        
        all_X, all_y = [], []
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}...")
            
            try:
                # 1. Get price data with technical indicators
                tsr_loader = TSRDataLoader(ticker, start_date, end_date, interval=interval)
                price_data = tsr_loader.fetch_data()
                if price_data is None or price_data.empty:
                    logger.warning(f"No price data for {ticker}")
                    continue
                price_data = add_technical_indicators(price_data)
                
                logger.info(f"Price data shape for {ticker}: {price_data.shape}")
                
                # 2. Create price features
                price_features = self._create_price_features(
                    price_data, normalize=normalize
                )
                
                logger.info(f"Price features shape for {ticker}: {price_features.shape}")
                
                # 3. Create sequences
                X, y = create_sequences(price_features, seq_length)
                
                logger.info(f"Sequences for {ticker}: X={X.shape}, y={y.shape}")
                
                all_X.append(X)
                all_y.append(y)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        if not all_X:
            raise ValueError("No data was successfully processed")
        
        # Combine all tickers
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_combined, dtype=torch.float32)
        y_tensor = torch.tensor(y_combined, dtype=torch.float32).unsqueeze(-1)
        
        logger.info(f"Final dataset: X={X_tensor.shape}, y={y_tensor.shape}")
        logger.info(f"Feature dimension: {X_tensor.shape[2]} (4 price features: Close, SMA_14, RSI_14, MACD)")
        
        return TensorDataset(X_tensor, y_tensor)
    
    def get_feature_info(self, ticker: str, start_date: str, end_date: str, 
                        interval: Optional[str] = None) -> Dict:
        """
        Get information about available features for a ticker.
        
        Args:
            ticker: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Price data interval. If None, reads from config.json
            
        Returns:
            Dictionary with feature information
        """
        # Get sample data
        tsr_loader = TSRDataLoader(ticker, start_date, end_date, interval)
        price_data = tsr_loader.fetch_data()
        if price_data is None or price_data.empty:
            raise ValueError(f"No price data available for {ticker}")
        price_data = add_technical_indicators(price_data)
        
        price_features = ['Close', 'SMA_14', 'RSI_14', 'MACD']
        
        return {
            'ticker': ticker,
            'price_features': price_features,
            'total_features': len(price_features),
            'price_data_range': (price_data.index.min(), price_data.index.max())
        }


