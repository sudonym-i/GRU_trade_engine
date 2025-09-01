import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Dict, List, Tuple, Optional
import logging

# Import local pipeline components
from .stock_pipeline import TSRDataLoader, add_technical_indicators, create_sequences
from .financial_pipeline import FinancialDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDataPipeline:
    """
    Unified data pipeline combining TSR (Technical/Price) data with Financial fundamentals.
    Creates sequences that include both historical price patterns and quarterly financial metrics.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the unified data pipeline.
        
        Args:
            api_key: FMP API key for financial data
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set FMP_API_KEY environment variable or pass api_key parameter.")
        
        self.financial_fetcher = FinancialDataFetcher(api_key=self.api_key)
        
    def _align_financial_with_price_data(self, price_data: pd.DataFrame, 
                                       financial_data: pd.DataFrame, 
                                       symbol: str) -> pd.DataFrame:
        """
        Align quarterly financial data with daily price data by forward-filling.
        
        Args:
            price_data: Daily OHLCV data with technical indicators
            financial_data: Quarterly financial fundamentals
            symbol: Stock symbol for logging
            
        Returns:
            DataFrame with financial features aligned to price data dates
        """
        logger.info(f"Aligning financial data for {symbol}")
        
        # Ensure price_data has DatetimeIndex
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)
        
        # Prepare financial data with dates
        if 'date' not in financial_data.columns:
            logger.warning(f"No date column in financial data for {symbol}")
            return None
        
        financial_data = financial_data.copy()
        financial_data['date'] = pd.to_datetime(financial_data['date'])
        financial_data = financial_data.set_index('date').sort_index()
        
        # Remove non-numeric columns for alignment
        financial_features = financial_data.select_dtypes(include=[np.number])
        
        # Create aligned dataset by reindexing and forward filling
        price_start = price_data.index.min()
        price_end = price_data.index.max()
        
        # Filter financial data to relevant date range
        financial_in_range = financial_features[
            (financial_features.index >= price_start - pd.Timedelta(days=365)) &
            (financial_features.index <= price_end)
        ]
        
        if financial_in_range.empty:
            logger.warning(f"No financial data in price data date range for {symbol}")
            return None
        
        # Reindex to price data dates and forward fill
        aligned_financial = financial_in_range.reindex(
            price_data.index, 
            method='ffill'
        ).fillna(method='bfill')
        
        # If still missing, fill with median values
        aligned_financial = aligned_financial.fillna(aligned_financial.median())
        
        logger.info(f"Aligned {len(aligned_financial.columns)} financial features to {len(price_data)} price points")
        return aligned_financial
    
    def _create_unified_features(self, price_data: pd.DataFrame, 
                               financial_data: pd.DataFrame,
                               normalize: bool = True) -> pd.DataFrame:
        """
        Combine price/technical features with financial features.
        
        Args:
            price_data: Price data with technical indicators
            financial_data: Aligned financial data
            normalize: Whether to normalize features
            
        Returns:
            Combined feature DataFrame
        """
        # Select price/technical features (same as TSR model)
        price_features = price_data[['Close', 'SMA_14', 'RSI_14', 'MACD']].copy()
        
        # Combine with financial features
        if financial_data is not None:
            combined_features = pd.concat([price_features, financial_data], axis=1)
        else:
            combined_features = price_features
        
        if normalize:
            # Normalize each feature independently
            combined_features = (combined_features - combined_features.mean()) / (combined_features.std() + 1e-8)
        
        return combined_features
    
    def create_unified_dataset(self, tickers: List[str], start_date: str, end_date: str,
                             seq_length: int = 60, interval: str = "1d", 
                             normalize: bool = True, financial_periods: int = 20) -> TensorDataset:
        """
        Create unified dataset combining price sequences with financial fundamentals.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date for price data (YYYY-MM-DD)
            end_date: End date for price data (YYYY-MM-DD) 
            seq_length: Length of price sequences
            interval: Price data interval ('1d', '1h', etc.)
            normalize: Whether to normalize features
            financial_periods: Number of quarterly periods for financial data
            
        Returns:
            TensorDataset with combined features and targets
        """
        logger.info(f"Creating unified dataset for {tickers} from {start_date} to {end_date}")
        
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
                
                # 2. Get financial data  
                financial_data = self.financial_fetcher.get_historical_features(
                    ticker, periods=financial_periods, period_type="quarter", normalize=False
                )
                
                if financial_data.empty:
                    logger.warning(f"No financial data for {ticker}, using price data only")
                    financial_data = None
                else:
                    logger.info(f"Financial data shape for {ticker}: {financial_data.shape}")
                
                # 3. Align financial data with price data
                if financial_data is not None:
                    aligned_financial = self._align_financial_with_price_data(
                        price_data, financial_data, ticker
                    )
                else:
                    aligned_financial = None
                
                # 4. Create unified features
                unified_features = self._create_unified_features(
                    price_data, aligned_financial, normalize=normalize
                )
                
                logger.info(f"Unified features shape for {ticker}: {unified_features.shape}")
                
                # 5. Create sequences (same as TSR model)
                X, y = create_sequences(unified_features, seq_length)
                
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
        logger.info(f"Feature dimension: {X_tensor.shape[2]} (4 price + {X_tensor.shape[2]-4} financial)")
        
        return TensorDataset(X_tensor, y_tensor)
    
    def get_feature_info(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """
        Get information about available features for a ticker.
        
        Args:
            ticker: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with feature information
        """
        # Get sample data
        tsr_loader = TSRDataLoader(ticker, start_date, end_date)
        price_data = tsr_loader.fetch_data()
        if price_data is None or price_data.empty:
            raise ValueError(f"No price data available for {ticker}")
        price_data = add_technical_indicators(price_data)
        
        financial_data = self.financial_fetcher.get_historical_features(
            ticker, periods=4, period_type="quarter", normalize=False
        )
        
        price_features = ['Close', 'SMA_14', 'RSI_14', 'MACD']
        financial_features = list(financial_data.columns) if not financial_data.empty else []
        
        return {
            'ticker': ticker,
            'price_features': price_features,
            'financial_features': financial_features,
            'total_features': len(price_features) + len(financial_features),
            'price_data_range': (price_data.index.min(), price_data.index.max()),
            'financial_data_periods': len(financial_data) if not financial_data.empty else 0
        }


if __name__ == "__main__":
    # Example usage
    pipeline = UnifiedDataPipeline()
    
    # Test with a single ticker
    ticker = "AAPL"
    
    # Get feature info
    feature_info = pipeline.get_feature_info(ticker, "2023-01-01", "2024-01-01")
    print(f"\nFeature info for {ticker}:")
    for key, value in feature_info.items():
        print(f"{key}: {value}")
    
    # Create small dataset for testing
    try:
        dataset = pipeline.create_unified_dataset(
            tickers=[ticker], 
            start_date="2023-01-01", 
            end_date="2024-01-01",
            seq_length=30
        )
        
        print(f"\nDataset created successfully!")
        print(f"Dataset size: {len(dataset)} samples")
        
        # Show sample
        X_sample, y_sample = dataset[0]
        print(f"Sample shape: X={X_sample.shape}, y={y_sample.shape}")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")