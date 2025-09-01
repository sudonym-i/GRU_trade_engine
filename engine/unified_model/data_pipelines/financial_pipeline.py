"""
Financial Data Pipeline

Lightweight pipeline for fetching financial fundamentals data from FMP API.
Used by the unified model for fundamental analysis features.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.preprocessing import RobustScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataFetcher:
    """
    Fetches financial fundamentals data from Financial Modeling Prep (FMP) API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the financial data fetcher.
        
        Args:
            api_key: FMP API key. If None, uses FMP_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set FMP_API_KEY environment variable.")
        
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request to FMP."""
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_income_statement(self, symbol: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        """Get historical income statement data."""
        try:
            data = self._make_request(f"income-statement/{symbol}", {'period': period, 'limit': limit})
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Failed to get income statement for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_key_metrics(self, symbol: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        """Get historical key metrics data."""
        try:
            data = self._make_request(f"key-metrics/{symbol}", {'period': period, 'limit': limit})
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Failed to get key metrics for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_financial_ratios(self, symbol: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        """Get historical financial ratios data."""
        try:
            data = self._make_request(f"ratios/{symbol}", {'period': period, 'limit': limit})
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Failed to get financial ratios for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_features(self, symbol: str, periods: int = 20, period_type: str = "quarter", 
                               normalize: bool = True) -> pd.DataFrame:
        """
        Get historical financial features for model training.
        
        Args:
            symbol: Stock ticker symbol
            periods: Number of historical periods to fetch
            period_type: 'annual' or 'quarter'
            normalize: Whether to normalize features
            
        Returns:
            DataFrame with historical financial features
        """
        logger.info(f"Fetching {periods} {period_type} periods of financial data for {symbol}")
        
        # Define consistent feature columns that we want
        feature_columns = [
            'revenue', 'net_income', 'gross_profit', 'operating_income',
            'pe_ratio', 'roe', 'roa', 'debt_to_equity',
            'current_ratio', 'quick_ratio', 'gross_profit_margin',
            'market_cap', 'volume'
        ]
        
        try:
            # Get historical financial statements
            income_hist = self.get_income_statement(symbol, period_type, periods)
            metrics_hist = self.get_key_metrics(symbol, period_type, periods)
            ratios_hist = self.get_financial_ratios(symbol, period_type, periods)
            
            # Check if we have any data
            datasets = [income_hist, metrics_hist, ratios_hist]
            non_empty_datasets = [df for df in datasets if not df.empty]
            
            if not non_empty_datasets:
                logger.warning(f"No financial data available for {symbol}")
                return pd.DataFrame(columns=feature_columns + ['date'])
            
            # Find minimum number of periods available
            min_periods = min(len(df) for df in non_empty_datasets)
            if min_periods == 0:
                return pd.DataFrame(columns=feature_columns + ['date'])
            
            # Extract features for each period
            historical_data = []
            for i in range(min_periods):
                features = {col: None for col in feature_columns}
                
                # From income statement
                if not income_hist.empty and i < len(income_hist):
                    row = income_hist.iloc[i]
                    features.update({
                        'revenue': row.get('revenue'),
                        'net_income': row.get('netIncome'),
                        'gross_profit': row.get('grossProfit'),
                        'operating_income': row.get('operatingIncome'),
                    })
                    # Get date
                    features['date'] = row.get('date')
                
                # From key metrics
                if not metrics_hist.empty and i < len(metrics_hist):
                    row = metrics_hist.iloc[i]
                    features.update({
                        'pe_ratio': row.get('peRatio'),
                        'roe': row.get('roe'),
                        'roa': row.get('roa'),
                        'debt_to_equity': row.get('debtToEquity'),
                        'market_cap': row.get('marketCap'),
                    })
                    # Use this date if we don't have one from income statement
                    if features['date'] is None:
                        features['date'] = row.get('date')
                
                # From financial ratios
                if not ratios_hist.empty and i < len(ratios_hist):
                    row = ratios_hist.iloc[i]
                    features.update({
                        'current_ratio': row.get('currentRatio'),
                        'quick_ratio': row.get('quickRatio'),
                        'gross_profit_margin': row.get('grossProfitMargin'),
                    })
                    # Use this date if we don't have one yet
                    if features['date'] is None:
                        features['date'] = row.get('date')
                
                # Set volume to 0 if not available (placeholder)
                if features['volume'] is None:
                    features['volume'] = 0
                
                historical_data.append(features)
            
            # Create DataFrame
            df = pd.DataFrame(historical_data)
            
            # Ensure we have the expected columns
            for col in feature_columns + ['date']:
                if col not in df.columns:
                    df[col] = None if col == 'date' else 0
            
            # Handle missing values in financial features
            financial_cols = [col for col in feature_columns if col in df.columns]
            df[financial_cols] = df[financial_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            if normalize and len(df) > 1:
                # Normalize financial features
                scaler = RobustScaler()
                numeric_cols = df[financial_cols].select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            logger.info(f"Retrieved {len(df)} financial periods for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical financial data for {symbol}: {e}")
            return pd.DataFrame(columns=feature_columns + ['date'])
    
    def get_current_features(self, symbol: str, normalize: bool = True) -> pd.DataFrame:
        """
        Get current financial features for real-time prediction.
        
        Args:
            symbol: Stock ticker symbol
            normalize: Whether to normalize features
            
        Returns:
            DataFrame with current financial features (single row)
        """
        # Get latest period data (limit=1)
        historical_df = self.get_historical_features(symbol, periods=1, normalize=normalize)
        
        if historical_df.empty:
            # Return empty DataFrame with expected structure
            feature_columns = [
                'revenue', 'net_income', 'gross_profit', 'operating_income',
                'pe_ratio', 'roe', 'roa', 'debt_to_equity',
                'current_ratio', 'quick_ratio', 'gross_profit_margin',
                'market_cap', 'volume'
            ]
            return pd.DataFrame(columns=feature_columns)
        
        # Remove date column for ML features and return latest row
        ml_features = historical_df.drop(columns=['date'], errors='ignore')
        return ml_features.head(1)


def get_financial_features(symbol: str, periods: int = 20, period_type: str = "quarter",
                          normalize: bool = True) -> pd.DataFrame:
    """
    Convenience function to get financial features for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        periods: Number of periods to fetch
        period_type: 'annual' or 'quarter'  
        normalize: Whether to normalize features
        
    Returns:
        DataFrame with financial features
    """
    fetcher = FinancialDataFetcher()
    return fetcher.get_historical_features(symbol, periods, period_type, normalize)


if __name__ == "__main__":
    # Test the financial pipeline
    print("Testing Financial Pipeline...")
    
    try:
        # Test with a well-known stock
        fetcher = FinancialDataFetcher()
        
        # Test historical features
        print("Testing historical features...")
        historical_data = fetcher.get_historical_features("AAPL", periods=8, period_type="quarter")
        
        if not historical_data.empty:
            print(f"✅ Historical data shape: {historical_data.shape}")
            print(f"Columns: {list(historical_data.columns)}")
            print(f"Date range: {historical_data['date'].min()} to {historical_data['date'].max()}")
            
            # Show sample of data
            print("\nSample financial features:")
            print(historical_data.head(2))
            
        else:
            print("❌ No historical data retrieved")
        
        # Test current features
        print("\nTesting current features...")
        current_data = fetcher.get_current_features("AAPL")
        
        if not current_data.empty:
            print(f"✅ Current data shape: {current_data.shape}")
            print("Current features sample:")
            print(current_data.iloc[0])
        else:
            print("❌ No current data retrieved")
        
        print("\n✅ Financial Pipeline test completed!")
        
    except Exception as e:
        print(f"❌ Financial Pipeline test failed: {e}")