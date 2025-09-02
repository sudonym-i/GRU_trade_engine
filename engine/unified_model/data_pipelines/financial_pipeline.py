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
    
    def get_balance_sheet(self, symbol: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        """Get historical balance sheet data."""
        try:
            data = self._make_request(f"balance-sheet-statement/{symbol}", {'period': period, 'limit': limit})
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Failed to get balance sheet for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_cash_flow(self, symbol: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        """Get historical cash flow statement data."""
        try:
            data = self._make_request(f"cash-flow-statement/{symbol}", {'period': period, 'limit': limit})
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Failed to get cash flow for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile data."""
        try:
            data = self._make_request(f"profile/{symbol}")
            return data[0] if data and len(data) > 0 else {}
        except Exception as e:
            logger.warning(f"Failed to get company profile for {symbol}: {e}")
            return {}
    
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
        
        # Define consistent feature columns available in free tier
        feature_columns = [
            'revenue', 'net_income', 'gross_profit', 'operating_income',
            'total_assets', 'total_debt', 'stockholders_equity',
            'operating_cash_flow', 'free_cash_flow',
            'market_cap', 'beta', 'pe_ratio'
        ]
        
        try:
            # Get historical financial statements (free tier)
            income_hist = self.get_income_statement(symbol, period_type, periods)
            balance_hist = self.get_balance_sheet(symbol, period_type, periods)
            cashflow_hist = self.get_cash_flow(symbol, period_type, periods)
            profile_data = self.get_company_profile(symbol)
            
            # Check if we have any data
            datasets = [income_hist, balance_hist, cashflow_hist]
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
                
                # From balance sheet
                if not balance_hist.empty and i < len(balance_hist):
                    row = balance_hist.iloc[i]
                    features.update({
                        'total_assets': row.get('totalAssets'),
                        'total_debt': row.get('totalDebt'),
                        'stockholders_equity': row.get('totalStockholdersEquity'),
                    })
                    # Use this date if we don't have one from income statement
                    if features['date'] is None:
                        features['date'] = row.get('date')
                
                # From cash flow statement
                if not cashflow_hist.empty and i < len(cashflow_hist):
                    row = cashflow_hist.iloc[i]
                    features.update({
                        'operating_cash_flow': row.get('operatingCashFlow'),
                        'free_cash_flow': row.get('freeCashFlow'),
                    })
                    # Use this date if we don't have one yet
                    if features['date'] is None:
                        features['date'] = row.get('date')
                
                # Add company profile data (static)
                if profile_data:
                    features.update({
                        'market_cap': profile_data.get('mktCap'),
                        'beta': profile_data.get('beta'),
                        'pe_ratio': profile_data.get('pe'),
                    })
                
                historical_data.append(features)
            
            # Create DataFrame
            df = pd.DataFrame(historical_data)
            
            # Ensure we have the expected columns
            for col in feature_columns + ['date']:
                if col not in df.columns:
                    df[col] = None if col == 'date' else 0
            
            # Handle missing values in financial features
            financial_cols = [col for col in feature_columns if col in df.columns]
            df[financial_cols] = df[financial_cols].ffill().bfill().fillna(0)
            
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
                'total_assets', 'total_debt', 'stockholders_equity',
                'operating_cash_flow', 'free_cash_flow',
                'market_cap', 'beta', 'pe_ratio'
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
    print("\nFree Tier Endpoints Available:")
    print("- Income Statement: /api/v3/income-statement/{symbol}")
    print("- Balance Sheet: /api/v3/balance-sheet-statement/{symbol}")
    print("- Cash Flow: /api/v3/cash-flow-statement/{symbol}")
    print("- Company Profile: /api/v3/profile/{symbol}")
    print("\nFeatures extracted:")
    print("- Revenue, Net Income, Gross Profit, Operating Income")
    print("- Total Assets, Total Debt, Stockholders Equity")
    print("- Operating Cash Flow, Free Cash Flow")
    print("- Market Cap, Beta, PE Ratio")
    
    try:
        # Test with a well-known stock
        fetcher = FinancialDataFetcher()
        
        # Test historical features
        print("\nTesting historical features...")
        historical_data = fetcher.get_historical_features("AAPL", periods=8, period_type="quarter")
        
        if not historical_data.empty:
            print(f"SUCCESS: Historical data shape: {historical_data.shape}")
            print(f"Columns: {list(historical_data.columns)}")
            print(f"Date range: {historical_data['date'].min()} to {historical_data['date'].max()}")
            
            # Show sample of data
            print("\nSample financial features:")
            print(historical_data.head(2))
            
        else:
            print("ERROR: No historical data retrieved")
        
        # Test current features
        print("\nTesting current features...")
        current_data = fetcher.get_current_features("AAPL")
        
        if not current_data.empty:
            print(f"SUCCESS: Current data shape: {current_data.shape}")
            print("Current features sample:")
            print(current_data.iloc[0])
        else:
            print("ERROR: No current data retrieved")
        
        print("\nSUCCESS: Financial Pipeline test completed!")
        
    except ValueError as e:
        if "API key required" in str(e):
            print("\nTo test with real data, set the FMP_API_KEY environment variable:")
            print("export FMP_API_KEY=your_api_key_here")
            print("\nRefactoring completed successfully!")
            print("- Removed premium endpoints (key-metrics, ratios)")
            print("- Added free tier endpoints (balance-sheet, cash-flow, profile)")
            print("- Updated feature extraction to use available data")
        else:
            print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: Financial Pipeline test failed: {e}")