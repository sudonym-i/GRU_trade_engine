#!/usr/bin/env python3
"""
Yahoo Finance Financial Data Pipeline

Alternative financial data pipeline using Yahoo Finance (yfinance) instead of 
the restricted FMP API. Provides comprehensive financial fundamentals including:
- Key financial ratios and metrics
- Income statements (annual & quarterly)
- Balance sheets
- Cash flow statements
- Earnings data

This replaces the broken financial_pipeline.py that relied on FMP API.

Usage:
    from yahoo_financial_pipeline import YahooFinancialFetcher
    fetcher = YahooFinancialFetcher()
    features = fetcher.get_financial_features('NVDA')
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from sklearn.preprocessing import RobustScaler
import logging
from datetime import datetime, timedelta

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


class YahooFinancialFetcher:
    """
    Fetches financial fundamentals data from Yahoo Finance using yfinance.
    Free alternative to FMP API with comprehensive financial data.
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance financial data fetcher."""
        if not install_yfinance():
            raise ImportError("Could not install/import yfinance")
        
        import yfinance as yf
        self.yf = yf
        logger.info("Yahoo Financial Fetcher initialized")
    
    def get_key_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Get key financial metrics from Yahoo Finance info.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary of key financial metrics
        """
        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info
            
            # Key financial metrics to extract
            metrics = {
                # Valuation metrics
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'trailing_pe': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'ev_to_revenue': info.get('enterpriseToRevenue', 0),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
                
                # Profitability metrics
                'profit_margins': info.get('profitMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                
                # Growth metrics
                'earnings_growth': info.get('earningsGrowth', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
                
                # Financial health
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                
                # Operational metrics
                'total_revenue': info.get('totalRevenue', 0),
                'revenue_per_share': info.get('revenuePerShare', 0),
                'book_value': info.get('bookValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                
                # Other useful metrics
                'beta': info.get('beta', 1.0),
                'dividend_rate': info.get('dividendRate', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0)
            }
            
            # Convert None values to 0 and handle inf values
            for key, value in metrics.items():
                if value is None or np.isinf(value) or np.isnan(value):
                    metrics[key] = 0.0
                else:
                    metrics[key] = float(value)
            
            logger.info(f"Retrieved {len(metrics)} key metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching key metrics for {symbol}: {e}")
            return {}
    
    def get_income_statement_metrics(self, symbol: str, quarterly: bool = False) -> Dict[str, float]:
        """
        Extract key metrics from income statement.
        
        Args:
            symbol: Stock ticker symbol
            quarterly: If True, use quarterly data; otherwise annual
            
        Returns:
            Dictionary of income statement metrics
        """
        try:
            ticker = self.yf.Ticker(symbol)
            
            if quarterly:
                financials = ticker.quarterly_financials
            else:
                financials = ticker.financials
            
            if financials.empty:
                logger.warning(f"No income statement data for {symbol}")
                return {}
            
            # Get most recent column (latest data)
            latest_data = financials.iloc[:, 0]
            
            # Key income statement items
            metrics = {
                'total_revenue': self._safe_get(latest_data, 'Total Revenue', 0),
                'gross_profit': self._safe_get(latest_data, 'Gross Profit', 0),
                'operating_income': self._safe_get(latest_data, 'Operating Income', 0),
                'ebitda': self._safe_get(latest_data, 'EBITDA', 0),
                'ebit': self._safe_get(latest_data, 'EBIT', 0),
                'net_income': self._safe_get(latest_data, 'Net Income', 0),
                'basic_eps': self._safe_get(latest_data, 'Basic EPS', 0),
                'diluted_eps': self._safe_get(latest_data, 'Diluted EPS', 0),
                'cost_of_revenue': self._safe_get(latest_data, 'Cost Of Revenue', 0),
                'research_development': self._safe_get(latest_data, 'Research And Development', 0),
                'selling_admin': self._safe_get(latest_data, 'Selling General Administrative', 0),
                'interest_expense': self._safe_get(latest_data, 'Interest Expense', 0),
                'tax_provision': self._safe_get(latest_data, 'Tax Provision', 0)
            }
            
            # Calculate derived metrics
            if metrics['total_revenue'] > 0:
                metrics['gross_margin'] = metrics['gross_profit'] / metrics['total_revenue']
                metrics['operating_margin'] = metrics['operating_income'] / metrics['total_revenue']
                metrics['net_margin'] = metrics['net_income'] / metrics['total_revenue']
            else:
                metrics['gross_margin'] = 0
                metrics['operating_margin'] = 0
                metrics['net_margin'] = 0
            
            logger.info(f"Retrieved income statement metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {e}")
            return {}
    
    def get_balance_sheet_metrics(self, symbol: str, quarterly: bool = False) -> Dict[str, float]:
        """
        Extract key metrics from balance sheet.
        
        Args:
            symbol: Stock ticker symbol
            quarterly: If True, use quarterly data; otherwise annual
            
        Returns:
            Dictionary of balance sheet metrics
        """
        try:
            ticker = self.yf.Ticker(symbol)
            
            if quarterly:
                balance_sheet = ticker.quarterly_balance_sheet
            else:
                balance_sheet = ticker.balance_sheet
            
            if balance_sheet.empty:
                logger.warning(f"No balance sheet data for {symbol}")
                return {}
            
            # Get most recent column
            latest_data = balance_sheet.iloc[:, 0]
            
            metrics = {
                'total_assets': self._safe_get(latest_data, 'Total Assets', 0),
                'total_liabilities': self._safe_get(latest_data, 'Total Liabilities Net Minority Interest', 0),
                'total_equity': self._safe_get(latest_data, 'Total Equity Gross Minority Interest', 0),
                'current_assets': self._safe_get(latest_data, 'Current Assets', 0),
                'current_liabilities': self._safe_get(latest_data, 'Current Liabilities', 0),
                'cash_equivalents': self._safe_get(latest_data, 'Cash Cash Equivalents And Short Term Investments', 0),
                'inventory': self._safe_get(latest_data, 'Inventory', 0),
                'total_debt': self._safe_get(latest_data, 'Total Debt', 0),
                'long_term_debt': self._safe_get(latest_data, 'Long Term Debt And Capital Lease Obligation', 0),
                'retained_earnings': self._safe_get(latest_data, 'Retained Earnings', 0),
                'working_capital': self._safe_get(latest_data, 'Working Capital', 0)
            }
            
            # Calculate financial ratios
            if metrics['current_liabilities'] > 0:
                metrics['current_ratio'] = metrics['current_assets'] / metrics['current_liabilities']
            else:
                metrics['current_ratio'] = 0
            
            if metrics['total_equity'] > 0:
                metrics['debt_to_equity'] = metrics['total_debt'] / metrics['total_equity']
            else:
                metrics['debt_to_equity'] = 0
            
            if metrics['total_assets'] > 0:
                metrics['asset_turnover'] = metrics['total_assets']  # Will be calculated with revenue later
            else:
                metrics['asset_turnover'] = 0
            
            logger.info(f"Retrieved balance sheet metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return {}
    
    def get_cash_flow_metrics(self, symbol: str, quarterly: bool = False) -> Dict[str, float]:
        """
        Extract key metrics from cash flow statement.
        
        Args:
            symbol: Stock ticker symbol  
            quarterly: If True, use quarterly data; otherwise annual
            
        Returns:
            Dictionary of cash flow metrics
        """
        try:
            ticker = self.yf.Ticker(symbol)
            
            if quarterly:
                cash_flow = ticker.quarterly_cashflow
            else:
                cash_flow = ticker.cashflow
            
            if cash_flow.empty:
                logger.warning(f"No cash flow data for {symbol}")
                return {}
            
            # Get most recent column
            latest_data = cash_flow.iloc[:, 0]
            
            metrics = {
                'operating_cash_flow': self._safe_get(latest_data, 'Operating Cash Flow', 0),
                'investing_cash_flow': self._safe_get(latest_data, 'Investing Cash Flow', 0),
                'financing_cash_flow': self._safe_get(latest_data, 'Financing Cash Flow', 0),
                'free_cash_flow': self._safe_get(latest_data, 'Free Cash Flow', 0),
                'capital_expenditure': abs(self._safe_get(latest_data, 'Capital Expenditure', 0)),  # Usually negative
                'dividends_paid': abs(self._safe_get(latest_data, 'Cash Dividends Paid', 0)),  # Usually negative
                'stock_repurchase': abs(self._safe_get(latest_data, 'Repurchase Of Capital Stock', 0))  # Usually negative
            }
            
            logger.info(f"Retrieved cash flow metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching cash flow for {symbol}: {e}")
            return {}
    
    def _safe_get(self, data_series: pd.Series, key: str, default: float = 0.0) -> float:
        """
        Safely get a value from pandas Series, handling missing keys and NaN values.
        """
        try:
            # Try exact match first
            if key in data_series.index:
                value = data_series[key]
            else:
                # Try partial matching (case insensitive)
                matching_keys = [k for k in data_series.index if key.lower() in k.lower()]
                if matching_keys:
                    value = data_series[matching_keys[0]]
                else:
                    return default
            
            # Handle NaN, None, inf values
            if pd.isna(value) or np.isinf(value):
                return default
            
            return float(value)
            
        except (KeyError, TypeError, ValueError):
            return default
    
    def get_comprehensive_features(self, symbol: str, quarterly: bool = False) -> Dict[str, float]:
        """
        Get comprehensive financial features combining all data sources.
        
        Args:
            symbol: Stock ticker symbol
            quarterly: Whether to use quarterly data for statements
            
        Returns:
            Dictionary with all financial features
        """
        logger.info(f"Fetching comprehensive financial features for {symbol}")
        
        # Combine all feature sets
        features = {}
        
        # Key metrics from info
        features.update(self.get_key_metrics(symbol))
        
        # Income statement metrics
        features.update(self.get_income_statement_metrics(symbol, quarterly))
        
        # Balance sheet metrics
        features.update(self.get_balance_sheet_metrics(symbol, quarterly))
        
        # Cash flow metrics
        features.update(self.get_cash_flow_metrics(symbol, quarterly))
        
        logger.info(f"Retrieved {len(features)} total financial features for {symbol}")
        return features


def get_financial_features(symbol: str, quarterly: bool = False, 
                          normalize: bool = True) -> pd.DataFrame:
    """
    Get financial features for a symbol in a format compatible with the existing pipeline.
    
    Args:
        symbol: Stock ticker symbol
        quarterly: Whether to use quarterly financial data
        normalize: Whether to normalize the features
        
    Returns:
        DataFrame with financial features (single row)
    """
    fetcher = YahooFinancialFetcher()
    features = fetcher.get_comprehensive_features(symbol, quarterly)
    
    if not features:
        logger.error(f"No financial features found for {symbol}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    df.index = [symbol]
    
    # Handle inf and nan values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    if normalize:
        # Use robust scaling to handle outliers
        scaler = RobustScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns
        )
        return df_scaled
    
    return df


# Maintain compatibility with existing financial_pipeline.py interface
class FinancialDataFetcher(YahooFinancialFetcher):
    """
    Compatibility class that maintains the same interface as the original
    FinancialDataFetcher from financial_pipeline.py but uses Yahoo Finance.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize - api_key is ignored since we're using Yahoo Finance."""
        super().__init__()
        if api_key:
            logger.info("API key parameter ignored - using free Yahoo Finance")


def main():
    """Test the Yahoo Financial pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Yahoo Finance financial data')
    parser.add_argument('--ticker', default='NVDA', help='Stock ticker symbol')
    parser.add_argument('--quarterly', action='store_true', help='Use quarterly data')
    
    args = parser.parse_args()
    
    logger.info(f"=== Testing Yahoo Financial Pipeline ===")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Quarterly: {args.quarterly}")
    
    try:
        # Test comprehensive features
        features = get_financial_features(args.ticker, args.quarterly, normalize=False)
        
        if not features.empty:
            logger.info("=== SUCCESS ===")
            logger.info(f"Retrieved {features.shape[1]} financial features")
            logger.info("Sample features:")
            print(features.head().T)  # Transpose to show features as rows
            
            # Test normalization
            features_norm = get_financial_features(args.ticker, args.quarterly, normalize=True)
            logger.info(f"Normalized features shape: {features_norm.shape}")
            
            return 0
        else:
            logger.error("No features retrieved")
            return 1
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)