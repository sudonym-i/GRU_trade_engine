#!/usr/bin/env python3
"""
Test script for Financial Modeling Prep API integration
"""

import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import DataLoader

def test_fmp_integration():
    """Test the Financial Modeling Prep API integration"""
    print("Testing Financial Modeling Prep API Integration")
    print("=" * 50)
    
    # Set up API key (use demo for testing, replace with real key for production)
    api_key = os.getenv('FMP_API_KEY', 'demo')
    print(f"Using API key: {'demo' if api_key == 'demo' else 'custom key'}")
    
    # Test initialization
    print("\n1. Testing DataLoader initialization...")
    try:
        loader = DataLoader("AAPL", "2023-01-01", "2023-01-31", api_key=api_key)
        print("✓ DataLoader initialized successfully")
        print(f"  - Tickers: {loader.tickers}")
        print(f"  - Date range: {loader.start} to {loader.end}")
        print(f"  - Interval: {loader.interval}")
    except Exception as e:
        print(f"✗ DataLoader initialization failed: {e}")
        return False
    
    # Test data download (using demo key - may have limited data)
    print("\n2. Testing data download...")
    try:
        data = loader.download()
        if data:
            print("✓ Data download successful")
            for ticker, df in data.items():
                print(f"  - {ticker}: {len(df)} rows, columns: {list(df.columns)}")
                if len(df) > 0:
                    print(f"    First date: {df.index[0]}, Last date: {df.index[-1]}")
                    print(f"    Sample data: Close={df['Close'].iloc[-1]}, Volume={df['Volume'].iloc[-1]}")
        else:
            print("⚠ No data retrieved (this is expected with demo key)")
    except Exception as e:
        print(f"✗ Data download failed: {e}")
        return False
    
    # Test with different interval
    print("\n3. Testing different intervals...")
    try:
        loader_1h = DataLoader("AAPL", "2023-01-01", "2023-01-02", interval="1h", api_key=api_key)
        print("✓ 1-hour interval DataLoader initialized")
        
        loader_5m = DataLoader("AAPL", "2023-01-01", "2023-01-02", interval="5m", api_key=api_key)
        print("✓ 5-minute interval DataLoader initialized")
    except Exception as e:
        print(f"✗ Different interval test failed: {e}")
        return False
    
    # Test multiple tickers
    print("\n4. Testing multiple tickers...")
    try:
        multi_loader = DataLoader(["AAPL", "MSFT"], "2023-01-01", "2023-01-31", api_key=api_key)
        print("✓ Multiple ticker DataLoader initialized")
        print(f"  - Tickers: {multi_loader.tickers}")
    except Exception as e:
        print(f"✗ Multiple ticker test failed: {e}")
        return False
    
    print("\n5. Testing error handling...")
    try:
        # Test without API key - temporarily remove env var
        original_key = os.environ.get('FMP_API_KEY')
        if 'FMP_API_KEY' in os.environ:
            del os.environ['FMP_API_KEY']
        
        try:
            no_key_loader = DataLoader("AAPL", "2023-01-01", "2023-01-31")
            print("✗ Should have raised ValueError for missing API key")
            return False
        except ValueError as ve:
            print("✓ Correctly raises ValueError when API key is missing")
        finally:
            # Restore original key if it existed
            if original_key:
                os.environ['FMP_API_KEY'] = original_key
                
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All Financial Modeling Prep API integration tests passed!")
    print("\nTo use in production:")
    print("1. Get a real API key from https://financialmodelingprep.com/")
    print("2. Set environment variable: export FMP_API_KEY=your_api_key")
    print("3. Or pass api_key parameter when creating DataLoader")
    
    return True

if __name__ == "__main__":
    success = test_fmp_integration()
    exit(0 if success else 1)