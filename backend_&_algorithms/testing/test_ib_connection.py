#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend_&_algorithms', 'engine'))
from tsr_model.data_pipelines.stock_data_sources import fetch_ib_data
from datetime import datetime, timedelta

def test_ib_connection():
    """Test Interactive Brokers connection and data fetching."""
    
    # Test parameters
    ticker = "AAPL"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    print("Testing Interactive Brokers connection...")
    print(f"Ticker: {ticker}")
    print(f"Date range: {start_date} to {end_date}")
    print("Make sure IB Gateway/TWS is running with API enabled on port 7497")
    print("-" * 60)
    
    # Test the connection
    data = fetch_ib_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval="1 day",
        host='127.0.0.1',
        port=7497,  # Paper trading port
        client_id=1
    )
    
    if data is not None:
        print("✅ SUCCESS! Interactive Brokers connection working")
        print(f"Retrieved {len(data)} data points")
        print("\nSample data:")
        print(data.head())
        print(f"\nColumns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
    else:
        print("❌ FAILED! Check the following:")
        print("1. IB Gateway/TWS is running")
        print("2. API is enabled in settings")
        print("3. Port 7497 is configured")
        print("4. Socket clients are enabled")

if __name__ == "__main__":
    test_ib_connection()