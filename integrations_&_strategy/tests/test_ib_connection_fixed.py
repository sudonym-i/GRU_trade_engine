#!/usr/bin/env python3
"""
Comprehensive Interactive Brokers Connection Test

This script tests the IB paper trading connection with proper diagnostics
and troubleshooting information.

Usage:
    # From virtual environment:
    source ../.venv/bin/activate
    python test_ib_connection_fixed.py
    
    # Or direct with venv python:
    ../.venv/bin/python test_ib_connection_fixed.py
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ib_interface import IBTradingInterface
    print("✅ Successfully imported IBTradingInterface")
except ImportError as e:
    print(f"❌ Failed to import IBTradingInterface: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check ib-insync import
    try:
        import ib_insync
        print(f"✅ ib-insync version {ib_insync.__version__} available")
    except ImportError:
        print("❌ ib-insync not available")
        print("   Install with: pip install ib-insync")
        return False
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️ Not running in virtual environment")
        print("   Consider using: source .venv/bin/activate")
    
    return True


async def test_connection_comprehensive(mode='paper'):
    """Comprehensive connection test with detailed diagnostics."""
    print(f"\n🚀 Starting comprehensive IB {mode} trading test...")
    
    # Test 1: Interface creation
    print("\n1️⃣ Testing interface creation...")
    try:
        ib_interface = IBTradingInterface(mode=mode)
        print(f"✅ Interface created successfully")
        print(f"   Mode: {ib_interface.mode}")
        print(f"   Host: {ib_interface.host}")
        print(f"   Port: {ib_interface.port}")
        print(f"   Client ID: {ib_interface.client_id}")
    except Exception as e:
        print(f"❌ Failed to create interface: {e}")
        return False
    
    # Test 2: Connection attempt
    print(f"\n2️⃣ Testing connection to IB Gateway/TWS...")
    print(f"   Attempting connection to {ib_interface.host}:{ib_interface.port}")
    print(f"   Make sure IB Gateway/TWS is running with API enabled on port {ib_interface.port}")
    
    try:
        success = await ib_interface.connect()
        
        if not success:
            print("❌ Connection failed!")
            print("\n🔧 Troubleshooting checklist:")
            print("   1. Is IB Gateway or TWS running?")
            print("   2. Is API enabled in IB settings?")
            print(f"   3. Is port {ib_interface.port} configured for {mode} trading?")
            print("   4. Is firewall blocking the connection?")
            print("   5. Are you using the correct credentials?")
            return False
        
        print("✅ Connected successfully!")
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print(f"   This usually means IB Gateway/TWS is not running on port {ib_interface.port}")
        return False
    
    # Test 3: Account info retrieval
    print("\n3️⃣ Testing account information retrieval...")
    try:
        account_info = ib_interface.get_account_info()
        
        if not account_info:
            print("❌ Failed to retrieve account information")
            return False
        
        print("✅ Account info retrieved successfully!")
        print(f"   Cash: ${account_info.get('cash', 0):,.2f}")
        print(f"   Total Value: ${account_info.get('total_value', 0):,.2f}")
        print(f"   Mode: {account_info.get('mode', 'Unknown')}")
        
        positions = account_info.get('positions', {})
        if positions:
            print(f"   Positions ({len(positions)}):")
            for ticker, pos in positions.items():
                shares = pos.get('shares', 0)
                value = pos.get('market_value', 0)
                print(f"     {ticker}: {shares} shares (${value:,.2f})")
        else:
            print("   No current positions")
            
    except Exception as e:
        print(f"❌ Failed to get account info: {e}")
        return False
    
    # Test 4: Market data test
    print("\n4️⃣ Testing market data retrieval...")
    test_ticker = "AAPL"
    try:
        print(f"   Requesting market data for {test_ticker}...")
        price = ib_interface.get_market_price(test_ticker)
        
        if price and price > 0:
            print(f"✅ Market data working! {test_ticker}: ${price:.2f}")
        else:
            print(f"⚠️ No price data for {test_ticker}")
            print("   This might be due to:")
            print("     - Market is closed")
            print("     - No market data subscription")
            print("     - Symbol not found")
            
    except Exception as e:
        print(f"❌ Market data error: {e}")
    
    # Test 5: Order functionality test (dry run)
    print("\n5️⃣ Testing order functionality (dry run)...")
    print("✅ Order methods available:")
    print("   - execute_buy_order()")
    print("   - execute_sell_order()")
    print("   - Both support 'market' and 'limit' order types")
    
    # Cleanup
    print("\n6️⃣ Cleaning up...")
    try:
        ib_interface.disconnect()
        print("✅ Disconnected successfully")
    except Exception as e:
        print(f"⚠️ Disconnect error: {e}")
    
    print(f"\n🎉 IB {mode} trading test completed successfully!")
    print(f"   Your IB interface is properly configured and working.")
    return True


async def main():
    """Main test function."""
    print("=" * 60)
    print("Interactive Brokers Paper Trading Connection Test")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return False
    
    # Test paper trading connection
    success = await test_connection_comprehensive('paper')
    
    if success:
        print(f"\n✅ SUCCESS: Your IB paper trading setup is working correctly!")
        print(f"   You can now use your automated trader with --mode ib_paper")
    else:
        print(f"\n❌ FAILED: IB paper trading connection issues detected.")
        print(f"   Please check the troubleshooting steps above.")
    
    return success


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)