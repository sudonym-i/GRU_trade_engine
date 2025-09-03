#!/usr/bin/env python3
"""
Interactive Brokers Trading Interface for Neural Trade Engine

This module provides a trading interface for Interactive Brokers using ib-insync.
Supports both paper trading and live trading through different port configurations.

Port configurations:
- 7497: Live trading
- 7496: Paper trading (default IB paper trading port)
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import asyncio
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util

logger = logging.getLogger(__name__)


class IBTradingInterface:
    """
    Interface for trading through Interactive Brokers API.
    """
    
    # Standard IB port configurations
    PORTS = {
        'paper': 7496,    # IB paper trading port
        'live': 7497      # IB live trading port
    }
    
    def __init__(self, mode: str = 'paper', host: str = '127.0.0.1', client_id: int = 1):
        """
        Initialize the IB trading interface.
        
        Args:
            mode: Trading mode ('paper' or 'live')
            host: IB Gateway/TWS host address
            client_id: Client ID for IB connection
        """
        self.mode = mode
        self.host = host
        self.client_id = client_id
        self.port = self.PORTS.get(mode, 7496)
        
        self.ib = IB()
        self.connected = False
        
        # Enable IB logging
        util.startLoop()
        
        logger.info(f"Initialized IB interface: mode={mode}, port={self.port}")
    
    async def connect(self) -> bool:
        """
        Connect to Interactive Brokers Gateway/TWS.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to IB {self.mode} trading at {self.host}:{self.port}")
            
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=30
            )
            
            self.connected = True
            account_summary = self.ib.accountSummary()
            logger.info(f"Connected to IB successfully. Account info available: {len(account_summary)} items")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB")
    
    def get_account_info(self) -> Dict:
        """
        Get account information and portfolio positions.
        
        Returns:
            Dictionary with account info and positions
        """
        if not self.connected:
            logger.error("Not connected to IB")
            return {}
        
        try:
            account_values = {v.tag: v.value for v in self.ib.accountValues()}
            positions = [
                {
                    'ticker': pos.contract.symbol,
                    'shares': float(pos.position),
                    'avg_price': float(pos.avgCost) / float(pos.position) if pos.position != 0 else 0,
                    'market_value': float(pos.marketValue)
                }
                for pos in self.ib.positions()
                if pos.position != 0
            ]
            
            return {
                'cash': float(account_values.get('AvailableFunds', '0')),
                'total_value': float(account_values.get('NetLiquidation', '0')),
                'positions': {pos['ticker']: pos for pos in positions},
                'last_updated': datetime.now().isoformat(),
                'mode': self.mode
            }
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def get_market_price(self, ticker: str) -> Optional[float]:
        """
        Get current market price for a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current market price or None if failed
        """
        if not self.connected:
            logger.error("Not connected to IB")
            return None
        
        try:
            stock = Stock(ticker, 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            
            market_data = self.ib.reqMktData(stock)
            self.ib.sleep(2)  # Wait for market data
            
            if market_data.last and market_data.last > 0:
                return float(market_data.last)
            elif market_data.close and market_data.close > 0:
                return float(market_data.close)
            else:
                logger.warning(f"No valid price data for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get market price for {ticker}: {e}")
            return None
    
    def execute_buy_order(self, ticker: str, shares: int, order_type: str = 'market') -> bool:
        """
        Execute a buy order.
        
        Args:
            ticker: Stock ticker symbol
            shares: Number of shares to buy
            order_type: Order type ('market' or 'limit')
            
        Returns:
            True if order submitted successfully, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to IB")
            return False
        
        try:
            stock = Stock(ticker, 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            
            if order_type == 'market':
                order = MarketOrder('BUY', shares)
            else:
                # For limit orders, you'd need to specify the limit price
                current_price = self.get_market_price(ticker)
                if not current_price:
                    logger.error(f"Cannot get price for limit order: {ticker}")
                    return False
                order = LimitOrder('BUY', shares, current_price * 1.01)  # 1% above current price
            
            trade = self.ib.placeOrder(stock, order)
            
            # Wait for order to be submitted
            self.ib.sleep(2)
            
            if trade.orderStatus.status in ['Submitted', 'PreSubmitted', 'Filled']:
                logger.info(f"Buy order submitted for {shares} shares of {ticker}")
                return True
            else:
                logger.error(f"Buy order failed for {ticker}: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute buy order for {ticker}: {e}")
            return False
    
    def execute_sell_order(self, ticker: str, shares: int, order_type: str = 'market') -> bool:
        """
        Execute a sell order.
        
        Args:
            ticker: Stock ticker symbol
            shares: Number of shares to sell
            order_type: Order type ('market' or 'limit')
            
        Returns:
            True if order submitted successfully, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to IB")
            return False
        
        try:
            stock = Stock(ticker, 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            
            if order_type == 'market':
                order = MarketOrder('SELL', shares)
            else:
                # For limit orders, you'd need to specify the limit price
                current_price = self.get_market_price(ticker)
                if not current_price:
                    logger.error(f"Cannot get price for limit order: {ticker}")
                    return False
                order = LimitOrder('SELL', shares, current_price * 0.99)  # 1% below current price
            
            trade = self.ib.placeOrder(stock, order)
            
            # Wait for order to be submitted
            self.ib.sleep(2)
            
            if trade.orderStatus.status in ['Submitted', 'PreSubmitted', 'Filled']:
                logger.info(f"Sell order submitted for {shares} shares of {ticker}")
                return True
            else:
                logger.error(f"Sell order failed for {ticker}: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute sell order for {ticker}: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class AsyncIBInterface:
    """Async wrapper for IB interface to be used with asyncio."""
    
    def __init__(self, mode: str = 'paper', host: str = '127.0.0.1', client_id: int = 1):
        self.interface = IBTradingInterface(mode, host, client_id)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.interface.connect()
        return self.interface
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.interface.disconnect()