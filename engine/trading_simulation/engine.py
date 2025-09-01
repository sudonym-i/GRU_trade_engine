"""
Paper Trading Engine

Main trading engine that combines portfolio management, order execution,
and real-time market data from FMP API.
"""

import os
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .portfolio import Portfolio, Position
from .orders import Order, OrderType, OrderSide, OrderStatus, OrderManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataProvider:
    """
    Provides real-time and historical market data from FMP API.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self._cache = {}
        self._cache_ttl = 60  # 1 minute cache
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        cache_key = f"price_{symbol}"
        now = time.time()
        
        # Check cache
        if cache_key in self._cache:
            price, timestamp = self._cache[cache_key]
            if now - timestamp < self._cache_ttl:
                return price
        
        try:
            url = f"{self.base_url}/quote/{symbol}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                price = data[0].get('price')
                if price:
                    # Cache the result
                    self._cache[cache_key] = (price, now)
                    return price
        
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
        
        return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols."""
        prices = {}
        
        try:
            # FMP supports batch quotes
            symbols_str = ",".join(symbols)
            url = f"{self.base_url}/quote/{symbols_str}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for quote in data:
                symbol = quote.get('symbol')
                price = quote.get('price')
                if symbol and price:
                    prices[symbol] = price
        
        except Exception as e:
            logger.error(f"Error fetching batch prices: {e}")
            # Fallback to individual requests
            for symbol in symbols:
                price = self.get_current_price(symbol)
                if price:
                    prices[symbol] = price
        
        return prices
    
    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open."""
        try:
            url = f"{self.base_url}/is-the-market-open"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('isTheStockMarketOpen', False)
        
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            # Fallback: simple time-based check (US Eastern Time)
            from datetime import datetime, timezone, timedelta
            et = timezone(timedelta(hours=-5))  # EST (simplified)
            now_et = datetime.now(et)
            
            # Market is open Monday-Friday, 9:30 AM - 4:00 PM ET
            if now_et.weekday() >= 5:  # Weekend
                return False
            
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now_et <= market_close


class PaperTradingEngine:
    """
    Main paper trading engine that simulates stock trading.
    """
    
    def __init__(self, initial_balance: float = 100000.0, api_key: str = None):
        """
        Initialize the paper trading engine.
        
        Args:
            initial_balance: Starting cash balance
            api_key: FMP API key (if None, uses FMP_API_KEY env variable)
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set FMP_API_KEY environment variable.")
        
        self.portfolio = Portfolio(initial_balance)
        self.order_manager = OrderManager()
        self.market_data = MarketDataProvider(self.api_key)
        
        # Trading settings
        self.trading_fees = 0.0  # Commission per trade
        self.auto_execute = True  # Auto-execute market orders
        self.market_data_refresh_interval = 60  # Seconds
        
        # State tracking
        self.is_running = False
        self.last_market_update = datetime.min
        
        logger.info(f"Paper trading engine initialized with ${initial_balance:,.2f}")
    
    def buy(self, symbol: str, quantity: int, order_type: str = "market", 
            limit_price: float = None, stop_price: float = None) -> str:
        """
        Place a buy order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            order_type: "market", "limit", "stop", or "stop_limit"
            limit_price: Limit price (for limit and stop_limit orders)
            stop_price: Stop price (for stop and stop_limit orders)
            
        Returns:
            Order ID
        """
        return self._place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType(order_type.lower()),
            limit_price=limit_price,
            stop_price=stop_price
        )
    
    def sell(self, symbol: str, quantity: int, order_type: str = "market",
             limit_price: float = None, stop_price: float = None) -> str:
        """
        Place a sell order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            order_type: "market", "limit", "stop", or "stop_limit"
            limit_price: Limit price (for limit and stop_limit orders)
            stop_price: Stop price (for stop and stop_limit orders)
            
        Returns:
            Order ID
        """
        return self._place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType(order_type.lower()),
            limit_price=limit_price,
            stop_price=stop_price
        )
    
    def _place_order(self, symbol: str, side: OrderSide, quantity: int,
                    order_type: OrderType, limit_price: float = None,
                    stop_price: float = None) -> str:
        """Internal method to place an order."""
        
        # Validate order parameters
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not limit_price:
            raise ValueError("Limit price required for limit orders")
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not stop_price:
            raise ValueError("Stop price required for stop orders")
        
        # Check if we have enough shares for sell orders
        if side == OrderSide.SELL:
            position = self.portfolio.get_position(symbol)
            if not position or position.quantity < quantity:
                available = position.quantity if position else 0
                raise ValueError(f"Insufficient shares. Have {available}, need {quantity}")
        
        # Check if we have enough buying power for buy orders
        if side == OrderSide.BUY and order_type == OrderType.MARKET:
            current_price = self.market_data.get_current_price(symbol)
            if current_price:
                estimated_cost = quantity * current_price + self.trading_fees
                if estimated_cost > self.portfolio.buying_power:
                    raise ValueError(f"Insufficient buying power. Need ${estimated_cost:,.2f}, have ${self.portfolio.buying_power:,.2f}")
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        
        # Add to order manager
        order_id = self.order_manager.add_order(order)
        
        logger.info(f"Placed order: {order}")
        
        # Auto-execute market orders if enabled
        if self.auto_execute and order_type == OrderType.MARKET:
            self._try_execute_market_order(order)
        
        return order_id
    
    def _try_execute_market_order(self, order: Order):
        """Try to execute a market order immediately."""
        current_price = self.market_data.get_current_price(order.symbol)
        
        if current_price:
            if order.side == OrderSide.BUY:
                success = self.portfolio.execute_buy(
                    order.symbol, order.quantity, current_price, self.trading_fees
                )
                if success:
                    order.execute(current_price)
                    logger.info(f"Executed market buy: {order.quantity} {order.symbol} @ ${current_price:.2f}")
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Market buy rejected: insufficient funds")
            
            else:  # SELL
                success = self.portfolio.execute_sell(
                    order.symbol, order.quantity, current_price, self.trading_fees
                )
                if success:
                    order.execute(current_price)
                    logger.info(f"Executed market sell: {order.quantity} {order.symbol} @ ${current_price:.2f}")
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Market sell rejected: insufficient shares")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        success = self.order_manager.cancel_order(order_id)
        if success:
            logger.info(f"Cancelled order: {order_id}")
        return success
    
    def update_market_data(self):
        """Update market data for all positions and process pending orders."""
        # Get all symbols we need data for
        symbols = set()
        
        # Add symbols from positions
        for symbol in self.portfolio.positions.keys():
            if self.portfolio.positions[symbol].quantity > 0:
                symbols.add(symbol)
        
        # Add symbols from active orders
        for order in self.order_manager.get_active_orders():
            symbols.add(order.symbol)
        
        if not symbols:
            return
        
        # Fetch current prices
        market_data = self.market_data.get_multiple_prices(list(symbols))
        
        # Update portfolio positions
        self.portfolio.update_market_data(market_data)
        
        # Process pending orders
        executed_orders = []
        for symbol, price in market_data.items():
            executed = self.order_manager.process_market_data(symbol, price)
            for order, filled_qty in executed:
                if order.side == OrderSide.BUY:
                    self.portfolio.execute_buy(symbol, filled_qty, price, self.trading_fees)
                else:
                    self.portfolio.execute_sell(symbol, filled_qty, price, self.trading_fees)
                
                executed_orders.append((order, filled_qty))
        
        # Log executed orders
        for order, filled_qty in executed_orders:
            logger.info(f"Executed {order.order_type.value} {order.side.value}: "
                       f"{filled_qty} {order.symbol} @ ${order.filled_price:.2f}")
        
        self.last_market_update = datetime.now()
    
    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary."""
        return self.portfolio.get_summary()
    
    def get_positions(self) -> List[dict]:
        """Get all positions."""
        return self.portfolio.get_positions()
    
    def get_orders(self, status: str = None) -> List[dict]:
        """
        Get orders.
        
        Args:
            status: Filter by status ("active", "filled", "cancelled")
        """
        if status == "active":
            orders = self.order_manager.get_active_orders()
        else:
            orders = self.order_manager.get_order_history()
            if status:
                orders = [o for o in orders if o.status.value == status]
        
        return [order.to_dict() for order in orders]
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[dict]:
        """Get trade history."""
        return self.portfolio.get_trade_history(symbol, limit)
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics."""
        return self.portfolio.get_performance_metrics()
    
    def is_market_open(self) -> bool:
        """Check if market is open."""
        return self.market_data.is_market_open()
    
    def start_live_trading(self):
        """Start live trading mode with periodic market updates."""
        self.is_running = True
        logger.info("Started live trading mode")
    
    def stop_live_trading(self):
        """Stop live trading mode."""
        self.is_running = False
        logger.info("Stopped live trading mode")
    
    def save_state(self, filepath: str):
        """Save engine state to file."""
        self.portfolio.save_to_file(filepath)
        logger.info(f"Saved portfolio state to {filepath}")
    
    def export_trades(self, filepath: str):
        """Export trade history to CSV."""
        self.portfolio.export_to_csv(filepath)
        logger.info(f"Exported trades to {filepath}")


if __name__ == "__main__":
    # Test the trading engine
    print("Testing Paper Trading Engine...")
    
    # Check for API key
    if not os.getenv('FMP_API_KEY'):
        print("❌ FMP_API_KEY not set. Please set your API key:")
        print("export FMP_API_KEY=your_api_key_here")
        exit(1)
    
    try:
        # Create engine
        engine = PaperTradingEngine(initial_balance=100000)
        
        print(f"Initial balance: ${engine.get_portfolio_summary()['total_equity']:,.2f}")
        print(f"Market is {'open' if engine.is_market_open() else 'closed'}")
        
        # Place some orders
        print("\nPlacing orders...")
        
        buy_order_id = engine.buy("AAPL", 10, "market")
        print(f"Placed buy order: {buy_order_id}")
        
        limit_sell_id = engine.sell("AAPL", 5, "limit", limit_price=200.0)
        print(f"Placed limit sell order: {limit_sell_id}")
        
        # Update market data
        print("\nUpdating market data...")
        engine.update_market_data()
        
        # Show results
        summary = engine.get_portfolio_summary()
        print(f"Total equity: ${summary['total_equity']:,.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
        
        positions = engine.get_positions()
        print(f"\nPositions ({len(positions)}):")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['quantity']} shares, "
                  f"P&L: ${pos['unrealized_pnl']:,.2f}")
        
        active_orders = engine.get_orders("active")
        print(f"\nActive orders ({len(active_orders)}):")
        for order in active_orders:
            print(f"  {order['order_id']}: {order['side']} {order['quantity']} "
                  f"{order['symbol']} @ {order['order_type']}")
        
        print("✅ Paper trading engine test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")