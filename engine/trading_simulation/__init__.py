"""
Paper Trading Module

Simulates stock trading using real market data from FMP API.
Tracks portfolio performance, positions, and trading history.

Main Functions:
- PaperTradingEngine: Core trading simulation engine
- Portfolio: Portfolio management and tracking
- Order: Trade order management
- TradingStrategy: Base class for trading strategies

Usage:
    from engine.paper_trading import PaperTradingEngine, Portfolio
    
    # Create trading engine
    engine = PaperTradingEngine(initial_balance=100000)
    
    # Place orders
    engine.buy("AAPL", quantity=10, order_type="market")
    engine.sell("AAPL", quantity=5, order_type="limit", limit_price=185.50)
    
    # Check portfolio
    print(engine.portfolio.get_summary())
"""

from .engine import PaperTradingEngine
from .portfolio import Portfolio, Position
from .orders import Order, OrderType, OrderStatus
from .strategies import TradingStrategy, BuyAndHoldStrategy, MomentumStrategy

__all__ = [
    'PaperTradingEngine',
    'Portfolio',
    'Position', 
    'Order',
    'OrderType',
    'OrderStatus',
    'TradingStrategy',
    'BuyAndHoldStrategy',
    'MomentumStrategy'
]