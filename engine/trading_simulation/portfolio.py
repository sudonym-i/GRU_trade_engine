"""
Portfolio Management for Paper Trading

Tracks positions, cash balance, and portfolio performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import json


@dataclass
class Position:
    """
    Represents a stock position in the portfolio.
    """
    symbol: str
    quantity: int = 0
    average_cost: float = 0.0
    market_value: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def total_cost(self) -> float:
        """Total cost basis of position."""
        return self.quantity * self.average_cost
    
    @property
    def pnl_percent(self) -> float:
        """Unrealized P&L as percentage."""
        if self.total_cost == 0:
            return 0.0
        return (self.unrealized_pnl / self.total_cost) * 100
    
    def update_market_data(self, current_price: float):
        """Update position with current market price."""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = self.market_value - self.total_cost
        self.last_updated = datetime.now()
    
    def add_shares(self, quantity: int, price: float):
        """
        Add shares to position (buy).
        
        Args:
            quantity: Number of shares to add
            price: Price per share
        """
        if self.quantity == 0:
            # New position
            self.quantity = quantity
            self.average_cost = price
        else:
            # Add to existing position
            total_cost = self.total_cost + (quantity * price)
            self.quantity += quantity
            self.average_cost = total_cost / self.quantity
        
        self.update_market_data(price)
    
    def remove_shares(self, quantity: int, price: float) -> float:
        """
        Remove shares from position (sell).
        
        Args:
            quantity: Number of shares to remove
            price: Price per share
            
        Returns:
            Realized P&L from the sale
        """
        if quantity > self.quantity:
            raise ValueError(f"Cannot sell {quantity} shares, only have {self.quantity}")
        
        # Calculate realized P&L
        realized_pnl = (price - self.average_cost) * quantity
        self.realized_pnl += realized_pnl
        
        # Update position
        self.quantity -= quantity
        
        if self.quantity == 0:
            # Position closed
            self.average_cost = 0.0
            self.market_value = 0.0
            self.unrealized_pnl = 0.0
        else:
            # Partial sale - average cost stays the same
            self.update_market_data(price)
        
        return realized_pnl
    
    def to_dict(self) -> dict:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_cost': round(self.average_cost, 2),
            'market_value': round(self.market_value, 2),
            'current_price': round(self.current_price, 2),
            'unrealized_pnl': round(self.unrealized_pnl, 2),
            'realized_pnl': round(self.realized_pnl, 2),
            'pnl_percent': round(self.pnl_percent, 2),
            'last_updated': self.last_updated.isoformat()
        }


class Portfolio:
    """
    Manages a portfolio of stock positions and cash balance.
    """
    
    def __init__(self, initial_balance: float = 100000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_balance: Starting cash balance
        """
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        self.created_at = datetime.now()
        
        # Performance tracking
        self.total_deposits = initial_balance
        self.total_withdrawals = 0.0
        self.total_fees = 0.0
    
    @property
    def total_market_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_equity(self) -> float:
        """Total portfolio equity (cash + positions)."""
        return self.cash_balance + self.total_market_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """Total realized P&L from all trades."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def total_return(self) -> float:
        """Total return (realized + unrealized P&L)."""
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    @property
    def total_return_percent(self) -> float:
        """Total return as percentage of initial balance."""
        return (self.total_return / self.initial_balance) * 100
    
    @property
    def buying_power(self) -> float:
        """Available buying power (currently just cash balance)."""
        return self.cash_balance
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has position in symbol."""
        pos = self.positions.get(symbol)
        return pos is not None and pos.quantity > 0
    
    def execute_buy(self, symbol: str, quantity: int, price: float, fees: float = 0.0) -> bool:
        """
        Execute a buy order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            fees: Trading fees
            
        Returns:
            True if trade was executed
        """
        total_cost = (quantity * price) + fees
        
        # Check if we have enough cash
        if total_cost > self.cash_balance:
            return False
        
        # Deduct cash
        self.cash_balance -= total_cost
        self.total_fees += fees
        
        # Update or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        self.positions[symbol].add_shares(quantity, price)
        
        # Record trade
        self._record_trade("BUY", symbol, quantity, price, fees)
        
        return True
    
    def execute_sell(self, symbol: str, quantity: int, price: float, fees: float = 0.0) -> bool:
        """
        Execute a sell order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            fees: Trading fees
            
        Returns:
            True if trade was executed
        """
        # Check if we have enough shares
        position = self.positions.get(symbol)
        if not position or position.quantity < quantity:
            return False
        
        # Calculate proceeds
        proceeds = (quantity * price) - fees
        
        # Add cash
        self.cash_balance += proceeds
        self.total_fees += fees
        
        # Update position
        realized_pnl = position.remove_shares(quantity, price)
        
        # Remove position if quantity is zero
        if position.quantity == 0:
            del self.positions[symbol]
        
        # Record trade
        self._record_trade("SELL", symbol, quantity, price, fees, realized_pnl)
        
        return True
    
    def update_market_data(self, market_data: Dict[str, float]):
        """
        Update all positions with current market prices.
        
        Args:
            market_data: Dictionary of symbol -> current_price
        """
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.update_market_data(market_data[symbol])
    
    def _record_trade(self, action: str, symbol: str, quantity: int, price: float, 
                     fees: float = 0.0, realized_pnl: float = 0.0):
        """Record a trade in history."""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': round(price, 2),
            'fees': round(fees, 2),
            'realized_pnl': round(realized_pnl, 2),
            'total_value': round(quantity * price, 2)
        }
        self.trade_history.append(trade)
    
    def get_summary(self) -> dict:
        """Get portfolio summary."""
        return {
            'cash_balance': round(self.cash_balance, 2),
            'total_market_value': round(self.total_market_value, 2),
            'total_equity': round(self.total_equity, 2),
            'total_return': round(self.total_return, 2),
            'total_return_percent': round(self.total_return_percent, 2),
            'unrealized_pnl': round(self.total_unrealized_pnl, 2),
            'realized_pnl': round(self.total_realized_pnl, 2),
            'total_fees': round(self.total_fees, 2),
            'positions_count': len([p for p in self.positions.values() if p.quantity > 0]),
            'created_at': self.created_at.isoformat()
        }
    
    def get_positions(self) -> List[dict]:
        """Get all positions with details."""
        return [pos.to_dict() for pos in self.positions.values() if pos.quantity > 0]
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[dict]:
        """
        Get trade history.
        
        Args:
            symbol: Filter by symbol (if None, returns all trades)
            limit: Maximum number of trades to return
            
        Returns:
            List of trades sorted by timestamp (newest first)
        """
        trades = self.trade_history.copy()
        
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        
        # Sort by timestamp (newest first)
        trades.sort(key=lambda t: t['timestamp'], reverse=True)
        
        return trades[:limit]
    
    def get_performance_metrics(self) -> dict:
        """Get detailed performance metrics."""
        # Calculate time-based metrics
        days_active = (datetime.now() - self.created_at).days
        if days_active == 0:
            days_active = 1  # Avoid division by zero
        
        # Calculate win/loss ratio
        profitable_trades = [t for t in self.trade_history 
                           if t['action'] == 'SELL' and t['realized_pnl'] > 0]
        losing_trades = [t for t in self.trade_history 
                        if t['action'] == 'SELL' and t['realized_pnl'] < 0]
        
        total_trades = len([t for t in self.trade_history if t['action'] == 'SELL'])
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': round(self.total_return, 2),
            'total_return_percent': round(self.total_return_percent, 2),
            'annualized_return': round((self.total_return_percent / days_active) * 365, 2),
            'days_active': days_active,
            'total_trades': total_trades,
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate * 100, 2),
            'average_win': round(sum(t['realized_pnl'] for t in profitable_trades) / len(profitable_trades), 2) if profitable_trades else 0,
            'average_loss': round(sum(t['realized_pnl'] for t in losing_trades) / len(losing_trades), 2) if losing_trades else 0,
            'total_fees': round(self.total_fees, 2)
        }
    
    def save_to_file(self, filepath: str):
        """Save portfolio to JSON file."""
        data = {
            'portfolio_summary': self.get_summary(),
            'positions': self.get_positions(),
            'trade_history': self.trade_history,
            'performance_metrics': self.get_performance_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_to_csv(self, filepath: str):
        """Export trade history to CSV."""
        import pandas as pd
        
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            df.to_csv(filepath, index=False)


if __name__ == "__main__":
    # Test the portfolio system
    print("Testing Portfolio System...")
    
    # Create portfolio
    portfolio = Portfolio(initial_balance=100000)
    print(f"Initial balance: ${portfolio.total_equity:,.2f}")
    
    # Execute some trades
    print("\nExecuting trades...")
    
    # Buy AAPL
    success = portfolio.execute_buy("AAPL", 100, 180.0, fees=1.0)
    print(f"Buy AAPL: {'Success' if success else 'Failed'}")
    
    # Buy MSFT
    success = portfolio.execute_buy("MSFT", 50, 340.0, fees=1.0)
    print(f"Buy MSFT: {'Success' if success else 'Failed'}")
    
    # Update with market data
    market_data = {"AAPL": 185.0, "MSFT": 350.0}
    portfolio.update_market_data(market_data)
    
    print(f"\nPortfolio after trades:")
    print(f"Cash balance: ${portfolio.cash_balance:,.2f}")
    print(f"Total equity: ${portfolio.total_equity:,.2f}")
    print(f"Unrealized P&L: ${portfolio.total_unrealized_pnl:,.2f}")
    
    # Show positions
    print(f"\nPositions:")
    for pos in portfolio.get_positions():
        print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['average_cost']:.2f}")
        print(f"    Current: ${pos['current_price']:.2f}, P&L: ${pos['unrealized_pnl']:.2f} ({pos['pnl_percent']:.1f}%)")
    
    # Sell some AAPL
    success = portfolio.execute_sell("AAPL", 50, 185.0, fees=1.0)
    print(f"\nSell 50 AAPL: {'Success' if success else 'Failed'}")
    
    # Show performance
    performance = portfolio.get_performance_metrics()
    print(f"\nPerformance metrics:")
    print(f"Total return: ${performance['total_return']:.2f} ({performance['total_return_percent']:.2f}%)")
    print(f"Win rate: {performance['win_rate']:.1f}%")
    print(f"Total trades: {performance['total_trades']}")
    
    print("✅ Portfolio system test completed!")