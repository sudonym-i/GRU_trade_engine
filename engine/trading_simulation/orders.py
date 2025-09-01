"""
Order Management for Paper Trading

Handles different types of trading orders and their execution logic.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid


class OrderType(Enum):
    """Types of trading orders."""
    MARKET = "market"
    LIMIT = "limit" 
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Represents a trading order.
    """
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    order_id: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.order_id is None:
            self.order_id = str(uuid.uuid4())[:8]
    
    @property
    def remaining_quantity(self) -> int:
        """Remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_partial(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIAL
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]
    
    def can_execute_at_price(self, current_price: float) -> bool:
        """
        Check if order can be executed at current market price.
        
        Args:
            current_price: Current market price of the stock
            
        Returns:
            True if order can be executed
        """
        if not self.is_active:
            return False
        
        if self.order_type == OrderType.MARKET:
            return True
        
        elif self.order_type == OrderType.LIMIT:
            if self.side == OrderSide.BUY:
                # Buy limit: execute if market price <= limit price
                return current_price <= self.limit_price
            else:
                # Sell limit: execute if market price >= limit price
                return current_price >= self.limit_price
        
        elif self.order_type == OrderType.STOP:
            if self.side == OrderSide.BUY:
                # Buy stop: execute if market price >= stop price
                return current_price >= self.stop_price
            else:
                # Sell stop: execute if market price <= stop price
                return current_price <= self.stop_price
        
        elif self.order_type == OrderType.STOP_LIMIT:
            # First check if stop is triggered
            stop_triggered = False
            if self.side == OrderSide.BUY and current_price >= self.stop_price:
                stop_triggered = True
            elif self.side == OrderSide.SELL and current_price <= self.stop_price:
                stop_triggered = True
            
            if not stop_triggered:
                return False
            
            # Then check limit price (same as limit order)
            if self.side == OrderSide.BUY:
                return current_price <= self.limit_price
            else:
                return current_price >= self.limit_price
        
        return False
    
    def execute(self, price: float, quantity: int = None) -> int:
        """
        Execute the order at given price.
        
        Args:
            price: Execution price
            quantity: Quantity to fill (if None, fills remaining quantity)
            
        Returns:
            Quantity actually filled
        """
        if not self.is_active:
            return 0
        
        if quantity is None:
            quantity = self.remaining_quantity
        else:
            quantity = min(quantity, self.remaining_quantity)
        
        # Update order state
        self.filled_quantity += quantity
        
        if self.filled_quantity == 0:
            # First fill
            self.filled_price = price
        else:
            # Average price for partial fills
            total_value = (self.filled_price * (self.filled_quantity - quantity) + 
                          price * quantity)
            self.filled_price = total_value / self.filled_quantity
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        else:
            self.status = OrderStatus.PARTIAL
        
        return quantity
    
    def cancel(self):
        """Cancel the order."""
        if self.is_active:
            self.status = OrderStatus.CANCELLED
    
    def to_dict(self) -> dict:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'status': self.status.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None
        }
    
    def __str__(self) -> str:
        """String representation of order."""
        return (f"Order({self.order_id}): {self.side.value.upper()} {self.quantity} "
                f"{self.symbol} @ {self.order_type.value.upper()} "
                f"[{self.status.value.upper()}]")


class OrderManager:
    """
    Manages multiple orders and their execution.
    """
    
    def __init__(self):
        self.orders = {}  # order_id -> Order
        self.active_orders = {}  # symbol -> [order_ids]
    
    def add_order(self, order: Order) -> str:
        """
        Add an order to the manager.
        
        Args:
            order: Order to add
            
        Returns:
            Order ID
        """
        self.orders[order.order_id] = order
        
        # Track active orders by symbol
        if order.is_active:
            if order.symbol not in self.active_orders:
                self.active_orders[order.symbol] = []
            self.active_orders[order.symbol].append(order.order_id)
        
        return order.order_id
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: str = None) -> list:
        """
        Get active orders.
        
        Args:
            symbol: Filter by symbol (if None, returns all active orders)
            
        Returns:
            List of active orders
        """
        if symbol:
            order_ids = self.active_orders.get(symbol, [])
            return [self.orders[oid] for oid in order_ids if self.orders[oid].is_active]
        else:
            return [order for order in self.orders.values() if order.is_active]
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was cancelled
        """
        order = self.orders.get(order_id)
        if order and order.is_active:
            order.cancel()
            self._remove_from_active(order)
            return True
        return False
    
    def process_market_data(self, symbol: str, current_price: float) -> list:
        """
        Process market data and execute eligible orders.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            
        Returns:
            List of executed orders
        """
        executed_orders = []
        
        # Get active orders for this symbol
        active_orders = self.get_active_orders(symbol)
        
        for order in active_orders:
            if order.can_execute_at_price(current_price):
                # Execute the order
                filled_qty = order.execute(current_price)
                executed_orders.append((order, filled_qty))
                
                # Remove from active orders if fully filled or cancelled
                if not order.is_active:
                    self._remove_from_active(order)
        
        return executed_orders
    
    def _remove_from_active(self, order: Order):
        """Remove order from active orders tracking."""
        if order.symbol in self.active_orders:
            if order.order_id in self.active_orders[order.symbol]:
                self.active_orders[order.symbol].remove(order.order_id)
            
            # Clean up empty symbol lists
            if not self.active_orders[order.symbol]:
                del self.active_orders[order.symbol]
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> list:
        """
        Get order history.
        
        Args:
            symbol: Filter by symbol (if None, returns all)
            limit: Maximum number of orders to return
            
        Returns:
            List of orders sorted by creation time (newest first)
        """
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        # Sort by creation time (newest first)
        orders.sort(key=lambda o: o.created_at, reverse=True)
        
        return orders[:limit]


if __name__ == "__main__":
    # Test the order system
    print("Testing Order System...")
    
    # Create orders
    market_buy = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    
    limit_sell = Order(
        symbol="AAPL", 
        side=OrderSide.SELL,
        quantity=50,
        order_type=OrderType.LIMIT,
        limit_price=185.0
    )
    
    print(f"Market buy order: {market_buy}")
    print(f"Limit sell order: {limit_sell}")
    
    # Test execution
    current_price = 180.0
    
    print(f"\nCan market buy execute at ${current_price}? {market_buy.can_execute_at_price(current_price)}")
    print(f"Can limit sell execute at ${current_price}? {limit_sell.can_execute_at_price(current_price)}")
    
    # Execute market buy
    filled = market_buy.execute(current_price)
    print(f"\nExecuted {filled} shares at ${current_price}")
    print(f"Order status: {market_buy.status}")
    
    # Test order manager
    manager = OrderManager()
    manager.add_order(market_buy)
    manager.add_order(limit_sell)
    
    print(f"\nActive orders: {len(manager.get_active_orders())}")
    
    # Process market data
    executed = manager.process_market_data("AAPL", 186.0)
    print(f"Executed orders at $186: {len(executed)}")
    
    print("✅ Order system test completed!")