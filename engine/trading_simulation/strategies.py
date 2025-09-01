"""
Trading Strategies for Paper Trading

Base classes and example strategies that can be used with the paper trading engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TradingStrategy(ABC):
    """
    Base class for trading strategies.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        self.positions = {}
        self.last_signal_time = {}
        
    @abstractmethod
    def generate_signal(self, symbol: str, market_data: dict, portfolio_data: dict) -> Dict[str, any]:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: Stock symbol
            market_data: Current market data
            portfolio_data: Current portfolio information
            
        Returns:
            Signal dictionary with action, quantity, order_type, etc.
            Format: {
                'action': 'buy'|'sell'|'hold',
                'quantity': int,
                'order_type': 'market'|'limit'|'stop',
                'limit_price': float (optional),
                'confidence': float (0-1),
                'reason': str
            }
        """
        pass
    
    @abstractmethod
    def should_update_position(self, symbol: str, current_position: dict) -> bool:
        """
        Check if position should be updated.
        
        Args:
            symbol: Stock symbol
            current_position: Current position info
            
        Returns:
            True if position should be modified
        """
        pass
    
    def activate(self):
        """Activate the strategy."""
        self.is_active = True
        logger.info(f"Activated strategy: {self.name}")
    
    def deactivate(self):
        """Deactivate the strategy."""
        self.is_active = False
        logger.info(f"Deactivated strategy: {self.name}")
    
    def reset(self):
        """Reset strategy state."""
        self.positions = {}
        self.last_signal_time = {}
        logger.info(f"Reset strategy: {self.name}")


class BuyAndHoldStrategy(TradingStrategy):
    """
    Simple buy and hold strategy.
    """
    
    def __init__(self, target_allocation: Dict[str, float], rebalance_threshold: float = 0.05):
        """
        Initialize buy and hold strategy.
        
        Args:
            target_allocation: Target allocation percentages (symbol -> percentage)
            rebalance_threshold: Rebalance when allocation deviates by this much
        """
        super().__init__("Buy and Hold")
        self.target_allocation = target_allocation
        self.rebalance_threshold = rebalance_threshold
        
        # Validate allocations sum to 1.0
        total_allocation = sum(target_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Target allocations must sum to 1.0, got {total_allocation}")
    
    def generate_signal(self, symbol: str, market_data: dict, portfolio_data: dict) -> Dict[str, any]:
        """Generate signal based on target allocation."""
        if symbol not in self.target_allocation:
            return {'action': 'hold', 'reason': 'Not in target allocation'}
        
        target_pct = self.target_allocation[symbol]
        total_equity = portfolio_data.get('total_equity', 0)
        current_value = portfolio_data.get('positions', {}).get(symbol, {}).get('market_value', 0)
        current_pct = current_value / total_equity if total_equity > 0 else 0
        
        deviation = abs(current_pct - target_pct)
        
        if deviation > self.rebalance_threshold:
            target_value = total_equity * target_pct
            value_diff = target_value - current_value
            current_price = market_data.get('current_price', 0)
            
            if current_price > 0:
                quantity_needed = int(abs(value_diff) / current_price)
                
                if quantity_needed > 0:
                    action = 'buy' if value_diff > 0 else 'sell'
                    return {
                        'action': action,
                        'quantity': quantity_needed,
                        'order_type': 'market',
                        'confidence': min(deviation / self.rebalance_threshold, 1.0),
                        'reason': f'Rebalance: current {current_pct:.1%} vs target {target_pct:.1%}'
                    }
        
        return {'action': 'hold', 'reason': 'Within rebalance threshold'}
    
    def should_update_position(self, symbol: str, current_position: dict) -> bool:
        """Check if rebalancing is needed."""
        # Could add time-based rebalancing logic here
        return True


class MomentumStrategy(TradingStrategy):
    """
    Momentum-based trading strategy.
    """
    
    def __init__(self, lookback_days: int = 20, momentum_threshold: float = 0.05,
                 stop_loss_pct: float = 0.10, take_profit_pct: float = 0.20):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_days: Days to look back for momentum calculation
            momentum_threshold: Minimum momentum to trigger buy signal
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__("Momentum")
        self.lookback_days = lookback_days
        self.momentum_threshold = momentum_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def generate_signal(self, symbol: str, market_data: dict, portfolio_data: dict) -> Dict[str, any]:
        """Generate signal based on price momentum."""
        current_price = market_data.get('current_price', 0)
        historical_prices = market_data.get('historical_prices', [])
        
        if not historical_prices or len(historical_prices) < self.lookback_days:
            return {'action': 'hold', 'reason': 'Insufficient historical data'}
        
        # Calculate momentum (% change over lookback period)
        old_price = historical_prices[-self.lookback_days]
        momentum = (current_price - old_price) / old_price if old_price > 0 else 0
        
        # Check current position
        current_position = portfolio_data.get('positions', {}).get(symbol)
        has_position = current_position and current_position.get('quantity', 0) > 0
        
        if has_position:
            # Check for exit conditions
            avg_cost = current_position.get('average_cost', 0)
            pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0
            
            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return {
                    'action': 'sell',
                    'quantity': current_position.get('quantity', 0),
                    'order_type': 'market',
                    'confidence': 1.0,
                    'reason': f'Stop loss triggered: {pnl_pct:.1%} loss'
                }
            
            # Take profit
            if pnl_pct >= self.take_profit_pct:
                return {
                    'action': 'sell',
                    'quantity': current_position.get('quantity', 0),
                    'order_type': 'market',
                    'confidence': 0.8,
                    'reason': f'Take profit: {pnl_pct:.1%} gain'
                }
            
            # Hold existing position
            return {'action': 'hold', 'reason': f'Holding position, P&L: {pnl_pct:.1%}'}
        
        else:
            # Look for entry signal
            if momentum > self.momentum_threshold:
                # Calculate position size (simple: 5% of portfolio)
                total_equity = portfolio_data.get('total_equity', 0)
                position_value = total_equity * 0.05
                quantity = int(position_value / current_price) if current_price > 0 else 0
                
                if quantity > 0:
                    return {
                        'action': 'buy',
                        'quantity': quantity,
                        'order_type': 'market',
                        'confidence': min(momentum / self.momentum_threshold, 1.0),
                        'reason': f'Positive momentum: {momentum:.1%} over {self.lookback_days} days'
                    }
            
            return {'action': 'hold', 'reason': f'Momentum too low: {momentum:.1%}'}
    
    def should_update_position(self, symbol: str, current_position: dict) -> bool:
        """Always check momentum positions."""
        return True


class MLPredictionStrategy(TradingStrategy):
    """
    Strategy that uses ML model predictions for trading decisions.
    """
    
    def __init__(self, model_predictor, confidence_threshold: float = 0.6, 
                 position_size_pct: float = 0.05):
        """
        Initialize ML prediction strategy.
        
        Args:
            model_predictor: Function that takes symbol and returns prediction
            confidence_threshold: Minimum prediction confidence to trade
            position_size_pct: Position size as percentage of portfolio
        """
        super().__init__("ML Prediction")
        self.model_predictor = model_predictor
        self.confidence_threshold = confidence_threshold
        self.position_size_pct = position_size_pct
    
    def generate_signal(self, symbol: str, market_data: dict, portfolio_data: dict) -> Dict[str, any]:
        """Generate signal based on ML model prediction."""
        try:
            # Get prediction from model
            prediction = self.model_predictor(symbol)
            
            if not prediction:
                return {'action': 'hold', 'reason': 'No prediction available'}
            
            predicted_price = prediction.get('predicted_price', 0)
            confidence = prediction.get('confidence', 0)
            current_price = market_data.get('current_price', 0)
            
            if confidence < self.confidence_threshold:
                return {'action': 'hold', 'reason': f'Low confidence: {confidence:.2f}'}
            
            if current_price <= 0 or predicted_price <= 0:
                return {'action': 'hold', 'reason': 'Invalid price data'}
            
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price
            
            # Check current position
            current_position = portfolio_data.get('positions', {}).get(symbol)
            has_position = current_position and current_position.get('quantity', 0) > 0
            
            if has_position:
                # Exit if prediction suggests decline
                if expected_return < -0.02:  # 2% expected decline
                    return {
                        'action': 'sell',
                        'quantity': current_position.get('quantity', 0),
                        'order_type': 'market',
                        'confidence': confidence,
                        'reason': f'Model predicts {expected_return:.1%} decline'
                    }
                else:
                    return {'action': 'hold', 'reason': f'Model expects {expected_return:.1%} return'}
            
            else:
                # Enter if prediction suggests significant upside
                if expected_return > 0.05:  # 5% expected gain
                    total_equity = portfolio_data.get('total_equity', 0)
                    position_value = total_equity * self.position_size_pct
                    quantity = int(position_value / current_price)
                    
                    if quantity > 0:
                        return {
                            'action': 'buy',
                            'quantity': quantity,
                            'order_type': 'market',
                            'confidence': confidence,
                            'reason': f'Model predicts {expected_return:.1%} gain'
                        }
                
                return {'action': 'hold', 'reason': f'Expected return too low: {expected_return:.1%}'}
        
        except Exception as e:
            logger.error(f"Error in ML prediction strategy: {e}")
            return {'action': 'hold', 'reason': f'Prediction error: {e}'}
    
    def should_update_position(self, symbol: str, current_position: dict) -> bool:
        """Check ML predictions regularly."""
        return True


class StrategyManager:
    """
    Manages multiple trading strategies.
    """
    
    def __init__(self):
        self.strategies = {}
        self.strategy_weights = {}
    
    def add_strategy(self, strategy: TradingStrategy, weight: float = 1.0):
        """Add a trading strategy."""
        self.strategies[strategy.name] = strategy
        self.strategy_weights[strategy.name] = weight
        logger.info(f"Added strategy: {strategy.name} (weight: {weight})")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a trading strategy."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            del self.strategy_weights[strategy_name]
            logger.info(f"Removed strategy: {strategy_name}")
    
    def get_combined_signal(self, symbol: str, market_data: dict, portfolio_data: dict) -> Dict[str, any]:
        """
        Get combined signal from all active strategies.
        
        Returns:
            Combined signal with weighted confidence
        """
        if not self.strategies:
            return {'action': 'hold', 'reason': 'No strategies active'}
        
        signals = []
        total_weight = 0
        
        for name, strategy in self.strategies.items():
            if strategy.is_active:
                signal = strategy.generate_signal(symbol, market_data, portfolio_data)
                weight = self.strategy_weights[name]
                signals.append((signal, weight))
                total_weight += weight
        
        if not signals:
            return {'action': 'hold', 'reason': 'No active strategies'}
        
        # Simple voting system: majority action wins
        actions = [s[0]['action'] for s in signals]
        action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        weighted_confidence = 0
        
        for signal, weight in signals:
            action = signal['action']
            confidence = signal.get('confidence', 0.5)
            action_counts[action] += weight
            weighted_confidence += confidence * weight
        
        # Determine winning action
        winning_action = max(action_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        
        # For buy/sell signals, we need to determine quantity
        quantity = 0
        order_type = 'market'
        reasons = [s[0].get('reason', '') for s in signals if s[0]['action'] == winning_action]
        
        if winning_action in ['buy', 'sell']:
            # Use quantity from strongest signal
            strongest_signal = max([s for s in signals if s[0]['action'] == winning_action],
                                 key=lambda x: x[1])
            quantity = strongest_signal[0].get('quantity', 0)
            order_type = strongest_signal[0].get('order_type', 'market')
        
        return {
            'action': winning_action,
            'quantity': quantity,
            'order_type': order_type,
            'confidence': avg_confidence,
            'reason': f"Combined signal from {len(signals)} strategies: {'; '.join(reasons[:2])}"
        }


if __name__ == "__main__":
    # Test trading strategies
    print("Testing Trading Strategies...")
    
    # Test buy and hold strategy
    buy_hold = BuyAndHoldStrategy({'AAPL': 0.6, 'MSFT': 0.4})
    
    market_data = {'current_price': 180.0}
    portfolio_data = {
        'total_equity': 100000,
        'positions': {
            'AAPL': {'market_value': 50000, 'quantity': 278}
        }
    }
    
    signal = buy_hold.generate_signal('AAPL', market_data, portfolio_data)
    print(f"Buy & Hold signal for AAPL: {signal}")
    
    # Test momentum strategy
    momentum = MomentumStrategy(lookback_days=20, momentum_threshold=0.05)
    
    market_data_with_history = {
        'current_price': 180.0,
        'historical_prices': [170.0] * 19 + [175.0]  # 20 days of data
    }
    
    signal = momentum.generate_signal('MSFT', market_data_with_history, portfolio_data)
    print(f"Momentum signal for MSFT: {signal}")
    
    # Test strategy manager
    manager = StrategyManager()
    manager.add_strategy(buy_hold, weight=1.0)
    manager.add_strategy(momentum, weight=0.5)
    
    buy_hold.activate()
    momentum.activate()
    
    combined_signal = manager.get_combined_signal('AAPL', market_data, portfolio_data)
    print(f"Combined signal: {combined_signal}")
    
    print("✅ Trading strategies test completed!")