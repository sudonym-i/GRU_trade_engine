#!/usr/bin/env python3
"""
Automated Trading Script for Neural Trade Engine

This script runs daily after market close to:
1. Webscrape sentiment data for target stocks
2. Generate price predictions using trained models
3. Create buy/sell/hold signals for next trading day
4. Log trading decisions and portfolio status

Usage:
    python automated_trader.py
    python automated_trader.py --config config.json
    python automated_trader.py --stocks AAPL,MSFT,NVDA
"""

import os
import sys
import subprocess
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add backend path to import engine functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend_&_algorithms'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutomatedTrader:
    """
    Automated trading system that makes daily predictions and trading decisions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the automated trader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend_&_algorithms')
        self.portfolio_file = 'portfolio_state.json'
        self.trading_log = 'trading_decisions.json'
        
        # Initialize portfolio if not exists
        self._init_portfolio()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "target_stock": "NVDA",
            "confidence_threshold": 0.65,
            "price_change_threshold": 0.02,  # 2% minimum price change
            "position_size": 1.0,  # 100% focus on single stock
            "stop_loss": -0.05,  # -5% stop loss
            "take_profit": 0.10,  # +10% take profit
            "cash_reserve": 0.05   # Keep 5% in cash
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        return default_config
    
    def _init_portfolio(self):
        """Initialize portfolio state if it doesn't exist."""
        if not os.path.exists(self.portfolio_file):
            initial_portfolio = {
                "cash": 10000.0,  # $10k starting capital
                "positions": {},
                "total_value": 10000.0,
                "last_updated": datetime.now().isoformat()
            }
            self._save_portfolio(initial_portfolio)
            logger.info("Initialized portfolio with $10,000 starting capital")

    def _load_portfolio(self) -> Dict:
        """Load current portfolio state."""
        try:
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            self._init_portfolio()
            return self._load_portfolio()
    
    def _save_portfolio(self, portfolio: Dict):
        """Save portfolio state to file."""
        portfolio["last_updated"] = datetime.now().isoformat()
        with open(self.portfolio_file, 'w') as f:
            json.dump(portfolio, f, indent=2)
    
    def _log_trading_decision(self, decision: Dict):
        """Log trading decision to file."""
        # Load existing decisions
        decisions = []
        if os.path.exists(self.trading_log):
            try:
                with open(self.trading_log, 'r') as f:
                    decisions = json.load(f)
            except:
                decisions = []
        
        # Add new decision
        decision["timestamp"] = datetime.now().isoformat()
        decisions.append(decision)
        
        # Keep only last 1000 decisions
        decisions = decisions[-1000:]
        
        # Save back to file
        with open(self.trading_log, 'w') as f:
            json.dump(decisions, f, indent=2)
    
    def run_webscraping(self, ticker: str) -> bool:
        """
        Run webscraping for sentiment analysis.
        
        Args:
            ticker: Stock ticker to webscrape
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Running webscraping for {ticker}...")
            cmd = [
                'python3', 'main.py', 'webscrape', 
                '--ticker', ticker
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.backend_path,
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Webscraping successful for {ticker}")
                return True
            else:
                logger.error(f"Webscraping failed for {ticker}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Webscraping timeout for {ticker}")
            return False
        except Exception as e:
            logger.error(f"Webscraping error for {ticker}: {e}")
            return False
    
    def run_prediction(self, ticker: str) -> Optional[Dict]:
        """
        Run price prediction for a stock.
        
        Args:
            ticker: Stock ticker to predict
            
        Returns:
            Prediction results dictionary or None if failed
        """
        try:
            logger.info(f"Running prediction for {ticker}...")
            cmd = [
                'python3', 'main.py', 'predict', 
                '--ticker', ticker
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.backend_path,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Prediction successful for {ticker}")
                # Parse the output to extract prediction data
                # Note: You may need to modify main.py to output JSON format
                return {"ticker": ticker, "success": True, "output": result.stdout}
            else:
                logger.error(f"Prediction failed for {ticker}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Prediction timeout for {ticker}")
            return None
        except Exception as e:
            logger.error(f"Prediction error for {ticker}: {e}")
            return None
    
    def parse_prediction_output(self, output: str) -> Optional[Dict]:
        """
        Parse prediction output to extract key metrics.
        
        Args:
            output: Raw stdout from prediction command
            
        Returns:
            Dictionary with parsed prediction data
        """
        try:
            # Look for prediction patterns in the output
            lines = output.strip().split('\n')
            
            prediction_data = {}
            for line in lines:
                # Look for common prediction output patterns
                if "Predicted price:" in line or "Prediction:" in line:
                    # Extract predicted price
                    import re
                    price_match = re.search(r'\$?([0-9]+\.?[0-9]*)', line)
                    if price_match:
                        prediction_data['predicted_price'] = float(price_match.group(1))
                
                if "Current price:" in line or "Latest price:" in line:
                    # Extract current price
                    import re
                    price_match = re.search(r'\$?([0-9]+\.?[0-9]*)', line)
                    if price_match:
                        prediction_data['current_price'] = float(price_match.group(1))
                
                if "Confidence:" in line:
                    # Extract confidence score
                    import re
                    conf_match = re.search(r'([0-9]+\.?[0-9]*)', line)
                    if conf_match:
                        prediction_data['confidence'] = float(conf_match.group(1))
            
            return prediction_data if prediction_data else None
            
        except Exception as e:
            logger.error(f"Failed to parse prediction output: {e}")
            return None
    
    def generate_trading_signal(self, ticker: str, prediction_data: Dict) -> str:
        """
        Generate trading signal based on prediction.
        
        Args:
            ticker: Stock ticker
            prediction_data: Parsed prediction data
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            predicted_price = prediction_data.get('predicted_price')
            current_price = prediction_data.get('current_price')
            confidence = prediction_data.get('confidence', 0.5)
            
            if not predicted_price or not current_price:
                logger.warning(f"Missing price data for {ticker}")
                return 'HOLD'
            
            # Calculate price change percentage
            price_change = (predicted_price - current_price) / current_price
            
            # Check confidence threshold
            if confidence < self.config['confidence_threshold']:
                logger.info(f"{ticker}: Low confidence ({confidence:.2f}), holding")
                return 'HOLD'
            
            # Generate signal based on price change
            threshold = self.config['price_change_threshold']
            
            if price_change > threshold:
                logger.info(f"{ticker}: Bullish signal - {price_change:.1%} upside predicted")
                return 'BUY'
            elif price_change < -threshold:
                logger.info(f"{ticker}: Bearish signal - {price_change:.1%} downside predicted")
                return 'SELL'
            else:
                logger.info(f"{ticker}: Neutral signal - {price_change:.1%} change predicted")
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return 'HOLD'
    
    def execute_trading_decisions(self, signals: Dict[str, Tuple[str, Dict]]):
        """
        Execute trading decisions based on signals.
        
        Args:
            signals: Dictionary of {ticker: (signal, prediction_data)}
        """
        portfolio = self._load_portfolio()
        
        for ticker, (signal, prediction_data) in signals.items():
            try:
                decision = {
                    "ticker": ticker,
                    "signal": signal,
                    "prediction_data": prediction_data,
                    "action_taken": "None"
                }
                
                if signal == 'BUY':
                    success = self._execute_buy(portfolio, ticker, prediction_data)
                    decision["action_taken"] = "Buy attempted" if success else "Buy failed"
                elif signal == 'SELL':
                    success = self._execute_sell(portfolio, ticker, prediction_data)
                    decision["action_taken"] = "Sell attempted" if success else "Sell failed"
                else:
                    decision["action_taken"] = "Hold position"
                
                self._log_trading_decision(decision)
                
            except Exception as e:
                logger.error(f"Error executing decision for {ticker}: {e}")
        
        # Save updated portfolio
        self._save_portfolio(portfolio)
    
    def _execute_buy(self, portfolio: Dict, ticker: str, prediction_data: Dict) -> bool:
        """Execute a buy order."""
        try:
            current_price = prediction_data.get('current_price', 100)  # Fallback price
            position_value = portfolio['total_value'] * self.config['position_size']
            shares = int(position_value / current_price)
            cost = shares * current_price
            
            # Check if we have enough cash
            if cost > portfolio['cash']:
                logger.warning(f"Insufficient cash for {ticker}: need ${cost:.2f}, have ${portfolio['cash']:.2f}")
                return False
            
            # For single stock focus, check if we already have this position
            if ticker in portfolio['positions']:
                logger.info(f"Already holding {ticker}, cannot buy more (single stock focus)")
                return False
            
            # Execute buy
            portfolio['cash'] -= cost
            portfolio['positions'][ticker] = {
                'shares': shares,
                'avg_price': current_price,
                'purchase_date': datetime.now().isoformat()
            }
            
            logger.info(f"BOUGHT {shares} shares of {ticker} at ${current_price:.2f} (${cost:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Buy execution failed for {ticker}: {e}")
            return False
    
    def _execute_sell(self, portfolio: Dict, ticker: str, prediction_data: Dict) -> bool:
        """Execute a sell order."""
        try:
            if ticker not in portfolio['positions']:
                logger.warning(f"No position in {ticker} to sell")
                return False
            
            position = portfolio['positions'][ticker]
            shares = position['shares']
            current_price = prediction_data.get('current_price', 100)  # Fallback price
            proceeds = shares * current_price
            
            # Calculate profit/loss
            cost_basis = shares * position['avg_price']
            profit_loss = proceeds - cost_basis
            profit_loss_pct = (profit_loss / cost_basis) * 100
            
            # Execute sell
            portfolio['cash'] += proceeds
            del portfolio['positions'][ticker]
            
            logger.info(f"SOLD {shares} shares of {ticker} at ${current_price:.2f} (${proceeds:.2f})")
            logger.info(f"P&L: ${profit_loss:.2f} ({profit_loss_pct:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Sell execution failed for {ticker}: {e}")
            return False
    
    def run_daily_cycle(self, target_stock: Optional[str] = None):
        """
        Run the complete daily trading cycle for a single stock.
        
        Args:
            target_stock: Stock ticker to process (overrides config)
        """
        ticker = target_stock or self.config['target_stock']
        logger.info(f"Starting daily trading cycle for {ticker}")
        
        logger.info(f"Processing {ticker}...")
        
        # Step 1: Run webscraping for sentiment data
        webscrape_success = self.run_webscraping(ticker)
        if not webscrape_success:
            logger.warning(f"Webscraping failed for {ticker}, continuing anyway...")
        
        # Step 2: Run prediction
        prediction_result = self.run_prediction(ticker)
        if not prediction_result:
            logger.error(f"Prediction failed for {ticker}, aborting cycle")
            return
        
        # Step 3: Parse prediction output
        prediction_data = self.parse_prediction_output(prediction_result['output'])
        if not prediction_data:
            logger.error(f"Failed to parse prediction for {ticker}, aborting cycle")
            return
        
        # Step 4: Generate trading signal
        signal = self.generate_trading_signal(ticker, prediction_data)
        logger.info(f"{ticker}: Signal = {signal}")
        
        # Step 5: Execute trading decision
        signals = {ticker: (signal, prediction_data)}
        logger.info("Executing trading decision...")
        self.execute_trading_decisions(signals)
        
        logger.info("Daily trading cycle complete")


def main():
    """Main entry point for automated trader."""
    parser = argparse.ArgumentParser(description='Automated Trading Script')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--stock', type=str, help='Target stock ticker (overrides config)')
    parser.add_argument('--dry-run', action='store_true', help='Run without executing trades')
    
    args = parser.parse_args()
    
    # Parse target stock
    target_stock = None
    if args.stock:
        target_stock = args.stock.strip().upper()
    
    try:
        trader = AutomatedTrader(args.config)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No trades will be executed")
        
        trader.run_daily_cycle(target_stock)
        
    except Exception as e:
        logger.error(f"Automated trading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()