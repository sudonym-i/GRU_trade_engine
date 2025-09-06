#!/usr/bin/env python3
"""
Automated Trading Script for Neural Trade Engine

This script runs daily after market close to:
1. Webscrape sentiment data for target stocks
2. Generate price predictions using trained models
3. Create buy/sell/hold signals for next trading day
4. Execute trades through multiple backends (simulation, IB paper, IB live)
5. Log trading decisions and portfolio status

Trading Modes:
- simulation: Paper trading with virtual portfolio (default)
- ib_paper: Interactive Brokers paper trading (port 7496)
- ib_live: Interactive Brokers live trading (port 7497)

Usage:
    # Simulation mode (default)
    python automated_trader.py --stock AAPL --semantic-name apple
    
    # Interactive Brokers paper trading
    python automated_trader.py --mode ib_paper --stock NVDA --semantic-name nvidia
    
    # Interactive Brokers live trading
    python automated_trader.py --mode ib_live --stock TSLA --semantic-name tesla
    
    # With custom configuration
    python automated_trader.py --config config.json --mode ib_paper --stock GOOGL --semantic-name google
    
    # Dry run (forces simulation mode)
    python automated_trader.py --dry-run --stock MSFT --semantic-name microsoft

Requirements for IB modes:
- Interactive Brokers TWS or IB Gateway running
- API enabled in TWS/Gateway settings
- Correct port configuration (7496 for paper, 7497 for live)
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
import asyncio

# Import IB interface
from ib_interface import IBTradingInterface, AsyncIBInterface

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
    
    def __init__(self, config_path: Optional[str] = None, trading_mode: Optional[str] = None, semantic_name: Optional[str] = None):
        """
        Initialize the automated trader.
        
        Args:
            config_path: Path to configuration file
            trading_mode: Trading mode ('simulation', 'ib_paper', 'ib_live') - overrides config
            semantic_name: Semantic name for webscraping (e.g., "google" for GOOGL)
        """
        self.config = self._load_config(config_path)
        # Use provided trading_mode or fall back to config, then default
        self.trading_mode = trading_mode or self.config.get('trading_mode', 'simulation')
        self.semantic_name = semantic_name
        self.backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend_&_algorithms')
        self.portfolio_file = f'portfolio_state_{self.trading_mode}.json'
        self.trading_log = f'trading_decisions_{self.trading_mode}.json'
        
        # Initialize IB interface if needed
        self.ib_interface = None
        if self.trading_mode in ['ib_paper', 'ib_live']:
            mode = 'paper' if self.trading_mode == 'ib_paper' else 'live'
            self.ib_interface = IBTradingInterface(mode=mode)
        
        # Initialize portfolio if not exists
        self._init_portfolio()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "target_stock": "NVDA",
            "trading_mode": "simulation",
            "time_interval": "1d",  # Default time interval
            "confidence_threshold": 0.65,
            "price_change_threshold": 0.02,  # 2% minimum price change
            "position_size": 1.0,  # 100% focus on single stock
            "stop_loss": -0.05,  # -5% stop loss
            "take_profit": 0.10,  # +10% take profit
            "cash_reserve": 0.05,  # Keep 5% in cash
            "sentiment_bias_strength": 0.15  # 15% max sentiment bias adjustment
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
            if self.trading_mode.startswith('ib_'):
                # For IB modes, we'll sync with actual account data
                initial_portfolio = {
                    "cash": 0.0,
                    "positions": {},
                    "total_value": 0.0,
                    "last_updated": datetime.now().isoformat(),
                    "mode": self.trading_mode
                }
                logger.info(f"Initialized IB portfolio file for {self.trading_mode} mode")
            else:
                # For simulation mode, start with virtual capital
                initial_portfolio = {
                    "cash": 10000.0,  # $10k starting capital
                    "positions": {},
                    "total_value": 10000.0,
                    "last_updated": datetime.now().isoformat(),
                    "mode": self.trading_mode
                }
                logger.info("Initialized simulation portfolio with $10,000 starting capital")
            
            self._save_portfolio(initial_portfolio)

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
    
    async def _sync_ib_portfolio(self) -> Optional[Dict]:
        """
        Sync portfolio with Interactive Brokers account data.
        
        Returns:
            Updated portfolio dict or None if failed
        """
        if not self.ib_interface:
            return None
        
        try:
            # Connect to IB if not connected
            if not self.ib_interface.connected:
                success = await self.ib_interface.connect()
                if not success:
                    logger.error("Failed to connect to IB for portfolio sync")
                    return None
            
            # Get account info from IB
            account_info = self.ib_interface.get_account_info()
            if account_info:
                logger.info(f"Synced portfolio with IB: ${account_info.get('total_value', 0):.2f} total value")
                return account_info
            else:
                logger.error("Failed to get account info from IB")
                return None
                
        except Exception as e:
            logger.error(f"Error syncing IB portfolio: {e}")
            return None
    
    def run_webscraping(self, ticker: str) -> bool:
        """
        Run webscraping for sentiment analysis.
        
        Args:
            ticker: Stock ticker (for logging purposes)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use semantic name for webscraping if provided, otherwise use ticker
            webscrape_name = self.semantic_name if self.semantic_name else ticker
            logger.info(f"Running webscraping for {ticker} (using semantic name: {webscrape_name})...")
            
            # Get time interval from config (in case webscraping needs it for frequency)
            time_interval = self.config.get('time_interval', '1d')
            
            cmd = [
                'python3', 'main.py', 'webscrape', 
                '--ticker', webscrape_name,
                '--interval', time_interval
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
            
            # Get time interval from config
            time_interval = self.config.get('time_interval', '1d')
            logger.info(f"Using time interval: {time_interval}")
            
            cmd = [
                'python3', 'main.py', 'predict', 
                '--ticker', ticker,
                '--interval', time_interval
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
    
    def get_sentiment_analysis(self, ticker: str) -> Optional[Dict]:
        """
        Get sentiment analysis results for a stock.
        Since sentiment is integrated into the prediction process, we'll simulate
        sentiment data based on recent webscraping results and news patterns.
        
        Args:
            ticker: Stock ticker (for logging purposes)
            
        Returns:
            Dictionary with sentiment data or None if failed
        """
        try:
            # Use semantic name for sentiment analysis if provided, otherwise use ticker
            sentiment_name = self.semantic_name if self.semantic_name else ticker
            logger.info(f"Analyzing sentiment patterns for {ticker} (using semantic name: {sentiment_name})...")
            
            # For now, we'll create a placeholder sentiment analysis that can be enhanced
            # when the backend provides dedicated sentiment output
            # This simulates sentiment based on basic patterns
            import random
            import time
            
            # Simulate sentiment analysis with some basic logic
            # In a real implementation, this would parse sentiment files or database
            current_time = time.time()
            
            # Simple sentiment simulation based on time and ticker
            # This should be replaced with actual sentiment data parsing
            sentiment_seed = hash(sentiment_name + str(int(current_time / 3600))) % 1000
            random.seed(sentiment_seed)
            
            # Generate somewhat realistic sentiment data
            sentiment_labels = ['bullish', 'neutral', 'bearish']
            weights = [0.35, 0.40, 0.25]  # Slightly optimistic bias
            
            sentiment_label = random.choices(sentiment_labels, weights=weights)[0]
            
            if sentiment_label == 'bullish':
                sentiment_score = random.uniform(0.2, 0.8)
            elif sentiment_label == 'bearish':
                sentiment_score = random.uniform(-0.8, -0.2)
            else:
                sentiment_score = random.uniform(-0.2, 0.2)
            
            sentiment_confidence = random.uniform(0.6, 0.9)
            
            sentiment_data = {
                'sentiment_score': sentiment_score,
                'sentiment_confidence': sentiment_confidence,
                'sentiment_label': sentiment_label,
                'source': 'simulated'  # Mark as simulated data
            }
            
            logger.info(f"Sentiment analysis complete for {ticker}: {sentiment_label} (score: {sentiment_score:+.2f})")
            return sentiment_data
                
        except Exception as e:
            logger.warning(f"Sentiment analysis error for {ticker}: {e}")
            return None
    
    def parse_sentiment_output(self, output: str) -> Optional[Dict]:
        """
        Parse sentiment analysis output to extract sentiment score.
        
        Args:
            output: Raw stdout from sentiment command
            
        Returns:
            Dictionary with parsed sentiment data
        """
        try:
            lines = output.strip().split('\n')
            sentiment_data = {}
            
            for line in lines:
                # Look for sentiment score patterns
                if "Sentiment Score:" in line or "Sentiment:" in line:
                    import re
                    # Extract sentiment score (-1.0 to 1.0)
                    sentiment_match = re.search(r'(-?[0-9]+\.?[0-9]*)', line)
                    if sentiment_match:
                        sentiment_data['sentiment_score'] = float(sentiment_match.group(1))
                
                if "Confidence:" in line:
                    # Extract sentiment confidence
                    import re
                    conf_match = re.search(r'([0-9]+\.?[0-9]*)', line)
                    if conf_match:
                        sentiment_data['sentiment_confidence'] = float(conf_match.group(1))
                        
                # Look for sentiment classification
                if "Bullish" in line or "Positive" in line:
                    sentiment_data['sentiment_label'] = 'bullish'
                elif "Bearish" in line or "Negative" in line:
                    sentiment_data['sentiment_label'] = 'bearish'
                elif "Neutral" in line:
                    sentiment_data['sentiment_label'] = 'neutral'
            
            # If no explicit score found, derive from label
            if 'sentiment_score' not in sentiment_data and 'sentiment_label' in sentiment_data:
                if sentiment_data['sentiment_label'] == 'bullish':
                    sentiment_data['sentiment_score'] = 0.5
                elif sentiment_data['sentiment_label'] == 'bearish':
                    sentiment_data['sentiment_score'] = -0.5
                else:
                    sentiment_data['sentiment_score'] = 0.0
            
            return sentiment_data if sentiment_data else None
            
        except Exception as e:
            logger.error(f"Failed to parse sentiment output: {e}")
            return None
    
    def generate_trading_signal(self, ticker: str, prediction_data: Dict, sentiment_data: Optional[Dict] = None) -> str:
        """
        Generate trading signal based on prediction with sentiment bias.
        
        Args:
            ticker: Stock ticker
            prediction_data: Parsed prediction data
            sentiment_data: Optional sentiment analysis data
            
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
            
            # Calculate base price change percentage
            base_price_change = (predicted_price - current_price) / current_price
            
            # Apply sentiment bias if available
            sentiment_adjusted_price_change = base_price_change
            sentiment_info = ""
            
            if sentiment_data:
                sentiment_score = sentiment_data.get('sentiment_score', 0.0)  # -1.0 to 1.0
                sentiment_confidence = sentiment_data.get('sentiment_confidence', 0.5)
                sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
                
                # Configure sentiment bias strength (adjustable via config)
                sentiment_bias_strength = self.config.get('sentiment_bias_strength', 0.15)  # 15% bias max
                
                # Apply sentiment bias weighted by sentiment confidence
                sentiment_adjustment = sentiment_score * sentiment_bias_strength * sentiment_confidence
                sentiment_adjusted_price_change = base_price_change + sentiment_adjustment
                
                sentiment_info = f" [Sentiment: {sentiment_label} ({sentiment_score:+.2f}), bias: {sentiment_adjustment:+.1%}]"
                logger.info(f"{ticker}: Price prediction: {base_price_change:.1%} → {sentiment_adjusted_price_change:.1%} with sentiment bias{sentiment_info}")
            
            # Check confidence threshold
            if confidence < self.config['confidence_threshold']:
                logger.info(f"{ticker}: Low model confidence ({confidence:.2f}), holding{sentiment_info}")
                return 'HOLD'
            
            # Generate signal based on sentiment-adjusted price change
            threshold = self.config['price_change_threshold']
            
            if sentiment_adjusted_price_change > threshold:
                signal_strength = sentiment_adjusted_price_change
                logger.info(f"{ticker}: BUY signal - {signal_strength:.1%} expected return{sentiment_info}")
                return 'BUY'
            elif sentiment_adjusted_price_change < -threshold:
                signal_strength = abs(sentiment_adjusted_price_change)
                logger.info(f"{ticker}: SELL signal - {signal_strength:.1%} expected decline{sentiment_info}")
                return 'SELL'
            else:
                logger.info(f"{ticker}: HOLD signal - {sentiment_adjusted_price_change:.1%} expected change below threshold{sentiment_info}")
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return 'HOLD'
    
    async def execute_trading_decisions(self, signals: Dict[str, Tuple[str, Dict]]):
        """
        Execute trading decisions based on signals.
        
        Args:
            signals: Dictionary of {ticker: (signal, prediction_data)}
        """
        # Sync with IB if using IB mode
        if self.trading_mode.startswith('ib_'):
            ib_portfolio = await self._sync_ib_portfolio()
            if ib_portfolio:
                self._save_portfolio(ib_portfolio)
        
        portfolio = self._load_portfolio()
        
        for ticker, (signal, prediction_data) in signals.items():
            try:
                decision = {
                    "ticker": ticker,
                    "signal": signal,
                    "prediction_data": prediction_data,
                    "action_taken": "None",
                    "trading_mode": self.trading_mode
                }
                
                if signal == 'BUY':
                    success = await self._execute_buy(portfolio, ticker, prediction_data)
                    decision["action_taken"] = "Buy attempted" if success else "Buy failed"
                elif signal == 'SELL':
                    success = await self._execute_sell(portfolio, ticker, prediction_data)
                    decision["action_taken"] = "Sell attempted" if success else "Sell failed"
                else:
                    decision["action_taken"] = "Hold position"
                
                self._log_trading_decision(decision)
                
            except Exception as e:
                logger.error(f"Error executing decision for {ticker}: {e}")
        
        # Sync again after trades for IB modes
        if self.trading_mode.startswith('ib_'):
            ib_portfolio = await self._sync_ib_portfolio()
            if ib_portfolio:
                self._save_portfolio(ib_portfolio)
        else:
            # Save updated portfolio for simulation mode
            self._save_portfolio(portfolio)
    
    async def _execute_buy(self, portfolio: Dict, ticker: str, prediction_data: Dict) -> bool:
        """Execute a buy order."""
        try:
            current_price = prediction_data.get('current_price', 100)  # Fallback price
            
            # For IB modes, use IB interface
            if self.trading_mode.startswith('ib_'):
                return await self._execute_ib_buy(ticker, prediction_data)
            
            # Simulation mode logic
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
            
            # Execute buy in simulation
            portfolio['cash'] -= cost
            portfolio['positions'][ticker] = {
                'shares': shares,
                'avg_price': current_price,
                'purchase_date': datetime.now().isoformat()
            }
            
            logger.info(f"SIMULATION BUY: {shares} shares of {ticker} at ${current_price:.2f} (${cost:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Buy execution failed for {ticker}: {e}")
            return False
    
    async def _execute_ib_buy(self, ticker: str, prediction_data: Dict) -> bool:
        """Execute a buy order through Interactive Brokers."""
        try:
            if not self.ib_interface:
                logger.error("IB interface not initialized")
                return False
            
            # Connect if not connected
            if not self.ib_interface.connected:
                success = await self.ib_interface.connect()
                if not success:
                    logger.error("Failed to connect to IB")
                    return False
            
            # Get current account info
            account_info = self.ib_interface.get_account_info()
            if not account_info:
                logger.error("Could not get account info for buy order")
                return False
            
            # Calculate position size
            available_cash = account_info.get('cash', 0)
            current_price = self.ib_interface.get_market_price(ticker)
            if not current_price:
                current_price = prediction_data.get('current_price', 0)
            
            if current_price <= 0:
                logger.error(f"Invalid price for {ticker}")
                return False
            
            # Calculate shares to buy
            position_value = account_info.get('total_value', 0) * self.config['position_size']
            shares = int(min(position_value, available_cash) / current_price)
            
            if shares <= 0:
                logger.warning(f"Cannot buy {ticker}: insufficient funds or invalid calculation")
                return False
            
            # Check if already holding this stock
            if ticker in account_info.get('positions', {}):
                logger.info(f"Already holding {ticker}, cannot buy more (single stock focus)")
                return False
            
            # Execute the buy order
            success = self.ib_interface.execute_buy_order(ticker, shares)
            if success:
                logger.info(f"IB BUY ORDER: {shares} shares of {ticker}")
                return True
            else:
                logger.error(f"IB buy order failed for {ticker}")
                return False
            
        except Exception as e:
            logger.error(f"IB buy execution failed for {ticker}: {e}")
            return False
    
    async def _execute_sell(self, portfolio: Dict, ticker: str, prediction_data: Dict) -> bool:
        """Execute a sell order."""
        try:
            # For IB modes, use IB interface
            if self.trading_mode.startswith('ib_'):
                return await self._execute_ib_sell(ticker, prediction_data)
            
            # Simulation mode logic
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
            
            # Execute sell in simulation
            portfolio['cash'] += proceeds
            del portfolio['positions'][ticker]
            
            logger.info(f"SIMULATION SELL: {shares} shares of {ticker} at ${current_price:.2f} (${proceeds:.2f})")
            logger.info(f"P&L: ${profit_loss:.2f} ({profit_loss_pct:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Sell execution failed for {ticker}: {e}")
            return False
    
    async def _execute_ib_sell(self, ticker: str, prediction_data: Dict) -> bool:
        """Execute a sell order through Interactive Brokers."""
        try:
            if not self.ib_interface:
                logger.error("IB interface not initialized")
                return False
            
            # Connect if not connected
            if not self.ib_interface.connected:
                success = await self.ib_interface.connect()
                if not success:
                    logger.error("Failed to connect to IB")
                    return False
            
            # Get current account info
            account_info = self.ib_interface.get_account_info()
            if not account_info:
                logger.error("Could not get account info for sell order")
                return False
            
            # Check if we have this position
            positions = account_info.get('positions', {})
            if ticker not in positions:
                logger.warning(f"No position in {ticker} to sell")
                return False
            
            position = positions[ticker]
            shares = int(position.get('shares', 0))
            
            if shares <= 0:
                logger.warning(f"Invalid position size for {ticker}: {shares}")
                return False
            
            # Execute the sell order
            success = self.ib_interface.execute_sell_order(ticker, shares)
            if success:
                logger.info(f"IB SELL ORDER: {shares} shares of {ticker}")
                return True
            else:
                logger.error(f"IB sell order failed for {ticker}")
                return False
            
        except Exception as e:
            logger.error(f"IB sell execution failed for {ticker}: {e}")
            return False
    
    async def run_daily_cycle(self, target_stock: Optional[str] = None):
        """
        Run the complete daily trading cycle for a single stock.
        
        Args:
            target_stock: Stock ticker to process (overrides config)
        """
        ticker = target_stock or self.config['target_stock']
        logger.info(f"Starting daily trading cycle for {ticker} (mode: {self.trading_mode})")
        
        try:
            # Connect to IB if using IB mode
            if self.trading_mode.startswith('ib_') and self.ib_interface:
                success = await self.ib_interface.connect()
                if not success:
                    logger.error("Failed to connect to IB, aborting cycle")
                    return
        
            logger.info(f"Processing {ticker}...")
            
            # Step 1: Run webscraping for sentiment data
            webscrape_success = self.run_webscraping(ticker)
            if not webscrape_success:
                logger.warning(f"Webscraping failed for {ticker}, continuing anyway...")
            
            # Step 2: Get sentiment analysis results
            sentiment_data = self.get_sentiment_analysis(ticker)
            if sentiment_data:
                sentiment_score = sentiment_data.get('sentiment_score', 0.0)
                sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
                logger.info(f"{ticker}: Sentiment analysis - {sentiment_label} (score: {sentiment_score:+.2f})")
            else:
                logger.warning(f"Sentiment analysis unavailable for {ticker}, proceeding without sentiment bias")
            
            # Step 3: Run prediction
            prediction_result = self.run_prediction(ticker)
            if not prediction_result:
                logger.error(f"Prediction failed for {ticker}, aborting cycle")
                return
            
            # Step 4: Parse prediction output
            prediction_data = self.parse_prediction_output(prediction_result['output'])
            if not prediction_data:
                logger.error(f"Failed to parse prediction for {ticker}, aborting cycle")
                return
            
            # Step 5: Generate trading signal with sentiment bias
            signal = self.generate_trading_signal(ticker, prediction_data, sentiment_data)
            logger.info(f"{ticker}: Final signal = {signal}")
            
            # Step 6: Execute trading decision
            signals = {ticker: (signal, prediction_data)}
            logger.info("Executing trading decision...")
            await self.execute_trading_decisions(signals)
            
            logger.info(f"Daily trading cycle complete for {ticker} ({self.trading_mode} mode)")
            
        finally:
            # Disconnect from IB if connected
            if self.trading_mode.startswith('ib_') and self.ib_interface:
                self.ib_interface.disconnect()


async def main():
    """Main entry point for automated trader."""
    parser = argparse.ArgumentParser(description='Automated Trading Script with Multiple Trading Modes')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--stock', type=str, help='Target stock ticker (overrides config)')
    parser.add_argument('--semantic-name', type=str, help='Semantic name for webscraping (e.g., "google" for GOOGL)')
    parser.add_argument('--dry-run', action='store_true', help='Run without executing trades')
    parser.add_argument('--mode', type=str, choices=['simulation', 'ib_paper', 'ib_live'], 
                       default='simulation', help='Trading mode (default: simulation)')
    parser.add_argument('--ib-host', type=str, default='127.0.0.1', 
                       help='Interactive Brokers host address (default: 127.0.0.1)')
    parser.add_argument('--ib-client-id', type=int, default=1,
                       help='IB client ID (default: 1)')
    
    args = parser.parse_args()
    
    # Parse target stock
    target_stock = None
    if args.stock:
        target_stock = args.stock.strip().upper()
    
    # Trading mode validation
    trading_mode = args.mode
    if args.dry_run and trading_mode != 'simulation':
        logger.info("DRY RUN MODE: Forcing simulation mode")
        trading_mode = 'simulation'
    
    logger.info(f"Starting automated trader in {trading_mode} mode")
    
    try:
        # Get semantic name from args
        semantic_name = getattr(args, 'semantic_name', None)
        trader = AutomatedTrader(args.config, trading_mode=trading_mode, semantic_name=semantic_name)
        
        # Set IB connection parameters if using IB
        if trading_mode.startswith('ib_') and trader.ib_interface:
            trader.ib_interface.host = args.ib_host
            trader.ib_interface.client_id = args.ib_client_id
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No trades will be executed")
        
        await trader.run_daily_cycle(target_stock)
        
    except Exception as e:
        logger.error(f"Automated trading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())