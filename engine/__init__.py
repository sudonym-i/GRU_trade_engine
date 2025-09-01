"""
Neural Trade Engine

Main package for stock prediction using unified technical and fundamental analysis
with sentiment analysis integration.

Available functions:
    # Sentiment Analysis (from sentiment_model)
    - pull_from_web: Extract content from web sources
    - analyze_sentiment: Analyze sentiment of text data
    
    # Stock Prediction (from unified_model)  
    - train_model: Train the unified prediction model
    - predict_price: Predict next stock price
    
Example usage:
    from engine import train_model, predict_price, analyze_sentiment
    
    # Train model
    trainer = train_model(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")
    
    # Make prediction
    prediction = predict_price("AAPL", model_path="models/unified_model.pth")
    
    # Analyze sentiment
    sentiment = analyze_sentiment("Stock market looks bullish today")
"""

# Import sentiment analysis functions
try:
    from .sentiment_model.route import pull_from_web, analyze_sentiment
except ImportError:
    # Fallback if sentiment model not available
    def pull_from_web(*args, **kwargs):
        raise ImportError("Sentiment model not available. Check sentiment_model installation.")
    
    def analyze_sentiment(*args, **kwargs):
        raise ImportError("Sentiment model not available. Check sentiment_model installation.")

# Import unified model functions
try:
    from .unified_model import train_model, predict_price, get_model_info, list_available_models
except ImportError as e:
    # Fallback with detailed error message
    def train_model(*args, **kwargs):
        raise ImportError(f"Unified model functions not available: {e}")
    
    def predict_price(*args, **kwargs):
        raise ImportError(f"Unified model functions not available: {e}")
    
    def get_model_info(*args, **kwargs):
        raise ImportError(f"Unified model functions not available: {e}")
    
    def list_available_models(*args, **kwargs):
        raise ImportError(f"Unified model functions not available: {e}")

# Import paper trading functions
try:
    from .paper_trading import PaperTradingEngine, Portfolio, BuyAndHoldStrategy, MomentumStrategy
    
    def create_trading_engine(*args, **kwargs):
        """Create a paper trading engine."""
        return PaperTradingEngine(*args, **kwargs)
        
except ImportError as e:
    # Fallback for paper trading
    def create_trading_engine(*args, **kwargs):
        raise ImportError(f"Paper trading functions not available: {e}")
    
    # Create dummy classes
    class PaperTradingEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Paper trading not available: {e}")
    
    class Portfolio:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Paper trading not available: {e}")
    
    class BuyAndHoldStrategy:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Paper trading not available: {e}")
    
    class MomentumStrategy:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Paper trading not available: {e}")

__all__ = [
    # Sentiment Analysis
    'pull_from_web',
    'analyze_sentiment',
    
    # Stock Prediction  
    'train_model', 
    'predict_price',
    'get_model_info',
    'list_available_models',
    
    # Paper Trading
    'create_trading_engine',
    'PaperTradingEngine',
    'Portfolio',
    'BuyAndHoldStrategy',
    'MomentumStrategy'
]