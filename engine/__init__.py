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
    from .sentiment_model.api import pull_from_web as _pull_from_web, analyze_sentiment as _analyze_sentiment_file
    
    # Create wrapper functions that match the expected interface
    def pull_from_web(ticker: str, output_name: str) -> None:
        """Extract content from URL. If no URL provided, use the existing pull_from_web function."""
        _pull_from_web(ticker, output_name)
        return
    
    def analyze_sentiment(text=None):
        """Analyze sentiment of provided text or use file-based analysis."""
        if text:
            # Simple fallback sentiment analysis using TextBlob if available
            try:
                from textblob import TextBlob
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                if polarity > 0.1:
                    sentiment = "positive"
                elif polarity < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                # Calculate confidence based on polarity magnitude and subjectivity
                # Higher subjectivity indicates more opinion-based text (better for sentiment)
                # Higher absolute polarity indicates stronger sentiment
                confidence = min(abs(polarity) + (subjectivity * 0.3), 1.0)
                
                # Ensure minimum confidence for neutral sentiment
                if sentiment == "neutral" and confidence < 0.2:
                    confidence = 0.2
                    
                return {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "polarity": polarity
                }
            except ImportError:
                # Fallback to very basic analysis
                text_lower = str(text).lower()
                positive_words = ['good', 'great', 'excellent', 'bullish', 'up', 'gain', 'profit', 'buy']
                negative_words = ['bad', 'terrible', 'bearish', 'down', 'loss', 'sell', 'drop', 'fall']
                
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    return {"sentiment": "positive", "confidence": 0.5}
                elif neg_count > pos_count:
                    return {"sentiment": "negative", "confidence": 0.5}
                else:
                    return {"sentiment": "neutral", "confidence": 0.3}
        else:
            return _analyze_sentiment_file()
            
except ImportError:
    # Fallback if sentiment model not available
    def pull_from_web(ticker: str, output_name: str) -> None:
        return

    def analyze_sentiment(text=None):
        if text:
            # Very basic fallback
            text_lower = str(text).lower()
            if any(word in text_lower for word in ['good', 'great', 'bullish', 'up']):
                return {"sentiment": "positive", "confidence": 0.3}
            elif any(word in text_lower for word in ['bad', 'bearish', 'down']):
                return {"sentiment": "negative", "confidence": 0.3}
            else:
                return {"sentiment": "neutral", "confidence": 0.2}
        raise ImportError("Sentiment model not available. Check sentiment_model installation.")

# Import unified model functions
try:
    from .unified_model import train_model, predict_price, get_model_info, list_available_models
except ImportError as e:
    # Fallback with detailed error message
    error_msg = str(e)  # Capture error message as string
    def train_model(*args, **kwargs):
        raise ImportError(f"Unified model functions not available: {error_msg}")
    
    def predict_price(*args, **kwargs):
        raise ImportError(f"Unified model functions not available: {error_msg}")
    
    def get_model_info(*args, **kwargs):
        raise ImportError(f"Unified model functions not available: {error_msg}")
    
    def list_available_models(*args, **kwargs):
        raise ImportError(f"Unified model functions not available: {error_msg}")

# Import paper trading functions
try:
    from .trading_simulation import PaperTradingEngine, Portfolio, BuyAndHoldStrategy, MomentumStrategy
    
    def create_trading_engine(*args, **kwargs):
        """Create a paper trading engine."""
        return PaperTradingEngine(*args, **kwargs)
        
except ImportError as e:
    # Fallback for paper trading
    error_msg = str(e)  # Capture error message as string
    def create_trading_engine(*args, **kwargs):
        raise ImportError(f"Paper trading functions not available: {error_msg}")
    
    # Create dummy classes
    class PaperTradingEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Paper trading not available: {error_msg}")
    
    class Portfolio:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Paper trading not available: {error_msg}")
    
    class BuyAndHoldStrategy:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Paper trading not available: {error_msg}")
    
    class MomentumStrategy:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Paper trading not available: {error_msg}")

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