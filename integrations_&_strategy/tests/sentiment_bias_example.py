#!/usr/bin/env python3
"""
Example demonstrating sentiment bias in trading decisions.

This script shows how sentiment analysis affects trading signals.
"""

def demonstrate_sentiment_bias():
    """Show examples of how sentiment bias affects trading decisions."""
    
    # Configuration
    config = {
        'confidence_threshold': 0.65,
        'price_change_threshold': 0.02,  # 2% threshold
        'sentiment_bias_strength': 0.15   # 15% max bias
    }
    
    # Example scenarios
    scenarios = [
        {
            'name': 'Bullish Sentiment + Positive Prediction',
            'base_price_change': 0.025,  # 2.5% predicted gain
            'sentiment_score': 0.6,      # Strong bullish sentiment
            'sentiment_confidence': 0.8,
            'model_confidence': 0.70
        },
        {
            'name': 'Bearish Sentiment + Negative Prediction', 
            'base_price_change': -0.025, # 2.5% predicted loss
            'sentiment_score': -0.7,     # Strong bearish sentiment
            'sentiment_confidence': 0.9,
            'model_confidence': 0.75
        },
        {
            'name': 'Bullish Sentiment + Negative Prediction (Conflict)',
            'base_price_change': -0.015, # 1.5% predicted loss (below threshold)
            'sentiment_score': 0.8,      # Very bullish sentiment
            'sentiment_confidence': 0.7,
            'model_confidence': 0.70
        },
        {
            'name': 'Neutral Sentiment + Marginal Prediction',
            'base_price_change': 0.018,  # 1.8% predicted gain (below threshold)
            'sentiment_score': 0.1,      # Slightly positive sentiment
            'sentiment_confidence': 0.6,
            'model_confidence': 0.68
        }
    ]
    
    print("=" * 80)
    print("SENTIMENT BIAS TRADING DECISION EXAMPLES")
    print("=" * 80)
    print()
    
    for scenario in scenarios:
        print(f"📊 Scenario: {scenario['name']}")
        print("-" * 60)
        
        # Extract values
        base_change = scenario['base_price_change']
        sentiment_score = scenario['sentiment_score']
        sentiment_conf = scenario['sentiment_confidence'] 
        model_conf = scenario['model_confidence']
        
        # Calculate sentiment adjustment
        sentiment_adjustment = sentiment_score * config['sentiment_bias_strength'] * sentiment_conf
        adjusted_change = base_change + sentiment_adjustment
        
        # Determine signal
        if model_conf < config['confidence_threshold']:
            signal = 'HOLD (Low Confidence)'
        elif adjusted_change > config['price_change_threshold']:
            signal = 'BUY'
        elif adjusted_change < -config['price_change_threshold']:
            signal = 'SELL'
        else:
            signal = 'HOLD (Below Threshold)'
        
        # Sentiment label
        if sentiment_score > 0.3:
            sentiment_label = 'Bullish'
        elif sentiment_score < -0.3:
            sentiment_label = 'Bearish'
        else:
            sentiment_label = 'Neutral'
        
        print(f"  Model Prediction:     {base_change:+.1%}")
        print(f"  Sentiment:           {sentiment_label} ({sentiment_score:+.2f}, conf: {sentiment_conf:.1%})")
        print(f"  Sentiment Adjustment: {sentiment_adjustment:+.1%}")
        print(f"  Final Expected Return: {adjusted_change:+.1%}")
        print(f"  Model Confidence:     {model_conf:.1%}")
        print(f"  📈 TRADING SIGNAL:     {signal}")
        print()
    
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("• Sentiment provides a bias that can push marginal decisions over thresholds")
    print("• Strong sentiment can override weak model predictions") 
    print("• Sentiment confidence weights the bias strength")
    print("• The bias is capped at 15% of the sentiment score")
    print("• Model confidence must still exceed 65% threshold")
    print()
    print("CONFIGURATION:")
    print(f"• Confidence Threshold: {config['confidence_threshold']:.0%}")
    print(f"• Price Change Threshold: {config['price_change_threshold']:.0%}")
    print(f"• Max Sentiment Bias: {config['sentiment_bias_strength']:.0%}")

if __name__ == "__main__":
    demonstrate_sentiment_bias()