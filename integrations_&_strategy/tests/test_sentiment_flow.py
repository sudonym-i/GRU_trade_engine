#!/usr/bin/env python3
"""
Test script to validate sentiment bias calculations in signal generation.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from automated_trader import AutomatedTrader

def test_sentiment_bias_calculations():
    """Test the sentiment bias logic with mock data."""
    
    print("=" * 80)
    print("TESTING SENTIMENT BIAS CALCULATIONS")
    print("=" * 80)
    
    # Create trader instance
    trader = AutomatedTrader(trading_mode='simulation')
    
    # Test scenarios with different sentiment and prediction combinations
    test_cases = [
        {
            'name': 'Strong Bullish Sentiment + Weak Prediction',
            'prediction_data': {
                'predicted_price': 102.0,
                'current_price': 100.0,  # 2% gain prediction
                'confidence': 0.70
            },
            'sentiment_data': {
                'sentiment_score': 0.8,  # Strong bullish
                'sentiment_confidence': 0.9,
                'sentiment_label': 'bullish'
            }
        },
        {
            'name': 'Strong Bearish Sentiment + Weak Prediction',
            'prediction_data': {
                'predicted_price': 98.5,
                'current_price': 100.0,  # -1.5% loss prediction
                'confidence': 0.72
            },
            'sentiment_data': {
                'sentiment_score': -0.7,  # Strong bearish
                'sentiment_confidence': 0.85,
                'sentiment_label': 'bearish'
            }
        },
        {
            'name': 'Neutral Sentiment + Strong Prediction',
            'prediction_data': {
                'predicted_price': 105.0,
                'current_price': 100.0,  # 5% gain prediction
                'confidence': 0.80
            },
            'sentiment_data': {
                'sentiment_score': 0.05,  # Nearly neutral
                'sentiment_confidence': 0.6,
                'sentiment_label': 'neutral'
            }
        },
        {
            'name': 'Conflicting Signals: Bearish Sentiment + Bullish Prediction',
            'prediction_data': {
                'predicted_price': 103.5,
                'current_price': 100.0,  # 3.5% gain prediction
                'confidence': 0.75
            },
            'sentiment_data': {
                'sentiment_score': -0.6,  # Bearish sentiment
                'sentiment_confidence': 0.8,
                'sentiment_label': 'bearish'
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 TEST CASE {i}: {test_case['name']}")
        print("-" * 60)
        
        # Calculate expected values manually
        pred_data = test_case['prediction_data']
        sent_data = test_case['sentiment_data']
        
        base_change = (pred_data['predicted_price'] - pred_data['current_price']) / pred_data['current_price']
        sentiment_adjustment = sent_data['sentiment_score'] * trader.config['sentiment_bias_strength'] * sent_data['sentiment_confidence']
        adjusted_change = base_change + sentiment_adjustment
        
        print(f"  Base Prediction:       {base_change:+.1%}")
        print(f"  Sentiment Score:       {sent_data['sentiment_score']:+.2f} ({sent_data['sentiment_label']})")
        print(f"  Sentiment Confidence:  {sent_data['sentiment_confidence']:.1%}")
        print(f"  Sentiment Adjustment:  {sentiment_adjustment:+.1%}")
        print(f"  Final Expected Return: {adjusted_change:+.1%}")
        print(f"  Model Confidence:      {pred_data['confidence']:.1%}")
        
        # Generate signal using the trader's method
        signal = trader.generate_trading_signal("TEST", pred_data, sent_data)
        print(f"  🎯 FINAL SIGNAL:       {signal}")
        
        # Validate the logic
        threshold = trader.config['price_change_threshold']
        if pred_data['confidence'] < trader.config['confidence_threshold']:
            expected_signal = 'HOLD'
        elif adjusted_change > threshold:
            expected_signal = 'BUY'
        elif adjusted_change < -threshold:
            expected_signal = 'SELL'
        else:
            expected_signal = 'HOLD'
        
        if signal == expected_signal:
            print(f"  ✅ Signal matches expected logic")
        else:
            print(f"  ❌ Signal mismatch! Expected: {expected_signal}")
    
    print("\n" + "=" * 80)
    print("CONFIGURATION SETTINGS:")
    print("=" * 80)
    print(f"Confidence Threshold:     {trader.config['confidence_threshold']:.0%}")
    print(f"Price Change Threshold:   {trader.config['price_change_threshold']:.0%}")
    print(f"Sentiment Bias Strength:  {trader.config['sentiment_bias_strength']:.0%}")

if __name__ == "__main__":
    test_sentiment_bias_calculations()