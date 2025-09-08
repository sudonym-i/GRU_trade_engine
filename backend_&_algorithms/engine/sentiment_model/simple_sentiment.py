#!/usr/bin/env python3

import requests
from typing import Dict, List, Optional

def analyze_text_sentiment(text: str) -> Dict[str, float]:
    """
    Simple sentiment analysis using TextBlob.
    
    Returns:
        Dict with sentiment score and label
    """
    try:
        from textblob import TextBlob
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            "sentiment": sentiment,
            "confidence": abs(polarity),
            "score": polarity
        }
    except ImportError:
        return {"sentiment": "neutral", "confidence": 0.5, "score": 0.0}

def get_news_sentiment(ticker: str) -> Dict[str, any]:
    """
    Get news sentiment for a stock ticker.
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        news = stock.news[:5]  # Get latest 5 news items
        
        sentiments = []
        for item in news:
            title = item.get('title', '')
            summary = item.get('summary', '')
            text = f"{title} {summary}"
            
            sentiment = analyze_text_sentiment(text)
            sentiments.append({
                "title": title,
                "sentiment": sentiment["sentiment"],
                "score": sentiment["score"]
            })
        
        # Calculate overall sentiment
        scores = [s["score"] for s in sentiments]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score > 0.1:
            overall = "positive"
        elif avg_score < -0.1:
            overall = "negative"
        else:
            overall = "neutral"
            
        return {
            "ticker": ticker,
            "overall_sentiment": overall,
            "average_score": avg_score,
            "news_count": len(sentiments),
            "individual_news": sentiments
        }
        
    except Exception as e:
        return {
            "ticker": ticker,
            "overall_sentiment": "neutral",
            "average_score": 0.0,
            "news_count": 0,
            "individual_news": [],
            "error": str(e)
        }

def analyze_sentiment(text: str = None, ticker: str = None) -> Dict[str, any]:
    """
    Main sentiment analysis function.
    """
    if ticker:
        return get_news_sentiment(ticker)
    elif text:
        return analyze_text_sentiment(text)
    else:
        return {"error": "No text or ticker provided"}