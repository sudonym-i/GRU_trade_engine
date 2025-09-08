#!/usr/bin/env python3

import logging
from typing import Dict

from .simple_sentiment import analyze_sentiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pull_from_web(ticker: str, output_name: str = None) -> Dict:
    """Skip web scraping, use direct news analysis."""
    logger.info(f"Getting news sentiment for {ticker}")
    return analyze_sentiment(ticker=ticker) 


def analyze_sentiment_text(text: str = None, ticker: str = None) -> Dict:
    """Simple sentiment analysis."""
    if text:
        return analyze_sentiment(text=text)
    elif ticker:
        return analyze_sentiment(ticker=ticker)
    else:
        return {"error": "No text or ticker provided"}


def full_pipeline(ticker: str = "AAPL") -> Dict:
    """Get news sentiment for ticker."""
    logger.info(f"Getting sentiment for {ticker}")
    return analyze_sentiment(ticker=ticker)

