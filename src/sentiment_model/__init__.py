"""
Sentiment Classification Model

This module contains the sentiment classification model implementation
for analyzing financial text sentiment.

Author: ML Trading Bot Project
"""

__version__ = "1.0.0"

# Import model components when available
try:
    from .model import *
except ImportError:
    pass