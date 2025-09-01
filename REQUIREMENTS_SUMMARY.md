# Requirements Management Summary

## Consolidated Dependencies

All Python dependencies for the Neural Trade Engine project are now consolidated in `engine/requirements.txt`.

### Installation

**For the entire project:**
```bash
pip install -r engine/requirements.txt
```

**For individual modules (standalone):**
```bash
# TSR Model only
pip install -r engine/tsr_model/requirements.txt

# News Sentiment only  
pip install -r engine/news_sentiment/requirements.txt
```

## New Dependencies Added

### Sentiment Analysis Libraries
- `textblob>=0.17.1` - TextBlob for sentiment analysis
- `vaderSentiment>=3.3.2` - VADER sentiment analysis tool  
- `nltk>=3.8.0` - Natural Language Toolkit

### API Changes
- **Financial Modeling Prep API**: Now used by TSR Model and News Sentiment modules
- **Yahoo Finance**: Kept for legacy support, but new modules use FMP API

## Module-Specific Dependencies

### TSR Model (`engine/tsr_model/`)
- PyTorch for neural networks
- Financial Modeling Prep API for market data
- Visualization libraries (matplotlib, seaborn, plotly)

### News Sentiment (`engine/news_sentiment/`)
- TextBlob and VADER for sentiment analysis
- Financial Modeling Prep API for news data
- NLTK for natural language processing
- Visualization libraries for charts and reports

### Social Media Sentiment (`engine/social_media_sentiment/`)
- Uses dependencies from main requirements.txt
- May require additional social media API libraries (Twitter, Reddit, etc.)

## Environment Variables Required

```bash
# Financial Modeling Prep API (required for TSR and News modules)
export FMP_API_KEY=your_api_key_here

# Optional configuration variables
export NEWS_DEFAULT_LIMIT=50
export NEWS_LOOKBACK_DAYS=7
export SENTIMENT_POSITIVE_THRESHOLD=0.1
export SENTIMENT_NEGATIVE_THRESHOLD=-0.1
```

## Development Tools Included

- **Testing**: pytest, pytest-cov
- **Code Quality**: black, flake8, mypy, pre-commit
- **Jupyter**: jupyter, ipykernel, ipywidgets
- **Configuration**: python-dotenv, pyyaml

## Installation Verification

Test that key dependencies are working:

```bash
# Test core ML libraries
python -c "import torch, pandas, numpy, sklearn; print('✓ Core ML libraries OK')"

# Test sentiment analysis
python -c "import textblob, vaderSentiment, nltk; print('✓ Sentiment analysis libraries OK')"

# Test visualization
python -c "import matplotlib, seaborn, plotly; print('✓ Visualization libraries OK')"
```

## Troubleshooting

**NLTK Data Missing:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**PyTorch Issues:**
- Check CUDA compatibility if using GPU
- Visit [pytorch.org](https://pytorch.org) for platform-specific installation

**API Key Issues:**
- Get free Financial Modeling Prep API key: https://financialmodelingprep.com/
- 250 requests/day on free tier
- Set `FMP_API_KEY` environment variable