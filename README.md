
# Neural Trade Engine

**Status:** Needs fixing

**Version:** 1.0.0

**Supported OS:** Linux

**Hardware Requirements:** 8-9 GB of ram ( for OS + Neural_Trade_Engine), a decently fast CPU, GPU not a requirement

## Overview

The Neural Trade Engine is a comprehensive trading platform that provides:

- **🧠 ML-Powered Predictions**: Main predictive power cmes from the Time Series Regression model trained on stock-price patterns
- **📊 Multiple Trading Strategies**: Buy & Hold, Momentum, and ML-driven trading
- **🤖 Automated Scheduling**: Background jobs for data collection and predictions
- **📈 Performance Analytics**: Comprehensive metrics including Sharpe ratio and drawdown analysis
- **😊 Sentiment Analysis**: BERT-based market sentiment from web sources (YouTube)
- **💻 Interactive Trading**: Paper trading interface with real-time market data

## 📁 Project Structure

```
neural_trade_engine/
├── README.md                        # Project documentation
├── backend_&_algorithms/            # Core ML and trading engine
│   ├── main.py                      # CLI application entry point
│   ├── config.json                  # Configuration settings
│   ├── install.sh                   # Installation script
│   ├── models/                      # Trained model storage
│   ├── processed_data/              # Processed training data
│   ├── saved_models/                # Saved model checkpoints
│   ├── testing/                     # Test files and scripts
│   ├── engine/                      # Core engine package
│   │   ├── __init__.py              # Package initialization
│   │   ├── requirements.txt         # Engine dependencies
│   │   ├── setup.py                 # Package setup
│   │   ├── tsr_model/               # Time Series Regression models
│   │   │   ├── api.py               # High-level API functions
│   │   │   ├── models.py            # Neural network architectures
│   │   │   ├── training.py          # Model training pipeline
│   │   │   ├── data_pipelines/      # Data processing pipelines
│   │   │   │   ├── __init__.py      # Pipeline package
│   │   │   │   ├── price_data.py    # Price & technical data
│   │   │   │   ├── stock_data_sources.py # Data source utilities
│   │   │   │   └── tsr_pipeline.py  # TSR-specific pipeline
│   │   │   └── __init__.py          # TSR model package
│   │   └── sentiment_model/         # Sentiment analysis
│   │       ├── api.py               # Sentiment API functions
│   │       ├── model.py             # BERT-based sentiment model
│   │       ├── tokenize_pipeline.py # Text processing pipeline
│   │       ├── train_with_labeled_data.py # Training script
│   │       ├── download_dataset.py  # Dataset utilities
│   │       ├── processed_data/      # Tokenized training data
│   │       ├── saved_models/        # Trained sentiment models
│   │       ├── web_scraper/         # C++ web scraping tools
│   │       │   ├── CMakeLists.txt   # Build configuration
│   │       │   ├── README.md        # Scraper documentation
│   │       │   └── tests/           # Unit tests
│   │       └── __init__.py          # Sentiment model package
│   └── performance_data/            # Model performance logs
│       ├── INFO.md                  # Performance documentation
│       ├── generalizing_tsr_model/  # Generalized model results
│       └── specialized_tsr_model/   # Specialized model results
├── integrations_&_strategy/         # Trading automation and integration
│   ├── automated_trader.py          # Main automated trading script
│   ├── schedule_trader.py           # Scheduling and automation
│   ├── config.json                  # Trading configuration
│   ├── requirements.txt             # Integration dependencies
│   └── README.md                    # Integration documentation
└── automated_setup.sh               # Complete project setup script
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Interactive Brokers account and Trader Workstation (The IB app) installed. <-- This is essentially your stock-market API

### Quick Installation & Setup

**Option 1: Automated Setup (Recommended)**
```bash
git clone https://github.com/yourusername/neural_trade_engine.git
cd neural_trade_engine
chmod +x automated_setup.sh
./automated_setup.sh
```
The automated setup will guide you through:
1. **Dependency Installation**: Python packages and system requirements
2. **Model Training**: Sentiment model training on financial datasets
3. **Stock Selection**: Choose your target stock and semantic name
4. **Data Collection**: Historical price data and sentiment scraping
5. **Model Training**: TSR model training on your selected stock
6. **Configuration**: Trading parameters and risk settings
7. **Trading Mode Selection**: Choose simulation, paper, or live trading

You can use this interface to get your engine actively running on your hardware, no extra steps neccesary.

---

### Architectures

#### Time Series Regression (TSR) Model
```
Input Layer (50-100 features)
    ├── Price Features: OHLCV data, returns, volatility measures
    ├── Technical Indicators: Moving averages, RSI, MACD, Bollinger Bands
    ├── Volume Features: Volume patterns, Money Flow Index, OBV
    ├── Sentiment Features: Aggregated sentiment scores and momentum
    └── Market Features: VIX levels, sector performance, market breadth

Hidden Layers (3-5 layers)
    ├── Dense Layer 1: 256 neurons, ReLU activation, 30% dropout
    ├── Dense Layer 2: 128 neurons, ReLU activation, 30% dropout  
    ├── Dense Layer 3: 64 neurons, ReLU activation, 20% dropout
    └── Optional LSTM Layer: 32 cells for temporal sequence learning

Output Layer
    ├── Price Prediction: Next day closing price (regression)
    ├── Confidence Score: Model certainty (0-1 probability)
    └── Risk Estimate: Predicted volatility for position sizing
```


- **Models are Specialized**: I use a single-stock model, trained on 2+ years of data for maximum accuracy, as this has proven to perform the best.

#### Sentiment Analysis Architecture
```
BERT-base-uncased (110M parameters)
    ├── Input: Tokenized text from financial news, social media, earnings calls
    ├── Embedding: 768-dimensional token embeddings
    ├── Transformer Layers: 12 layers with multi-head attention
    ├── Classification Head: 3-class sentiment (Bullish/Neutral/Bearish)
    └── Output: Sentiment probability distribution and confidence score
```


### Data Pipeline & Feature Engineering

#### Enhanced Real-Time Data Flow with Sentiment Integration
```
Market Data  →  Data cleaning & normalization   →   Model Inference
                                                                ↓
Sentiment Scraping →  Sentiment Analysis →  Sentiment Bias → Signal Generation → Trade Execution
     ↑                                                            ↓
Semantic Names (nvidia, apple, tesla, etc.)                Logging & Monitoring
```

**Feature Engineering Pipeline:**
1. **Price Normalization**: Log returns, Z-score normalization
2. **Technical Indicators**: 
   - Trend: SMA(5,10,20,50), EMA(12,26), MACD, ADX
   - Momentum: RSI(14), Stochastic, Williams %R
   - Volatility: Bollinger Bands, ATR, VIX correlation
   - Volume: OBV, Money Flow Index, VWAP
3. **Sentiment Aggregation**: 
   - Temporal sampling (collecting data filtered to be recent)
   - Source reliability scoring
   - Sentiment momentum calculation
4. **Market Regime Detection**: Bull/bear market classification
5. **Risk Factors**: Correlation with market indices, beta calculation

## 🔧 Advanced Configuration

### Model Hyperparameters

```json
{
  
  "sentiment_model": {
    "model_name": "bert-base-uncased",
    "max_sequence_length": 512,
    "learning_rate": 2e-5,
    "warmup_steps": 1000,
    "num_train_epochs": 3
  }
}

```

### Trading Parameters
```json
{
  "risk_management": {
    "max_position_size": 0.95,
    "stop_loss_percent": 0.05,
    "take_profit_percent": 0.10,
    "confidence_threshold": 0.65,
    "minimum_return_threshold": 0.02,
    "sentiment_bias_strength": 0.15
  },
  "execution": {
    "order_type": "market",
    "max_slippage": 0.001,
    "position_sizing_method": "kelly_criterion",
    "rebalance_frequency": "daily"
  },
  "data_sources": {
    "price_data": "interactive_brokers",
    "sentiment_sources": ["financial_news", "social_media"],
    "update_frequency": "1h"
  }
}
```

---

## ⚠️ Risk Disclaimer & Best Practices

### Known Limitations
- **Market Regime Changes**: Models may underperform during unusual market conditions
- **Black Swan Events**: System cannot predict extreme, unpredictable events
- **Data Dependencies**: Performance relies on quality of input data
- **Execution Risk**: Slippage and latency can impact theoretical performance

---

### Contributing
Contributions are welcome! Please see CONTRIBUTING.md for guidelines on:
- Code style and standards
- Testing requirements  
- Pull request process
- Bug reporting

---

**⚠️ FINAL DISCLAIMER: This software is for educational and research purposes. Trading involves substantial risk of loss. The authors assume no responsibility for financial losses. Always conduct thorough testing and consider your risk tolerance before live trading.**

