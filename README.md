
# Neural Trade Advisory Engine

**Status:** Active Development

**Version:** 0.0.4 | Advisory System

**Supported OS:** Linux

**Hardware Requirements:**  | ~8 GB of ram  |  ~12 GB storage,  |  a decently fast CPU  |  GPU not a requirement

## Overview
<img width="1884" height="1060" alt="Screenshot From 2025-09-07 12-48-11" src="https://github.com/user-attachments/assets/8d1e0ea6-ccc4-417f-a2a4-8bc4a8fef0e8" />

This project provides intelligent trading recommendations using advanced machine learning models. The system has evolved from automated trading to a **advisory-focused approach** that empowers users with data-driven insights while maintaining full control over trading decisions.

Key Features:
- **🧠 ML-Powered Predictions**: Time Series Regression model with GRU architecture trained on stock-price patterns
- **💬 Smart Advisory Messages**: Automated recommendations advising specific buy/sell actions with reasoning
- **🤖 Automated Scheduling**: Background jobs for continuous data collection and prediction updates
- **📈 Performance Analytics**: Comprehensive metrics, prediction history, and accuracy tracking
- **😊 Sentiment Analysis**: BERT-based market sentiment from daily web sources (YouTube transcripts - filtered to most recent)

## 📁 Project Structure

```
TSR_trade_engine/
├── README.md                        # Project documentation
├── interact.sh                      # Interactive setup and management script
├── backend_&_algorithms/            # Core ML and prediction engine
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
├── integrations/                    # Advisory system and scheduling
│   ├── scheduler.py                 # Automated prediction scheduler
│   ├── config.json                  # Scheduler configuration
│   └── scheduler.log                # Scheduler activity logs
└── .claude/                         # Claude Code project configuration
    └── CLAUDE.md                    # Development instructions and context
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Internet connection for data fetching and sentiment analysis
- ~8 GB RAM for ML model training and inference

### Quick Installation & Setup

**Automated Setup (Recommended)**
```bash
git clone https://github.com/yourusername/neural_trade_advisory_engine.git
cd TSR_trade_engine
chmod +x interact.sh
./interact.sh
```

The automated setup will guide you through:
1. **Dependency Installation**: Python packages, system requirements, and virtual environment setup
2. **Sentiment Model Training**: Train BERT-based sentiment model on labeled financial data
3. **Stock Selection**: Choose your target stock and configure semantic name for web scraping
4. **Data Collection**: Fetch historical price data and configure sentiment data sources
5. **Model Training**: Train TSR/GRU model on your selected stock's historical patterns
6. **Advisory Configuration**: Set prediction intervals and notification preferences
7. **Scheduler Setup**: Configure automated prediction schedules and advisory message delivery

The system will then run continuously, providing regular trading recommendations based on ML predictions.

## 🔮 How the Advisory System Works

### Advisory Workflow
1. **Scheduled Predictions**: The scheduler runs predictions at configurable intervals (default: every 2 hours)
2. **Data Integration**: Each prediction cycle combines:
   - Latest price data and technical indicators
   - Recent sentiment analysis from web sources
   - Historical pattern recognition from trained models
3. **Smart Recommendations**: Based on prediction confidence and market conditions:
   - **BUY signals**: When predicted price > current price with high confidence
   - **SELL signals**: When predicted price < current price for held positions
   - **HOLD signals**: When confidence is low or predicted price is within normal range
4. **Advisory Messages**: Clear, actionable recommendations with:
   - Specific action (BUY/SELL/HOLD)
   - Confidence level and reasoning
   - Target price and suggested position size
   - Risk assessment and timeline

### Running the Advisory System
```bash
# Start the prediction scheduler
cd integrations/
python scheduler.py

# Manual prediction (for testing)
cd backend_&_algorithms/
python main.py predict --ticker AAPL
```

---

### Architectures

#### Time Series Regression (TSR/GRU) Model
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


- **Models are Specialized**: I use a single-stock model, trained on 2+ years of data for maximum accuracy, as this has proven to perform the best in my testing.

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

#### Advisory System Data Flow
```
Market Data  →  Data cleaning & normalization   →   Model Inference
                                                                ↓
Sentiment Scraping →  Sentiment Analysis →  Sentiment Bias → Advisory Generation → Recommendations
     ↑                                                            ↓
Semantic Names (nvidia, apple, tesla, etc.)                Message Delivery & Logging
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

### Advisory System Configuration
```json
{
  "scheduler": {
    "ticker": "AAPL",
    "interval": "2hr",
    "max_runtime_minutes": 5
  },
  "advisory_settings": {
    "confidence_threshold": 0.65,
    "minimum_return_threshold": 0.02,
    "sentiment_bias_strength": 0.15,
    "position_size_recommendations": true,
    "risk_level": "moderate"
  },
  "recommendations": {
    "include_reasoning": true,
    "include_confidence": true,
    "include_risk_assessment": true,
    "suggested_position_size": 0.05,
    "stop_loss_percent": 0.05,
    "take_profit_percent": 0.10
  },
  "data_sources": {
    "sentiment_sources": ["youtube", "financial_news"],
    "update_frequency": "daily",
    "sentiment_days_lookback": 7
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

**⚠️ IMPORTANT DISCLAIMER: This software provides advisory recommendations only and is for educational and research purposes. All trading decisions remain entirely with the user. Trading involves substantial risk of loss. The authors assume no responsibility for financial losses. Always conduct thorough testing, verify recommendations independently, and consider your risk tolerance before making any trading decisions. Past performance does not guarantee future results.**

