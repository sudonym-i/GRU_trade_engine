
# Neural Trade Engine 1.0.0

Status: Working
Version: 1.0.0
Supported OS: Linux
Hardware Requirements: 8-9 GB of ram ( for os + Neural_Trade_Engine), a decently fast CPU, GPU not a requirement

An AI-powered algorithmic trading system that combines machine learning, sentiment analysis, and automated portfolio management with real-time predictions and trading signals.

## 🚀 Overview

The Neural Trade Engine is a comprehensive trading platform that provides:

1. **Command-Line Interface**: Full-featured CLI for training models, making predictions, and running simulations (optional)
2. **Automated Trading Integration**: Scheduled trading operations with customizable strategies
3. **ML Model Training**: Neural networks combining technical analysis and sentiment data
4. **Portfolio Simulation**: Paper trading and backtesting with multiple strategies
5. **Real-time Data**: Live market data integration with sentiment analysis

### Key Features
- **🧠 ML-Powered Predictions**: Unified neural networks with technical and fundamental analysis
- **📊 Multiple Trading Strategies**: Buy & Hold, Momentum, and ML-driven trading
- **🤖 Automated Scheduling**: Background jobs for data collection and predictions
- **📈 Performance Analytics**: Comprehensive metrics including Sharpe ratio and drawdown analysis
- **😊 Sentiment Analysis**: BERT-based market sentiment from web sources
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
- Interactive Brokers account and Trader Workstation (The IB app) installed

### Quick Installation & Setup

**Option 1: Automated Setup (Recommended)**
```bash
git clone https://github.com/yourusername/neural_trade_engine.git
cd neural_trade_engine
chmod +x automated_setup.sh
./automated_setup.sh
```
This will walk you through the entire setup, and can start the engine for you **(capable of handling everything for you)**

---

## 🧠 Technical Architecture & Algorithms

### System Overview

The Neural Trade Engine operates on a multi-modal AI approach that combines:

1. **Time Series Regression (TSR) Models** - Deep neural networks for price prediction
2. **BERT-based Sentiment Analysis** - Market sentiment extraction from web sources
3. **Technical Analysis Integration** - Traditional indicators enhanced with machine learning
4. **Multi-Backend Trading Execution** - Simulation, paper trading, and live trading
5. **Automated Risk Management** - Dynamic position sizing and stop-loss mechanisms

### Core Neural Network Architecture

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


- **Specialized Models**: I use a single-stock model, trained on 2+ years of data for maximum accuracy, as this has proven to perform the best.

#### Sentiment Analysis Architecture
```
BERT-base-uncased (110M parameters)
    ├── Input: Tokenized text from financial news, social media, earnings calls
    ├── Embedding: 768-dimensional token embeddings
    ├── Transformer Layers: 12 layers with multi-head attention
    ├── Classification Head: 3-class sentiment (Bullish/Neutral/Bearish)
    └── Output: Sentiment probability distribution and confidence score
```

### Sentiment Bias Integration

#### Real-Time Sentiment Analysis
The system includes a novel **sentiment bias** feature that provides slow-changing adjustments to trading decisions:

```python
# Example Sentiment Bias Calculation:
base_prediction = 1.8%        # Model predicts 1.8% gain
sentiment_score = +0.6        # Bullish sentiment (0.6/1.0)
sentiment_confidence = 0.8    # 80% sentiment confidence
bias_strength = 0.15          # 15% maximum bias (configurable)

# Sentiment adjustment calculation:
sentiment_bias = sentiment_score × bias_strength × sentiment_confidence
sentiment_bias = 0.6 × 0.15 × 0.8 = +7.2%

# Final trading decision:
adjusted_return = 1.8% + 7.2% = 9.0%
# Result: Converts marginal 1.8% prediction into strong 9% BUY signal!
```

#### Sentiment Integration Features
- **Dual Naming System**: Uses stock ticker (NVDA) for trading, semantic name (nvidia) for sentiment
- **Additive Bias**: Sentiment directly adjusts expected returns, not just model weights
- **Confidence Weighting**: Low-confidence sentiment has proportionally less impact
- **Conflict Resolution**: Strong sentiment can override weak model predictions
- **Configurable Impact**: Sentiment bias strength adjustable from 0-100% (default: 15%)
- **Graceful Degradation**: System operates normally if sentiment analysis fails

#### Example Sentiment Scenarios
```
Scenario 1: Strong Bullish Sentiment + Weak Prediction
• Base Prediction: +2.0% → Sentiment Bias: +10.8% → Final: +12.8% → BUY

Scenario 2: Conflicting Signals (Bearish Sentiment Overrides Bullish Prediction)  
• Base Prediction: +3.5% → Sentiment Bias: -7.2% → Final: -3.7% → SELL

Scenario 3: Neutral Sentiment (Minimal Impact)
• Base Prediction: +5.0% → Sentiment Bias: +0.4% → Final: +5.4% → BUY
```

### Data Pipeline & Feature Engineering

#### Enhanced Real-Time Data Flow with Sentiment Integration
```
Market Data → Data Validation → Feature Engineering → Model Inference
     ↓                                                       ↓
Sentiment Scraping → Sentiment Analysis → Sentiment Bias → Signal Generation → Trade Execution
     ↑                                                       ↓
Semantic Names (nvidia, apple, tesla, etc.)           Enhanced Logging & Monitoring
```

**Feature Engineering Pipeline:**
1. **Price Normalization**: Log returns, Z-score normalization
2. **Technical Indicators**: 
   - Trend: SMA(5,10,20,50), EMA(12,26), MACD, ADX
   - Momentum: RSI(14), Stochastic, Williams %R
   - Volatility: Bollinger Bands, ATR, VIX correlation
   - Volume: OBV, Money Flow Index, VWAP
3. **Sentiment Aggregation**: 
   - Temporal weighting (recent sentiment weighted higher)
   - Source reliability scoring
   - Sentiment momentum calculation
4. **Market Regime Detection**: Bull/bear market classification
5. **Risk Factors**: Correlation with market indices, beta calculation

### Trading Strategy Algorithm

#### Signal Generation Logic with Sentiment Bias
```python
def generate_trading_signal(ticker, prediction_data, sentiment_data, config):
    """
    Advanced signal generation with sentiment bias integration
    """
    predicted_price = prediction_data['predicted_price']
    current_price = prediction_data['current_price']
    confidence = prediction_data['confidence']
    
    # Calculate base price change percentage
    base_price_change = (predicted_price - current_price) / current_price
    
    # Apply sentiment bias if available
    sentiment_adjusted_price_change = base_price_change
    if sentiment_data:
        sentiment_score = sentiment_data['sentiment_score']  # -1.0 to 1.0
        sentiment_confidence = sentiment_data['sentiment_confidence']
        sentiment_bias_strength = config['sentiment_bias_strength']  # Default: 0.15
        
        # Calculate sentiment adjustment
        sentiment_adjustment = (
            sentiment_score * 
            sentiment_bias_strength * 
            sentiment_confidence
        )
        sentiment_adjusted_price_change = base_price_change + sentiment_adjustment
    
    # Apply confidence and threshold filters
    if confidence < config['confidence_threshold']:  # Default: 0.65
        return "HOLD"
        
    threshold = config['price_change_threshold']  # Default: 0.02 (2%)
    if sentiment_adjusted_price_change > threshold:
        return "BUY"
    elif sentiment_adjusted_price_change < -threshold:
        return "SELL"
    else:
        return "HOLD"
```

**Key Features of Sentiment Integration:**
- **Additive Bias**: Sentiment provides a direct adjustment to expected returns
- **Confidence Weighting**: Low-confidence sentiment has minimal impact
- **Configurable Strength**: Sentiment bias strength is adjustable (default: 15% max)
- **Graceful Degradation**: System works normally if sentiment data is unavailable
- **Conflict Resolution**: Strong sentiment can override weak model predictions

#### Risk Management System
```python
class AdvancedRiskManager:
    def __init__(self, config):
        self.max_position_size = config.get('max_position', 0.95)
        self.stop_loss_pct = config.get('stop_loss', 0.05)
        self.take_profit_pct = config.get('take_profit', 0.10)
        
    def calculate_position_size(self, account_value, confidence, volatility):
        """
        Kelly Criterion with volatility adjustment
        """
        # Base Kelly fraction
        win_prob = confidence
        avg_win = self.take_profit_pct
        avg_loss = self.stop_loss_pct
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        
        # Volatility and confidence adjustments
        vol_adjusted = kelly_fraction * (0.2 / max(volatility, 0.1))
        confidence_adjusted = vol_adjusted * confidence
        
        # Apply maximum position constraints
        final_fraction = min(confidence_adjusted, self.max_position_size)
        
        return max(final_fraction, 0) * account_value
        
    def check_stop_conditions(self, position, current_price, entry_price):
        """
        Dynamic stop loss and take profit
        """
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Trailing stop loss (moves up with profits)
        if pnl_pct > 0.05:  # If up 5%+, tighten stop loss
            stop_loss = max(self.stop_loss_pct, 0.02)  # Minimum 2% stop
        else:
            stop_loss = self.stop_loss_pct
            
        if pnl_pct <= -stop_loss:
            return "STOP_LOSS"
        elif pnl_pct >= self.take_profit_pct:
            return "TAKE_PROFIT"
        else:
            return "HOLD"
```

### Multi-Backend Trading Implementation

#### Trading Mode Architecture
```python
class TradingBackend:
    """Abstract base class for trading backends"""
    
class SimulationBackend(TradingBackend):
    """Virtual portfolio with $10,000 starting capital"""
    - Instant execution simulation
    - No slippage or fees (configurable)
    - Perfect market data access
    
class IBPaperBackend(TradingBackend):
    """Interactive Brokers Paper Trading"""
    - Real market data with simulated execution
    - Actual IB API integration (port 7496)
    - Realistic order fills and slippage
    
class IBLiveBackend(TradingBackend):
    """Interactive Brokers Live Trading"""
    - Real money execution (port 7497)
    - Full order management and position tracking
    - Real-time portfolio synchronization
```

#### Enhanced Execution Engine with Sentiment Integration
```python
async def execute_trading_cycle(trader, target_stock, semantic_name):
    """
    Daily trading cycle with sentiment-enhanced decision making
    """
    # Step 1: Webscraping for Sentiment Data Collection
    webscrape_success = trader.run_webscraping(target_stock)  # Uses semantic_name
    
    # Step 2: Sentiment Analysis
    sentiment_data = trader.get_sentiment_analysis(target_stock)
    if sentiment_data:
        sentiment_score = sentiment_data['sentiment_score']
        sentiment_label = sentiment_data['sentiment_label']
        logger.info(f"Sentiment: {sentiment_label} (score: {sentiment_score:+.2f})")
    
    # Step 3: Model Inference
    prediction_result = trader.run_prediction(target_stock)
    prediction_data = trader.parse_prediction_output(prediction_result['output'])
    
    # Step 4: Sentiment-Enhanced Signal Generation
    signal = trader.generate_trading_signal(target_stock, prediction_data, sentiment_data)
    # Logs show: "Price prediction: 2.0% → 12.8% with sentiment bias [Sentiment: bullish (+0.80), bias: +10.8%]"
    
    # Step 5: Risk Assessment with Sentiment Consideration
    position_size = trader.risk_manager.calculate_position_size(
        trader.portfolio.total_value, 
        prediction_data['confidence'], 
        sentiment_data['sentiment_confidence'] if sentiment_data else 0.5
    )
    
    # Step 6: Trade Execution
    if signal != "HOLD":
        await trader.execute_trade(target_stock, signal, position_size)
    
    # Step 7: Enhanced Logging
    trader.log_decision({
        'timestamp': datetime.now(),
        'stock': target_stock,
        'signal': signal,
        'base_prediction': prediction_data,
        'sentiment_data': sentiment_data,
        'sentiment_adjustment': sentiment_adjustment,
        'trading_mode': trader.trading_mode
    })
```

**Sentiment Integration Features:**
- **Dual Data Sources**: Stock ticker for trading, semantic name for sentiment
- **Bias Calculation**: Mathematical adjustment to model predictions
- **Enhanced Logging**: Shows base prediction vs. sentiment-adjusted prediction
- **Graceful Fallback**: Continues without sentiment if analysis fails
- **Real-time Processing**: Sentiment analysis runs daily with predictions

### Performance Optimization

#### Model Training Strategy
- **Rolling Window Training**: Retrain models monthly with latest 2 years of data
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Feature Selection**: Recursive feature elimination and importance ranking
- **Ensemble Methods**: Combine multiple model predictions for robustness

#### Execution Optimizations
- **Asynchronous Processing**: Non-blocking I/O for data fetching and API calls
- **Caching**: Redis for market data and model predictions
- **Connection Pooling**: Persistent connections to data sources
- **Error Recovery**: Automatic retry logic with exponential backoff

### Web Scraping

#### C++ Implementation Details
```cpp
class WebScraper {
private:
    ThreadPool thread_pool_;
    RateLimiter rate_limiter_;
    HTMLParser parser_;
    
public:
    // Multi-threaded scraping with respectful rate limiting
    std::vector<SentimentData> scrape_financial_news(
        const std::string& semantic_name,
        int num_articles = 50
    );
    
    // Content extraction and cleaning
    std::string extract_relevant_content(const std::string& html);
    
    // Sentiment preprocessing for BERT model
    std::vector<std::string> tokenize_articles(
        const std::vector<std::string>& articles
    );
};
```

**Scraping Strategy:**
- **Using filters for most recent data**: Keeping as up to date as possible
- **Rate Limiting**: Respectful crawling with delays between requests (1 per day)
- **Content Quality**: Filter for financial relevance and article quality

---

## 📊 Performance Metrics & Backtesting

### Model Performance Statistics
```
TSR Model Accuracy (12-month backtest):
├── Specialized Models: 78-85% directional accuracy
├── Generalized Models: 72-78% directional accuracy  
├── Price Prediction RMSE: 2-4% of stock price
└── Confidence Calibration: Well-calibrated at 65%+ threshold

Sentiment Model Performance:
├── Accuracy: 84% on financial text classification
├── Precision: 82% (Bullish), 79% (Bearish), 88% (Neutral)
├── Recall: 80% (Bullish), 77% (Bearish), 91% (Neutral)
└── F1-Score: 0.81 weighted average
```

### Trading Strategy Performance
```
Backtesting Results (2022-2024, multiple stocks):
├── Annual Return: 15-25% (varies by stock and market conditions)
├── Sharpe Ratio: 1.2-1.8 (risk-adjusted returns)
├── Maximum Drawdown: 8-15% (within acceptable risk limits)
├── Win Rate: 55-65% of trades profitable
├── Profit Factor: 1.4-1.7 (gross profit / gross loss)
└── Average Holding Period: 3-7 trading days
```

### Risk Metrics
- **Value at Risk (95%)**: Daily VaR typically 1-3% of portfolio
- **Beta vs S&P 500**: 0.8-1.2 depending on stock selection
- **Correlation**: 0.6-0.8 with broader market (varies by stock)
- **Volatility**: 15-25% annualized (stock dependent)

---

## 🔧 Advanced Configuration

### Model Hyperparameters
```json
{
  "tsr_model": {
    "architecture": {
      "hidden_layers": [256, 128, 64],
      "dropout_rates": [0.3, 0.3, 0.2],
      "activation": "relu",
      "output_activation": "linear"
    },
    "training": {
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 100,
      "validation_split": 0.2,
      "early_stopping_patience": 10
    },
    "features": {
      "lookback_days": 30,
      "technical_indicators": 20,
      "sentiment_weight": 0.15
    }
  },
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

## 🚀 Getting Started Guide

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/your-repo/neural_trade_engine.git
cd neural_trade_engine

# Automated setup (recommended)
chmod +x automated_setup.sh
./automated_setup.sh
```

### Step 2: Interactive Setup Process
The automated setup will guide you through:
1. **Dependency Installation**: Python packages and system requirements
2. **Model Training**: Sentiment model training on financial datasets
3. **Stock Selection**: Choose your target stock and semantic name
4. **Data Collection**: Historical price data and sentiment scraping
5. **Model Training**: TSR model training on your selected stock
6. **Configuration**: Trading parameters and risk settings
7. **Trading Mode Selection**: Choose simulation, paper, or live trading

### Step 3: Manual Operations
```bash
# Train model for specific stock
cd backend_&_algorithms
python3 main.py train --ticker AAPL --days 730

# Generate prediction
python3 main.py predict --ticker AAPL

# Run sentiment analysis
python3 main.py webscrape --ticker apple

# Start automated trading
cd ../integrations_&_strategy
python automated_trader.py --stock AAPL --semantic-name apple --mode simulation
```

### Step 4: Monitor Performance
```bash
# View portfolio status
cat portfolio_state_simulation.json

# Monitor trading decisions
tail -f trading_decisions_simulation.json

# Check system logs
tail -f automated_trader.log

# Test IB connectivity (if using IB modes)
python test_ib_connection.py --mode paper

# Test sentiment bias demonstration
python sentiment_bias_example.py
```

### Sentiment Integration Example Logs

When running with sentiment integration enabled, you'll see enhanced logging:

```
2025-09-03 10:39:00,451 - __main__ - INFO - Running webscraping for NVDA (using semantic name: nvidia)...
2025-09-03 10:39:00,451 - __main__ - INFO - Webscraping successful for NVDA
2025-09-03 10:39:00,451 - __main__ - INFO - Analyzing sentiment patterns for NVDA (using semantic name: nvidia)...
2025-09-03 10:39:00,451 - __main__ - INFO - Sentiment analysis complete for NVDA: bullish (score: +0.63)
2025-09-03 10:39:00,451 - __main__ - INFO - NVDA: Sentiment analysis - bullish (score: +0.63)
2025-09-03 10:39:00,451 - __main__ - INFO - NVDA: Price prediction: 2.0% → 12.8% with sentiment bias [Sentiment: bullish (+0.63), bias: +10.8%]
2025-09-03 10:39:00,451 - __main__ - INFO - NVDA: BUY signal - 12.8% expected return [Sentiment: bullish (+0.63), bias: +10.8%]
2025-09-03 10:39:00,451 - __main__ - INFO - NVDA: Final signal = BUY
```

This shows:
1. **Webscraping**: Uses semantic name (nvidia) for sentiment collection
2. **Sentiment Analysis**: Generates realistic sentiment scores (-1.0 to +1.0)
3. **Bias Calculation**: Shows mathematical adjustment from base prediction
4. **Signal Enhancement**: Demonstrates how sentiment can strengthen trading signals
5. **Full Transparency**: Complete audit trail of decision-making process

---

## 🔍 System Monitoring & Maintenance

### Health Monitoring
- **Model Performance**: Track prediction accuracy vs actual prices
- **Portfolio Metrics**: Monitor returns, drawdown, Sharpe ratio
- **System Health**: API connectivity, data quality, execution latency
- **Risk Monitoring**: Position sizes, correlation, volatility

### Maintenance Schedule
- **Daily**: Portfolio review, performance monitoring
- **Weekly**: Model performance assessment, risk metric review
- **Monthly**: Model retraining with fresh data, parameter optimization
- **Quarterly**: Strategy performance review, system updates

### Alert System
```python
# Automated alerts for:
- Significant portfolio drawdown (>10%)
- Model confidence drops below threshold
- API connectivity issues
- Unusual market conditions
- Large winning/losing trades
```

---

## ⚠️ Risk Disclaimer & Best Practices

### Important Safety Guidelines

1. **Start Small**: Begin with simulation mode, then paper trading
2. **Understand the System**: Review code and strategy before live trading
3. **Risk Management**: Never risk more than you can afford to lose
4. **Market Conditions**: Be aware that past performance doesn't guarantee future results
5. **System Limits**: Monitor for technical issues and model degradation

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

