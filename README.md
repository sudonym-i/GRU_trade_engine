
# Neural Trade Engine

An advanced AI-powered stock prediction and trading system that combines technical analysis, financial fundamentals, and sentiment analysis for comprehensive market intelligence.

## 🚀 Overview

The Neural Trade Engine is a comprehensive trading platform that integrates:

- **🧠 Unified ML Models**: Combines price patterns with financial fundamentals
- **📊 Trading Simulation**: Realistic paper trading with real market data
- **😊 Sentiment Analysis**: Market sentiment from news and social media  
- **⚡ Automated Strategies**: Multiple trading algorithms with performance tracking
- **📈 Performance Analytics**: Detailed metrics and backtesting capabilities

## 📁 Project Structure

```
neural_trade_engine/
├── main.py                          # CLI application entry point
├── config.json                      # Configuration settings
├── engine/                          # Core engine package
│   ├── __init__.py                  # Main package exports
│   ├── unified_model/               # Unified prediction models
│   │   ├── integrated_model.py     # Neural network architectures
│   │   ├── train.py                 # Model training pipeline  
│   │   ├── api.py                   # High-level API functions
│   │   ├── models/                  # Trained model storage
│   │   └── data_pipelines/          # Data processing pipelines
│   │       ├── stock_pipeline.py   # Price & technical data
│   │       ├── financial_pipeline.py # Fundamental data
│   │       └── integrated_data_pipeline.py # Combined pipeline
│   ├── trading_simulation/          # Trading simulation engine
│   │   ├── engine.py               # Main trading engine
│   │   ├── portfolio.py            # Portfolio management
│   │   ├── orders.py               # Order execution system
│   │   └── strategies.py           # Trading strategies
│   ├── sentiment_model/             # Sentiment analysis
│   │   ├── model.py                # BERT-based sentiment model
│   │   ├── route.py                # Web scraping & analysis
│   │   ├── processed_data/          # Tokenized training data
│   │   ├── raw_data/                # Raw sentiment data
│   │   └── web_scraper/             # C++ web scraping tools
│   └── requirements.txt             # Engine dependencies
├── frontend_&_integrations/         # Frontend and integration tools
│   ├── dashboard/                   # Web dashboard (planned)
│   ├── google_sheets/              # Google Sheets integration
│   └── message_api/                # Messaging/notification API
└── performance_data/                # Model performance logs
    ├── INFO.md                     # Performance documentation
    ├── generalizing_tsr_model/     # Generalized model results
    └── specialized_tsr_model/      # Specialized model results
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Financial Modeling Prep API key ([Get one here](https://financialmodelingprep.com/developer/docs))

### Environment Setup
```bash
# Set your API key
export FMP_API_KEY=your_api_key_here

# Install dependencies (if not already installed)
pip install torch pandas numpy scikit-learn requests matplotlib
```

## 💻 Command Line Interface

The Neural Trade Engine provides a comprehensive CLI through `main.py` with the following commands:

```bash
python main.py <command> [options]
```

**Available Commands:**
- `train` - Train unified prediction models
- `predict` - Make price predictions using trained models
- `sentiment` - Analyze text sentiment from web sources
- `models` - List all available trained models
- `info` - Show detailed information about a specific model
- `paper-trade` - Start interactive paper trading session
- `simulate` - Run automated trading simulations

---

### 🏋️ `train` - Model Training

Train unified models combining technical and fundamental analysis.

**Usage:**
```bash
python main.py train [options]
```

**Examples:**
```bash
# Basic training (AAPL, 2 years of data)
python main.py train --tickers AAPL

# Multi-stock training with custom parameters
python main.py train --tickers AAPL MSFT GOOGL --days 1095 --epochs 100

# Advanced training with specific architecture
python main.py train \
    --tickers AAPL MSFT TSLA \
    --days 1460 \
    --model-type adaptive \
    --epochs 75 \
    --batch-size 64 \
    --learning-rate 0.001
```

**Options:**
- `--tickers <symbols>`: Stock symbols to train on (space-separated, default: AAPL)
- `--days <int>`: Historical data period in days (default: 730)
- `--model-type <type>`: Architecture type - `standard` or `adaptive` (default: standard)
- `--epochs <int>`: Number of training epochs (default: 50)
- `--batch-size <int>`: Training batch size (default: 32)
- `--learning-rate <float>`: Learning rate for optimizer (default: 0.001)

---

### 🔮 `predict` - Price Prediction

Make stock price predictions using trained models.

**Usage:**
```bash
python main.py predict --ticker <symbol> [options]
```

**Examples:**
```bash
# Basic prediction (uses latest model)
python main.py predict --ticker AAPL

# Prediction with confidence intervals
python main.py predict --ticker AAPL --confidence

# Prediction with specific model
python main.py predict \
    --ticker MSFT \
    --model engine/unified_model/models/unified_standard_model.pth \
    --confidence
```

**Options:**
- `--ticker <symbol>`: Stock symbol to predict (required)
- `--model <path>`: Path to trained model file (uses latest if not specified)
- `--confidence`: Include 95% confidence intervals and uncertainty metrics

---

### 🧠 `sentiment` - Sentiment Analysis

Analyze market sentiment from text or web sources using BERT-based models.

**Usage:**
```bash
python main.py sentiment (--text <text> | --url <url>)
```

**Examples:**
```bash
# Analyze text sentiment
python main.py sentiment --text "The stock market is looking bullish today"

# Extract and analyze web content
python main.py sentiment --url "https://finance.yahoo.com/news/article"
```

**Options:**
- `--text <text>`: Text to analyze for sentiment (mutually exclusive with --url)
- `--url <url>`: Web URL to extract and analyze content (mutually exclusive with --text)

---

### 📋 `models` - Model Management

List all available trained models with their metadata.

**Usage:**
```bash
python main.py models
```

**Output includes:**
- Model filename and type
- Creation timestamp
- Validation loss
- File size
- Training epochs and performance metrics

---

### ℹ️ `info` - Model Information

Show detailed information about a specific trained model.

**Usage:**
```bash
python main.py info --model <path>
```

**Example:**
```bash
python main.py info --model engine/unified_model/models/unified_standard_model.pth
```

**Options:**
- `--model <path>`: Path to model file (required)

**Output includes:**
- Model architecture details
- Training history and performance
- Configuration parameters
- File metadata

---

### 📊 `paper-trade` - Interactive Paper Trading

Start an interactive paper trading session with real market data.

**Usage:**
```bash
python main.py paper-trade [options]
```

**Examples:**
```bash
# Start with default $100k balance
python main.py paper-trade

# Custom balance with portfolio state saving
python main.py paper-trade --balance 50000 --save my_portfolio.json
```

**Options:**
- `--balance <amount>`: Starting balance in USD (default: 100000)
- `--save <filename>`: Save portfolio state to JSON file on exit

**Interactive Commands:**
- `buy <symbol> <quantity> [price]` - Place buy order (market or limit)
- `sell <symbol> <quantity> [price]` - Place sell order (market or limit)
- `portfolio` - Show portfolio summary with P&L
- `positions` - Show current stock positions
- `orders` - Show active orders
- `update` - Refresh market data
- `quit` - Exit trading session

**Examples:**
```
💼 > buy AAPL 100              # Market buy 100 shares of AAPL
💼 > sell MSFT 50 350.50       # Limit sell 50 MSFT at $350.50
💼 > portfolio                 # Show portfolio summary
💼 > quit                      # Exit session
```

---

### 🤖 `simulate` - Automated Trading Simulation

Run automated trading simulations with different strategies and performance tracking.

**Usage:**
```bash
python main.py simulate [options]
```

**Examples:**
```bash
# Buy & Hold strategy simulation
python main.py simulate --tickers AAPL MSFT --strategy buy_hold --days 90

# Momentum strategy with results export
python main.py simulate \
    --tickers AAPL MSFT GOOGL TSLA \
    --strategy momentum \
    --days 180 \
    --balance 250000 \
    --save simulation_results.json \
    --export-csv trades.csv

# ML-powered trading with trained model
python main.py simulate \
    --tickers AAPL MSFT \
    --strategy ml_prediction \
    --model engine/unified_model/models/unified_standard_model.pth \
    --days 120 \
    --frequency 4

# Real-time simulation (slower but more realistic)
python main.py simulate \
    --tickers AAPL \
    --strategy ml_prediction \
    --model engine/unified_model/models/best_model.pth \
    --days 30 \
    --frequency 1 \
    --realtime
```

**Options:**
- `--tickers <symbols>`: Stock symbols to trade (space-separated, default: AAPL MSFT)
- `--strategy <name>`: Trading strategy to use (default: buy_hold)
  - `buy_hold`: Portfolio rebalancing with equal allocations
  - `momentum`: Trend-following with stop-loss and take-profit
  - `ml_prediction`: ML-driven decisions using trained models
- `--days <int>`: Simulation duration in days (default: 30)
- `--balance <amount>`: Starting balance in USD (default: 100000)
- `--model <path>`: Path to trained model (required for ml_prediction strategy)
- `--frequency <hours>`: Update frequency in hours (default: 6)
- `--realtime`: Run in real-time mode (slower but more realistic)
- `--save <filename>`: Save detailed simulation results to JSON file
- `--export-csv <filename>`: Export trade history to CSV file

**Performance Metrics:**
All simulations provide comprehensive performance analysis:
- Total return (absolute and percentage)
- Win rate and average win/loss per trade
- Maximum drawdown and Sharpe ratio
- Daily portfolio value progression
- Complete trade history and final positions

## 🧠 Model Architecture

### Unified Stock Predictor
The core model combines two data streams:

- **Price Encoder**: GRU network processing OHLCV + technical indicators
- **Fundamental Encoder**: GRU network processing financial ratios & metrics
- **Attention Layer**: Multi-head attention on price patterns
- **Feature Fusion**: Combines both encoders for final prediction
- **Confidence Estimation**: Monte Carlo dropout for uncertainty quantification

### Available Trading Strategies

1. **Buy & Hold** (`buy_hold`): Portfolio rebalancing with equal allocations across selected tickers
2. **Momentum** (`momentum`): Trend-following strategy with configurable stop-loss and take-profit levels
3. **ML Prediction** (`ml_prediction`): Machine learning-driven decisions using trained unified models

## 📊 Data Sources

- **Price Data**: Real-time and historical stock prices via Financial Modeling Prep API
- **Financial Data**: Income statements, balance sheets, financial ratios, and key metrics
- **Market Data**: Live quotes with market hours detection and trading status
- **Sentiment Data**: Web content extraction and BERT-based sentiment analysis

## 🔄 Typical Workflow

Here's a common workflow for using the Neural Trade Engine:

1. **Set API Key**: 
   ```bash
   export FMP_API_KEY=your_api_key_here
   ```

2. **Train Model**: 
   ```bash
   python main.py train --tickers AAPL MSFT --epochs 50
   ```

3. **Test Predictions**: 
   ```bash
   python main.py predict --ticker AAPL --confidence
   ```

4. **Run Simulation**: 
   ```bash
   python main.py simulate --strategy ml_prediction --model engine/unified_model/models/latest.pth
   ```

5. **Analyze Results**: Review exported JSON/CSV files and performance metrics

6. **Interactive Trading**: 
   ```bash
   python main.py paper-trade
   ```

## 📈 Performance Metrics

All trading simulations and paper trading sessions provide comprehensive performance analysis:

- **Total Return**: Absolute dollar returns and percentage gains/losses
- **Win Rate**: Percentage of profitable trades vs. total trades
- **Average Win/Loss**: Average profit and loss amounts per trade
- **Sharpe Ratio**: Risk-adjusted returns calculation (when sufficient data available)
- **Maximum Drawdown**: Largest peak-to-trough portfolio decline
- **Daily Performance**: Time series tracking of portfolio value over time
- **Trade History**: Complete log of all executed trades with timestamps

## 🚨 Risk Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consider consulting with financial professionals before making investment decisions.

## 🛠️ Development

### Adding New Strategies
Extend the `TradingStrategy` class in `engine/paper_trading/strategies.py`

### Custom Model Architectures
Modify `engine/unified_model/integrated_model.py`

### Additional Data Sources
Extend the data pipelines in `engine/unified_model/data_pipelines/`

## 🛠️ Troubleshooting

### Common Issues

**API Key Errors:**
```bash
❌ FMP_API_KEY environment variable not set!
```
- Solution: Set your Financial Modeling Prep API key: `export FMP_API_KEY=your_api_key_here`

**Import Errors:**
```bash
❌ Failed to import engine functions
```
- Solution: Run from project root directory and ensure dependencies are installed

**Model Not Found:**
```bash
❌ No trained models found. Train a model first.
```
- Solution: Train a model first using: `python main.py train --tickers AAPL`

### Getting Help

- **Command Help**: `python main.py --help` or `python main.py <command> --help`
- **Check Logs**: Review error messages and stack traces
- **Verify Setup**: Ensure FMP_API_KEY is set and dependencies are installed
- **Model Status**: Use `python main.py models` to see available trained models

### Requirements

- Python 3.8+
- Financial Modeling Prep API key ([Get one here](https://financialmodelingprep.com/developer/docs))
- Dependencies: `torch pandas numpy scikit-learn requests matplotlib`

---

**Built with ❤️ for algorithmic trading research and education**
