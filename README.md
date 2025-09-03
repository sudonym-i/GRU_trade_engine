
# Neural Trade Engine

An AI-powered algorithmic trading system that combines machine learning, sentiment analysis, and automated portfolio management with real-time predictions and trading signals.

## 🚀 Overview

The Neural Trade Engine is a comprehensive trading platform that provides:

1. **Command-Line Interface**: Full-featured CLI for training models, making predictions, and running simulations
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
├── .claude/                         # Claude Code configuration
│   ├── CLAUDE.md                    # Development plan and instructions
│   └── settings.local.json          # Local settings
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
- Financial Modeling Prep API key ([Get one here](https://financialmodelingprep.com/developer/docs))

### Quick Installation

**Option 1: Automated Setup (Recommended)**
```bash
git clone https://github.com/yourusername/neural_trade_engine.git
cd neural_trade_engine
chmod +x automated_setup.sh
./automated_setup.sh
```

**Option 2: Manual Setup**
```bash
git clone https://github.com/yourusername/neural_trade_engine.git
cd neural_trade_engine/backend_&_algorithms
chmod +x install.sh
./install.sh

# Set your API key
export FMP_API_KEY=your_api_key_here
```

## 🚀 Quick Start

### Basic Usage

**Train a model:**
```bash
cd backend_&_algorithms
python main.py train --tickers AAPL MSFT --days 730
```

**Make predictions:**
```bash
python main.py predict --ticker AAPL --confidence
```

**Run trading simulation:**
```bash
python main.py simulate --strategy ml_prediction --days 30
```

**Start automated trading:**
```bash
cd ../integrations_&_strategy
python automated_trader.py --stocks AAPL,MSFT,NVDA
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

Here's the typical workflow for automated trading:

### Automated Trading Setup
1. **Configure Trading Parameters**:
   ```bash
   cd integrations_&_strategy
   nano config.json  # Edit trading configuration
   ```

2. **Test Configuration**:
   ```bash
   python automated_trader.py --dry-run
   ```

3. **Schedule Daily Execution**:
   ```bash
   python schedule_trader.py --start
   ```

4. **Monitor Performance**: Check logs and trading decisions in real-time

### Command Line Trading
1. **Set API Key**: 
   ```bash
   export FMP_API_KEY=your_api_key_here
   ```

2. **Train Model**: 
   ```bash
   cd backend_&_algorithms
   python main.py train --tickers AAPL MSFT --epochs 50
   ```

3. **Test Predictions**: 
   ```bash
   python main.py predict --ticker AAPL --confidence
   ```

4. **Run Simulation**: 
   ```bash
   python main.py simulate --strategy ml_prediction --days 30
   ```

5. **Interactive Trading**: 
   ```bash
   python main.py paper-trade --balance 100000
   ```

6. **Automated Trading**: 
   ```bash
   cd ../integrations_&_strategy
   python automated_trader.py --config config.json
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
Create new trading strategies by extending the existing framework in the engine package

### Custom Model Architectures
Modify `backend_&_algorithms/engine/tsr_model/models.py` for neural network architectures

### Additional Data Sources
Extend the data pipelines in `backend_&_algorithms/engine/tsr_model/data_pipelines/`

### Integration Extensions
Add new trading strategies in `integrations_&_strategy/automated_trader.py` or extend the CLI in `backend_&_algorithms/main.py`

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

**Permission Errors:**
```bash
❌ Permission denied when running automated_trader.py
```
- Solution: Ensure scripts are executable: `chmod +x automated_trader.py`

### Getting Help

- **Command Help**: `python main.py --help` or `python main.py <command> --help`
- **Check Model Status**: Use `python main.py models` to see available trained models
- **Check Logs**: Review error messages and stack traces
- **Verify Setup**: Ensure FMP_API_KEY is set and dependencies are installed
- **Model Status**: Use `python main.py models` to see available trained models

### Requirements

- Python 3.8+
- Financial Modeling Prep API key ([Get one here](https://financialmodelingprep.com/developer/docs))
- Dependencies: Install via `pip install -r requirements.txt`

---

**Built with ❤️ for automated algorithmic trading and portfolio management**
