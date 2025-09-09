# TSR Trade Engine

A comprehensive stock prediction and analysis system using GRU neural networks and Yahoo Finance data.

## Project Structure

```
TSR_trade_engine/
├── algorithms/
│   ├── __init__.py
│   ├── gru_predictor.py      # GRU neural network model
│   └── train_gru.py          # Training script with full pipeline
├── data_pipelines/
│   ├── __init__.py
│   ├── yahoo_finance_data.py # Yahoo Finance data collection
│   └── format_for_gru.py     # Data formatting and normalization
├── data/                     # Data storage directory (CSV files)
├── integrations/             # External integration scripts
├── models/                   # Trained model storage (created during training)
├── requirements.txt          # Python dependencies
├── test.py                   # Testing utilities
└── interact.sh               # Interactive shell script
```

## Features

- **Data Collection**: Automated Yahoo Finance data pulling with 3-year historical data
- **Data Processing**: Comprehensive data formatting and normalization for neural networks
- **GRU Model**: PyTorch implementation of GRU for time series stock prediction
- **Training Pipeline**: Complete training workflow with validation and early stopping
- **Visualization**: Training curves and performance monitoring

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Train a model**:
```bash
cd algorithms
python train_gru.py
```

2. **Format data manually**:
```python
from data_pipelines.format_for_gru import format_stock_data
input_tensor, target_tensor, scaler = format_stock_data("AAPL")
```

3. **Collect data only**:
```python
from data_pipelines.yahoo_finance_data import YahooFinanceDataPuller
puller = YahooFinanceDataPuller()
data = puller.get_stock_data("AAPL")
puller.save_to_csv(data, "AAPL")
```

## Data Pipeline

### 1. Data Collection (`yahoo_finance_data.py`)
- Fetches OHLCV data from Yahoo Finance
- Supports multiple symbols and time periods
- Automatic CSV storage with date stamps
- Error handling and retry logic

### 2. Data Formatting (`format_for_gru.py`)
- Converts CSV data to PyTorch tensors
- MinMaxScaler normalization
- Technical indicator calculation (SMA, volatility, price changes)
- Sequence creation for time series modeling
- Handles 7 features: Open, High, Low, Close, Volume, Price_Change_Pct, Volatility

### 3. Model Training (`train_gru.py`)
- End-to-end training pipeline
- Multi-stock training support
- Train/validation/test splits
- Early stopping and learning rate scheduling
- Model checkpointing and performance tracking
- Automated visualization and logging

## Model Architecture

The GRU predictor uses:
- **Input**: 30-day sequences of 7 features
- **Architecture**: 3-layer GRU with 128 hidden units
- **Output**: Next-day closing price prediction
- **Training**: Adam optimizer with MSE loss
- **Regularization**: Dropout and early stopping

## Configuration

Default training parameters:
- Sequence length: 30 days
- Hidden size: 128
- Layers: 3
- Learning rate: 0.001
- Batch size: 64
- Training symbols: AAPL, GOOGL, MSFT, TSLA, AMZN

Modify these in `train_gru.py` main function.

## Output Files

Training produces:
- `models/best_gru_model.pt` - Best model checkpoint
- `training_curves.png` - Loss visualization
- `training_history.json` - Training metadata
- `data/tensors/` - Processed tensor files

## Requirements

- Python 3.8+
- PyTorch 2.0+
- yfinance 0.2.18+
- scikit-learn 1.3+
- pandas 1.5+
- matplotlib 3.7+

See `requirements.txt` for complete list.

## License

This project is for educational and research purposes.