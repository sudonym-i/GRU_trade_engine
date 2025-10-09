"""
Multi-stock data loader for pre-training GRU models.
Aggregates data from multiple stocks for transfer learning.
"""

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from algorithms.gru_model.data_pipeline.yahoo_finance_data import YahooFinanceDataPuller
from algorithms.gru_model.data_pipeline.formatify import (
    calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_volatility
)


def add_technical_indicators(df):
    """
    Add technical indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with technical indicators added
    """
    df = df.copy()

    # Calculate technical indicators
    df['EMA_14'] = calculate_ema(df['Close'], span=14)
    df['RSI_14'] = calculate_rsi(df['Close'], period=14)
    macd, macd_signal = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    upper_band, lower_band = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = upper_band
    df['BB_Lower'] = lower_band
    df['Volatility_20'] = calculate_volatility(df['Close'], window=20)

    return df


def pull_multiple_stocks(stock_symbols, period="3y", interval="1d", data_dir="./data"):
    """
    Pull data for multiple stock symbols.

    Args:
        stock_symbols: List of stock ticker symbols
        period: Data period (default: '3y')
        interval: Data interval (default: '1d')
        data_dir: Directory to save data

    Returns:
        Dictionary mapping stock symbols to DataFrames
    """
    puller = YahooFinanceDataPuller(data_dir=data_dir)
    stock_data = {}

    print(f"\n{'='*60}")
    print(f"Pulling data for {len(stock_symbols)} stocks: {', '.join(stock_symbols)}")
    print(f"{'='*60}\n")

    for symbol in stock_symbols:
        try:
            data = puller.get_stock_data(symbol, period=period, interval=interval)
            if not data.empty and 'error' not in data.columns:
                # Add technical indicators
                data = add_technical_indicators(data)
                stock_data[symbol] = data
                print(f"✓ Successfully loaded {symbol}: {len(data)} records")
            else:
                print(f"✗ Failed to load {symbol}")
        except Exception as e:
            print(f"✗ Error loading {symbol}: {e}")

    print(f"\n{'='*60}")
    print(f"Successfully loaded {len(stock_data)}/{len(stock_symbols)} stocks")
    print(f"{'='*60}\n")

    return stock_data


def format_multi_stock_data(stock_data_dict, sequence_length=60, validation_split=0.2):
    """
    Format multiple stock DataFrames into combined training tensors.
    Each stock is normalized independently before combining.

    Args:
        stock_data_dict: Dictionary mapping symbols to DataFrames
        sequence_length: Number of time steps per sample
        validation_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_tensor, train_target, val_tensor, val_target, scaler_dict, feature_cols)
    """
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'EMA_14', 'RSI_14', 'MACD', 'MACD_Signal',
                    'BB_Upper', 'BB_Lower', 'Volatility_20']

    all_train_sequences = []
    all_train_targets = []
    all_val_sequences = []
    all_val_targets = []
    scaler_dict = {}

    print(f"\n{'='*60}")
    print(f"Formatting data for {len(stock_data_dict)} stocks")
    print(f"Sequence length: {sequence_length}, Validation split: {validation_split}")
    print(f"{'='*60}\n")

    for symbol, df in stock_data_dict.items():
        # Drop rows with missing values
        df = df.dropna(subset=feature_cols)

        if len(df) < sequence_length + 1:
            print(f"⚠ Skipping {symbol}: insufficient data ({len(df)} rows)")
            continue

        # Extract feature data
        data = df[feature_cols].values

        # Normalize per stock (important!)
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(data)
        scaler_dict[symbol] = scaler

        # Create sequences
        sequences = []
        targets = []
        close_idx = feature_cols.index('Close')

        for i in range(len(data_norm) - sequence_length):
            seq = data_norm[i:i+sequence_length]
            target = data_norm[i+sequence_length, close_idx]
            sequences.append(seq)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Split into train and validation
        split_idx = int(len(sequences) * (1 - validation_split))

        train_seq = sequences[:split_idx]
        train_tgt = targets[:split_idx]
        val_seq = sequences[split_idx:]
        val_tgt = targets[split_idx:]

        all_train_sequences.append(train_seq)
        all_train_targets.append(train_tgt)
        all_val_sequences.append(val_seq)
        all_val_targets.append(val_tgt)

        print(f"✓ {symbol}: {len(train_seq)} train samples, {len(val_seq)} val samples")

    # Combine all stocks
    train_sequences = np.concatenate(all_train_sequences, axis=0)
    train_targets = np.concatenate(all_train_targets, axis=0)
    val_sequences = np.concatenate(all_val_sequences, axis=0)
    val_targets = np.concatenate(all_val_targets, axis=0)

    # Convert to tensors
    train_tensor = torch.tensor(train_sequences, dtype=torch.float32)
    train_target_tensor = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1)
    val_tensor = torch.tensor(val_sequences, dtype=torch.float32)
    val_target_tensor = torch.tensor(val_targets, dtype=torch.float32).unsqueeze(1)

    print(f"\n{'='*60}")
    print(f"Total training samples: {len(train_tensor)}")
    print(f"Total validation samples: {len(val_tensor)}")
    print(f"Input shape: {train_tensor.shape}")
    print(f"{'='*60}\n")

    return (train_tensor, train_target_tensor,
            val_tensor, val_target_tensor,
            scaler_dict, feature_cols)
