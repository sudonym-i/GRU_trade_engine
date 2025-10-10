import pandas as pd

# --- Technical Indicator Calculations ---
def calculate_ema(series, span=14):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band

def calculate_volatility(series, window=20):
    return series.rolling(window=window).std()
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def format_dataframe_for_gru(df, sequence_length=60, feature_cols=None, scaler=None):
    """
    Takes a pandas DataFrame from YahooFinanceDataPuller, normalizes it, and returns a tensor suitable for GRUStockPredictor.

    Args:
        df (pd.DataFrame): Raw stock data.
        sequence_length (int): Number of time steps per sample (default: 60).
        feature_cols (list): Columns to use as features (default: OHLCV).
        scaler: Pre-fitted MinMaxScaler (optional). If provided, use this instead of fitting a new one.

    Returns:
        torch.Tensor: Normalized tensor of shape (num_samples, sequence_length, num_features)
        torch.Tensor: Target tensor (or None if using existing scaler for prediction)
        MinMaxScaler: The scaler used (either the one provided or newly fitted)
        list: Feature column names
    """

    if feature_cols is None:
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Ensure 'Volume' column exists
    if 'Volume' not in df.columns:
        print("Warning: 'Volume' column missing in input DataFrame. Filling with zeros.")
        df['Volume'] = 0

    # --- Calculate technical indicators ---
    df['EMA_14'] = calculate_ema(df['Close'], span=14)
    df['RSI_14'] = calculate_rsi(df['Close'], period=14)
    macd, macd_signal = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    upper_band, lower_band = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = upper_band
    df['BB_Lower'] = lower_band
    df['Volatility_20'] = calculate_volatility(df['Close'], window=20)

    # New feature columns
    indicator_cols = ['EMA_14', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volatility_20']
    all_feature_cols = feature_cols + indicator_cols

    # Drop rows with missing values in required columns
    df = df.dropna(subset=all_feature_cols)
    data = df[all_feature_cols].values

    # Validate we have enough data for the sequence length
    if len(data) < sequence_length + 1:
        raise ValueError(
            f"Insufficient data for sequence creation. "
            f"Need at least {sequence_length + 1} rows after technical indicators, "
            f"but only have {len(data)} rows. "
            f"Try increasing the data period or reducing sequence_length."
        )

    # Use provided scaler or create new one
    if scaler is None:
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(data)
    else:
        # Use existing scaler (for prediction)
        data_norm = scaler.transform(data)

    sequences = []
    for i in range(len(data_norm) - sequence_length):
        seq = data_norm[i:i+sequence_length]
        sequences.append(seq)

    if len(sequences) == 0:
        raise ValueError(
            f"No sequences created. Data has {len(data_norm)} rows, "
            f"sequence_length is {sequence_length}. Need more data."
        )

    train_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    target_tensor = torch.tensor(data_norm[sequence_length:, all_feature_cols.index('Close')], dtype=torch.float32).unsqueeze(1)
    return train_tensor, target_tensor, scaler, all_feature_cols
