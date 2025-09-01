
import numpy as np
import pandas as pd


def add_technical_indicators(df):
    """Add SMA, RSI, and MACD to the dataframe as new columns."""
    # Simple Moving Average (SMA)
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df = df.dropna()
    return df


def create_sequences(data: pd.DataFrame, seq_length: int):
    """Convert dataframe to sequences for time series prediction."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)].values
        y = data.iloc[i+seq_length]["Close"]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
