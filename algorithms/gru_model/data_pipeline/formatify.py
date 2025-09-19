import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def format_dataframe_for_gru(df, sequence_length=30, feature_cols=None):
    """
    Takes a pandas DataFrame from YahooFinanceDataPuller, normalizes it, and returns a tensor suitable for GRUStockPredictor.

    Args:
        df (pd.DataFrame): Raw stock data.
        sequence_length (int): Number of time steps per sample.
        feature_cols (list): Columns to use as features (default: OHLCV).

    Returns:
        torch.Tensor: Normalized tensor of shape (num_samples, sequence_length, num_features)
    """

    if feature_cols is None:
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Ensure 'Volume' column exists
    if 'Volume' not in df.columns:
        print("Warning: 'Volume' column missing in input DataFrame. Filling with zeros.")
        df['Volume'] = 0

    # Drop rows with missing values in required columns
    df = df.dropna(subset=feature_cols)
    data = df[feature_cols].values

    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)

    sequences = []
    for i in range(len(data_norm) - sequence_length):
        seq = data_norm[i:i+sequence_length]
        sequences.append(seq)

    train_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    target_tensor = torch.tensor(data_norm[sequence_length:, feature_cols.index('Close')], dtype=torch.float32).unsqueeze(1)
    return train_tensor, target_tensor, scaler
