#!/usr/bin/env python3
"""
GRU-based Stock Price Predictor
PyTorch implementation of a GRU model for time series regression to predict next-day closing prices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class GRUPredictor(nn.Module):
    """GRU-based neural network for stock price prediction"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_size (int): Number of input features (default 5 for OHLCV: Open, High, Low, Close, Volume)
            hidden_size (int): Number of hidden units in GRU.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout rate.
        """
        super(GRUPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, past_series: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(past_series)
        last_output = gru_out[:, -1, :]
        output = self.relu(self.fc1(last_output))
        output = self.dropout(output)
        output = self.fc2(output)
        return output  # Return all predictions in the batch
