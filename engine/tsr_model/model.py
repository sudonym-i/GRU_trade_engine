import torch
import torch.nn as nn

class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)
