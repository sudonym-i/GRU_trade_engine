# LogisticSentimentModel Usage
# ----------------------------
# This module defines a simple PyTorch logistic regression model for sentiment analysis.
#
# Usage Example:
#   from logistic_model import LogisticSentimentModel
#   import torch
#   model = LogisticSentimentModel(input_dim=YOUR_FEATURE_DIM)
#   input_tensor = torch.tensor([[...]], dtype=torch.float32)  # shape: (batch_size, input_dim)
#   output = model(input_tensor)  # output: probabilities (batch_size, 1)
#
# Arguments:
#   input_dim (int): Number of input features.
#
# Returns:
#   torch.Tensor: Probabilities for each input sample (after sigmoid activation).

# Simple PyTorch Logistic Regression model for sentiment analysis
import torch
import torch.nn as nn

class LogisticSentimentModel(nn.Module):
	def __init__(self, input_dim):
		super(LogisticSentimentModel, self).__init__()
		self.linear = nn.Linear(input_dim, 1)

	def forward(self, x):
		logits = self.linear(x)
		probs = torch.sigmoid(logits)
		return probs

# Example usage:
# model = LogisticSentimentModel(input_dim=YOUR_FEATURE_DIM)
# output = model(torch.tensor([[...]], dtype=torch.float32))
