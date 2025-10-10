#!/usr/bin/env python3
"""
Custom Loss Functions for Stock Price Prediction
Addresses the systematic underestimation bias in stock price predictions.
"""

import torch
import torch.nn as nn


class DirectionalLoss(nn.Module):
    """
    Directional Loss Function for Stock Price Prediction

    Combines three components:
    1. MSE Loss - for magnitude accuracy
    2. Directional Loss - heavily penalizes wrong direction predictions
    3. Bias Penalty - reduces systematic over/under-prediction

    This loss function helps prevent the model from consistently predicting
    values that are too low or too high.
    """

    def __init__(self, mse_weight=0.3, direction_weight=0.5, bias_weight=0.2,
                 direction_penalty=2.0):
        """
        Args:
            mse_weight (float): Weight for MSE component (magnitude accuracy)
            direction_weight (float): Weight for directional component
            bias_weight (float): Weight for bias penalty component
            direction_penalty (float): Multiplier for wrong direction predictions
        """
        super(DirectionalLoss, self).__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.bias_weight = bias_weight
        self.direction_penalty = direction_penalty
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, previous_values=None):
        """
        Calculate the directional loss.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Ground truth values (batch_size, 1)
            previous_values: Previous timestep values for direction calculation
                           If None, only MSE is used

        Returns:
            Combined loss value
        """
        # Component 1: Standard MSE for magnitude
        mse_loss = self.mse(predictions, targets)

        # If no previous values, return only MSE
        if previous_values is None:
            return mse_loss

        # Component 2: Directional Loss
        # Calculate actual and predicted directions
        actual_direction = targets - previous_values  # Positive = up, Negative = down
        predicted_direction = predictions - previous_values

        # Check if directions match (same sign)
        direction_match = (actual_direction * predicted_direction) > 0

        # Calculate directional error
        # If directions don't match, apply penalty
        direction_errors = torch.abs(predictions - targets)
        direction_loss = torch.where(
            direction_match,
            direction_errors,  # Normal error if direction is correct
            direction_errors * self.direction_penalty  # Penalized if wrong direction
        ).mean()

        # Component 3: Bias Penalty
        # Penalize systematic over/under-prediction
        prediction_bias = (predictions - targets).mean()
        bias_loss = prediction_bias ** 2

        # Combine all components
        total_loss = (
            self.mse_weight * mse_loss +
            self.direction_weight * direction_loss +
            self.bias_weight * bias_loss
        )

        return total_loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that gives more importance to recent predictions.
    Helps the model adapt to current price trends.
    """

    def __init__(self, temporal_decay=0.95):
        """
        Args:
            temporal_decay (float): Decay factor for older samples (0-1)
                                   1.0 = no decay, lower = more emphasis on recent
        """
        super(WeightedMSELoss, self).__init__()
        self.temporal_decay = temporal_decay

    def forward(self, predictions, targets):
        """
        Calculate weighted MSE with exponential time decay.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Ground truth values (batch_size, 1)

        Returns:
            Weighted MSE loss
        """
        batch_size = predictions.size(0)

        # Create exponentially decaying weights (most recent = highest weight)
        weights = torch.tensor(
            [self.temporal_decay ** (batch_size - i - 1) for i in range(batch_size)],
            device=predictions.device,
            dtype=predictions.dtype
        ).unsqueeze(1)

        # Normalize weights
        weights = weights / weights.sum()

        # Calculate weighted MSE
        squared_errors = (predictions - targets) ** 2
        weighted_loss = (squared_errors * weights).sum()

        return weighted_loss


class HuberDirectionalLoss(nn.Module):
    """
    Combines Huber Loss (robust to outliers) with directional penalty.
    Good for financial data with occasional extreme price movements.
    """

    def __init__(self, delta=1.0, direction_weight=0.3, direction_penalty=2.0):
        """
        Args:
            delta (float): Threshold for Huber loss
            direction_weight (float): Weight for directional component
            direction_penalty (float): Multiplier for wrong direction
        """
        super(HuberDirectionalLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.direction_weight = direction_weight
        self.direction_penalty = direction_penalty

    def forward(self, predictions, targets, previous_values=None):
        """
        Calculate Huber loss with directional penalty.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Ground truth values (batch_size, 1)
            previous_values: Previous timestep values (optional)

        Returns:
            Combined loss value
        """
        # Base Huber loss
        huber_loss = self.huber(predictions, targets)

        if previous_values is None:
            return huber_loss

        # Directional component
        actual_direction = targets - previous_values
        predicted_direction = predictions - previous_values
        direction_match = (actual_direction * predicted_direction) > 0

        direction_errors = torch.abs(predictions - targets)
        direction_loss = torch.where(
            direction_match,
            direction_errors,
            direction_errors * self.direction_penalty
        ).mean()

        # Combine
        total_loss = (1 - self.direction_weight) * huber_loss + self.direction_weight * direction_loss

        return total_loss


def get_loss_function(loss_type='directional', **kwargs):
    """
    Factory function to get loss function by name.

    Args:
        loss_type (str): Type of loss function
            - 'mse': Standard MSE
            - 'directional': DirectionalLoss (recommended for stock prediction)
            - 'weighted_mse': WeightedMSELoss
            - 'huber_directional': HuberDirectionalLoss
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance
    """
    loss_functions = {
        'mse': nn.MSELoss,
        'directional': DirectionalLoss,
        'weighted_mse': WeightedMSELoss,
        'huber_directional': HuberDirectionalLoss
    }

    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_functions.keys())}")

    return loss_functions[loss_type](**kwargs)
