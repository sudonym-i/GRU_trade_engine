import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class TSRStockPredictor(nn.Module):
    """
    TSR (Technical Stock Return) model for stock prediction using price patterns.
    Uses GRU with optional attention mechanism for sequential price data.
    """
    
    def __init__(self, price_features: int = 4, 
                 hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2,
                 use_attention: bool = True):
        """
        Initialize the TSR model.
        
        Args:
            price_features: Number of price/technical features (Close, SMA, RSI, MACD)
            hidden_dim: Hidden dimension for GRU layers
            num_layers: Number of GRU layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.price_features = price_features
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Price pattern encoder (processes sequential price data)
        self.price_encoder = nn.GRU(
            input_size=price_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for price patterns
        if use_attention:
            self.price_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Feature processing layer
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GRU, nn.LSTM)):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass through the TSR model.
        
        Args:
            x: Input tensor [batch_size, seq_len, price_features]
            
        Returns:
            Predicted stock price [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode price patterns
        price_output, price_hidden = self.price_encoder(x)
        # price_output: [batch, seq, hidden_dim]
        
        # Apply attention to price patterns if enabled
        if self.use_attention:
            price_attended, _ = self.price_attention(
                price_output, price_output, price_output
            )
            # Use last time step of attended output
            price_features = price_attended[:, -1, :]  # [batch, hidden_dim]
        else:
            # Use last time step of GRU output
            price_features = price_output[:, -1, :]  # [batch, hidden_dim]
        
        # Process features
        processed_features = self.feature_processor(price_features)  # [batch, hidden_dim//2]
        
        # Final prediction
        prediction = self.predictor(processed_features)  # [batch, 1]
        
        return prediction
    
    def get_feature_importance(self, x, target_layer='fusion'):
        """
        Get feature importance using gradients (experimental).
        
        Args:
            x: Input tensor
            target_layer: Layer to analyze ('price', 'financial', 'fusion')
            
        Returns:
            Feature importance scores
        """
        x.requires_grad_(True)
        output = self.forward(x)
        
        # Get gradients
        gradients = torch.autograd.grad(
            outputs=output.sum(), 
            inputs=x, 
            create_graph=True
        )[0]
        
        # Calculate importance as gradient magnitude
        importance = gradients.abs().mean(dim=(0, 1))  # Average over batch and sequence
        
        return importance
    
    def predict_with_confidence(self, x, num_samples=10):
        """
        Prediction with uncertainty estimation using Monte Carlo Dropout.
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            Dictionary with prediction, confidence interval, and uncertainty
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        self.eval()  # Back to evaluation mode
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch, 1]
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # 95% confidence interval
        ci_lower = mean_pred - 1.96 * std_pred
        ci_upper = mean_pred + 1.96 * std_pred
        
        return {
            'prediction': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'uncertainty': std_pred / (mean_pred.abs() + 1e-8)  # Relative uncertainty
        }


class AdaptiveTSRPredictor(TSRStockPredictor):
    """
    Enhanced TSR model with adaptive attention weighting.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive attention weighting network
        self.attention_weight_network = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass with adaptive attention weighting."""
        batch_size, seq_len, _ = x.shape
        
        # Encode price patterns
        price_output, _ = self.price_encoder(x)
        
        if self.use_attention:
            price_attended, _ = self.price_attention(
                price_output, price_output, price_output
            )
            price_features = price_attended[:, -1, :]
        else:
            price_features = price_output[:, -1, :]
        
        # Calculate adaptive attention weight
        attention_weight = self.attention_weight_network(price_features)
        
        # Apply adaptive weighting to features
        weighted_features = price_features * attention_weight
        
        # Process features
        processed_features = self.feature_processor(weighted_features)
        
        # Final prediction
        prediction = self.predictor(processed_features)
        
        return prediction

