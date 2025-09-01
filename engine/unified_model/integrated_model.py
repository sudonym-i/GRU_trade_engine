import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class UnifiedStockPredictor(nn.Module):
    """
    Unified model combining technical price patterns with financial fundamentals.
    Uses a hybrid architecture with separate processing paths that merge for final prediction.
    """
    
    def __init__(self, price_features: int = 4, financial_features: int = 13, 
                 hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2,
                 use_attention: bool = True):
        """
        Initialize the unified model.
        
        Args:
            price_features: Number of price/technical features (Close, SMA, RSI, MACD)
            financial_features: Number of financial fundamental features
            hidden_dim: Hidden dimension for LSTM/GRU layers
            num_layers: Number of LSTM/GRU layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.price_features = price_features
        self.financial_features = financial_features
        self.total_features = price_features + financial_features
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
        
        # Financial fundamentals encoder (processes sequential financial data)
        self.financial_encoder = nn.GRU(
            input_size=financial_features,
            hidden_size=hidden_dim // 2,  # Smaller since fundamentals change less frequently
            num_layers=max(1, num_layers - 1),
            batch_first=True,
            dropout=dropout if num_layers > 2 else 0
        )
        
        # Attention mechanism for price patterns
        if use_attention:
            self.price_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Combined feature dimension
        combined_dim = hidden_dim + (hidden_dim // 2)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        Forward pass through the unified model.
        
        Args:
            x: Input tensor [batch_size, seq_len, total_features]
            
        Returns:
            Predicted stock price [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Split input into price and financial features
        price_data = x[:, :, :self.price_features]  # [batch, seq, price_features]
        financial_data = x[:, :, self.price_features:]  # [batch, seq, financial_features]
        
        # Encode price patterns
        price_output, price_hidden = self.price_encoder(price_data)
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
        
        # Encode financial fundamentals
        financial_output, _ = self.financial_encoder(financial_data)
        financial_features = financial_output[:, -1, :]  # [batch, hidden_dim//2]
        
        # Combine features
        combined_features = torch.cat([price_features, financial_features], dim=1)
        # combined_features: [batch, hidden_dim + hidden_dim//2]
        
        # Fuse features
        fused_features = self.fusion(combined_features)  # [batch, hidden_dim//2]
        
        # Final prediction
        prediction = self.predictor(fused_features)  # [batch, 1]
        
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


class AdaptiveUnifiedPredictor(UnifiedStockPredictor):
    """
    Enhanced version with adaptive weighting of price vs fundamental features.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive weighting network
        self.weight_network = nn.Sequential(
            nn.Linear(self.hidden_dim + (self.hidden_dim // 2), 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Weights for [price_features, financial_features]
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """Forward pass with adaptive feature weighting."""
        batch_size, seq_len, _ = x.shape
        
        # Split input into price and financial features
        price_data = x[:, :, :self.price_features]
        financial_data = x[:, :, self.price_features:]
        
        # Encode features
        price_output, _ = self.price_encoder(price_data)
        financial_output, _ = self.financial_encoder(financial_data)
        
        if self.use_attention:
            price_attended, _ = self.price_attention(
                price_output, price_output, price_output
            )
            price_features = price_attended[:, -1, :]
        else:
            price_features = price_output[:, -1, :]
        
        financial_features = financial_output[:, -1, :]
        
        # Combine for weight calculation
        combined_features = torch.cat([price_features, financial_features], dim=1)
        
        # Calculate adaptive weights
        feature_weights = self.weight_network(combined_features)  # [batch, 2]
        
        # Apply adaptive weighting
        weighted_price = price_features * feature_weights[:, 0:1]
        weighted_financial = financial_features * feature_weights[:, 1:2]
        
        # Combine weighted features
        adaptive_features = torch.cat([weighted_price, weighted_financial], dim=1)
        
        # Fuse and predict
        fused_features = self.fusion(adaptive_features)
        prediction = self.predictor(fused_features)
        
        return prediction


if __name__ == "__main__":
    # Test the unified model
    print("Testing Unified Stock Predictor...")
    
    # Model parameters
    price_features = 4  # Close, SMA_14, RSI_14, MACD
    financial_features = 13  # From financial model
    seq_length = 60
    batch_size = 32
    
    # Create model
    model = UnifiedStockPredictor(
        price_features=price_features,
        financial_features=financial_features,
        hidden_dim=128,
        num_layers=3,
        use_attention=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_length, price_features + financial_features)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    # Test confidence prediction
    confidence_result = model.predict_with_confidence(x[:1], num_samples=5)
    print(f"Prediction with confidence:")
    for key, value in confidence_result.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test adaptive model
    print("\nTesting Adaptive Unified Predictor...")
    adaptive_model = AdaptiveUnifiedPredictor(
        price_features=price_features,
        financial_features=financial_features,
        hidden_dim=128
    )
    
    with torch.no_grad():
        adaptive_output = adaptive_model(x)
        print(f"Adaptive output shape: {adaptive_output.shape}")
    
    print("Model tests completed successfully!")