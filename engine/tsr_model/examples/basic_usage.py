#!/usr/bin/env python3
"""
Basic usage example for TSR Model with Financial Modeling Prep API
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import DataLoader, make_dataset
from model import GRUPredictor
from train import train_gru_predictor
from trade import simulate_trading

def main():
    """Example of basic TSR model usage"""
    print("TSR Model Basic Usage Example")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        print("Please set FMP_API_KEY environment variable")
        print("See docs/FMP_API_SETUP.md for instructions")
        return
    
    # Parameters
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-06-01"
    seq_length = 14
    
    print(f"Using ticker: {ticker}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Sequence length: {seq_length}")
    
    try:
        # 1. Load and prepare data
        print("\n1. Loading data...")
        dataset, input_dim = make_dataset(
            ticker=ticker,
            start=start_date, 
            end=end_date,
            seq_length=seq_length,
            normalize=True,
            plot_indicators=False
        )
        print(f"Dataset created with {len(dataset)} samples, input_dim: {input_dim}")
        
        # 2. Create and train model
        print("\n2. Training model...")
        model = GRUPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)
        losses = train_gru_predictor(
            model=model,
            dataset=dataset,
            epochs=5,  # Small number for example
            batch_size=16,
            lr=0.001,
            plot_loss=False
        )
        print(f"Training completed. Final loss: {losses[-1]:.4f}")
        
        # 3. Test with trading simulation (using same data for simplicity)
        print("\n3. Running trading simulation...")
        test_dataset, _ = make_dataset(
            ticker=ticker,
            start=start_date,
            end=end_date,
            seq_length=seq_length,
            normalize=False,  # Use unnormalized data for testing
            plot_indicators=False
        )
        
        # Get actual prices for simulation
        prices = test_dataset.tensors[1].numpy()
        
        portfolio_values, trades = simulate_trading(
            model=model,
            dataset=test_dataset,
            prices=prices,
            plot_results=False,  # Set to True to see plots
            initial_balance=10000
        )
        
        # 4. Print results
        print(f"\n4. Results:")
        print(f"Initial balance: $10,000")
        print(f"Final portfolio value: ${portfolio_values[-1]:.2f}")
        profit = portfolio_values[-1] - 10000
        print(f"Total profit/loss: ${profit:.2f} ({profit/10000*100:.2f}%)")
        
        num_trades = len([t for t in trades if t['action'] in ['buy', 'sell']])
        print(f"Number of trades executed: {num_trades}")
        
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("\n✓ Example completed successfully!")
    print("\nNext steps:")
    print("- Try with different tickers and date ranges")
    print("- Adjust model parameters (hidden_dim, num_layers)")
    print("- Experiment with different sequence lengths")
    print("- Enable plotting to see visualizations")

if __name__ == "__main__":
    main()