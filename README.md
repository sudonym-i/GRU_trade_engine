
# ML Trading Bot

This project is an automated trading bot that uses deep learning to predict future stock prices and make trading decisions.

## High Level Project Structure
<img width="1150" height="781" alt="Screenshot from 2025-08-30 23-15-35" src="https://github.com/user-attachments/assets/60bc1a9d-1084-4771-9aa1-756a8d104cae" />

## How it Works

- **Data Loading:**
    - Downloads historical price data for tickers specified in `config.json` using Yahoo Finance (`yfinance`).
    - Adds technical indicators (SMA, RSI, MACD) to the data for richer feature representation.
    - Data is split into sequences of configurable length for time series prediction.

- **Machine Learning Model:**
    - The core model is a multi-layer Gated Recurrent Unit (GRU) neural network implemented in PyTorch (`nn/model.py`).
        - 3 GRU layers, 128 hidden units each
        - Two fully connected layers (64 units, then 1 output)
        - SiLU (Swish) activation function
        - Dropout (0.2) after the first fully connected layer for regularization
        - Trained to predict the next closing price given a sequence of past prices and indicators

- **Training:**
    - Model is trained using Mean Squared Error (MSE) loss and Adam optimizer.
    - Training parameters (epochs, batch size, learning rate, etc.) are set in `train_nn.json`.
      
<img width="1568" height="957" alt="Screenshot from 2025-08-24 11-21-57" src="https://github.com/user-attachments/assets/3666287c-01ed-4a8e-8cff-57b91a18edaf" />

- **Trading Logic:**
    - For each time step, the model predicts the next price.
    - **Buy** if prediction > current price (or hold if already holding)
    - **Sell** if prediction < current price
    - The bot currently trades the entire portfolio (all-in/all-out, no position sizing)
 
      
<img width="1568" height="957" alt="Screenshot from 2025-08-24 11-22-07" src="https://github.com/user-attachments/assets/2262b8d4-06da-4af4-908a-e3e02c3cacbc" />

## Configuration
Edit `src/config.json` to set:

- `tickers`: List of stock symbols
- `start`, `end`: Date range for historical data
- `seq_length`: Number of time steps in each input sequence
- `interval`: Data frequency (e.g., '1d')
- `epochs`, `batch_size`, `lr`: Training hyperparameters

  <img width="1570" height="912" alt="Screenshot from 2025-08-24 11-22-25" src="https://github.com/user-attachments/assets/b2c8bfa6-9825-4883-b035-0d69cabedc6f" />


## Limitations

- The trading strategy is intentionally simple and does not account for transaction costs, slippage, or risk management.
- The model does not use advanced portfolio optimization or reinforcement learning (yet).

---
For more details, see the code in `src/`.
