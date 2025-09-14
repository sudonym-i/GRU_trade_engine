# GRU Trade Engine

The end goal is a server-side stock predictor that runs a prediction daily and sends subscribers a daily notification with its prediction and recommended action (buy/hold/sell) based on its time series prediction and sentiment analysis via YouTube transcripts.

## Project Structure

```
TSR_trade_engine/
├── algorithms/
│   ├── gru_model/
│   │   ├── __init__.py
│   │   ├── gru_object.py         # GRU model wrapper class
│   │   ├── gru_architecture.py   # GRU neural network definition
│   │   ├── train_gru.py          # Training script and functions
│   │   └── data_pipeline/
│   │       ├── __init__.py
│   │       ├── yahoo_finance_data.py # Yahoo Finance data collection
│   │       ├── format_for_gru.py     # Data formatting and normalization
│   │       └── formatify.py          # Alternative data formatting
│   └── sentiment_model/             # Sentiment analysis model and scripts
│          └── webscraper/              # Webscraper for collecting sentiment data
│               ├── main.cpp
│               ├── scraper.cpp
│               ├── scraper.h          # C++ code for web scraping
│               ├── CMakeLists.txt
│               └── build/
│                     └── webscrape.exe   # The actual executable
├── data/                     # Data storage directory (CSV files)
├── integrations/             # External integration scripts
├── requirements.txt          # Python dependencies
├── test.py                   # Main test script
├── interact.sh               # Interactive shell script
└── README.md                 # Project documentation
```

---

## How the GRU Model Works

### `gru_architecture.py`

This file defines the GRU-based neural network for time series regression. The model (`GRUPredictor` class) uses PyTorch's GRU layers to process sequences of historical stock data (such as OHLCV features) and predicts the next closing price.

- **Input:** A sequence of past stock prices and features (e.g., Open, High, Low, Close, Volume).
- **Architecture:** 
  - Multiple GRU layers extract temporal patterns.
  - Fully connected layers process the GRU output.
  - The final output is a single predicted value (the next price).
- **Output:** The predicted next price for each input sequence.

---

## Using the `gru_object.py` Wrapper

The `GRUModel` class in `gru_object.py` provides a high-level interface for working with the GRU predictor. It handles data loading, formatting, training, prediction, and model persistence.

### Typical Workflow

1. **Initialize the Model**
    ```python
    from algorithms.gru_model.gru_object import GRUModel
    gru_model = GRUModel(input_size, hidden_size, output_size)
    ```

2. **Pull and Format Data**
    ```python
    gru_model.data_dir = "./data"
    gru_model.pull_data(symbol="NVDA", period="max")
    gru_model.format_data()
    ```

3. **Train the Model**
    ```python
    gru_model.train(epochs=30, lr=0.001)
    ```

4. **Make Predictions**
    ```python
    gru_model.pull_data(symbol="NVDA", period="1y")
    )gru_model.format_data()
    gru_model.predict()
    price_prediction = gru_model.un_normalize()
    print(price_prediction)
    ```

5. **Save/Load Model**
    ```python
    gru_model.save_model("my_gru_model.pth")
    gru_model.load_model("my_gru_model.pth")
    ```

### Key Features

- **Data Handling:** Automatically fetches and formats stock data for training and prediction.
- **Normalization:** Uses MinMaxScaler for feature normalization and provides utilities for un-normalizing predictions.
- **Training:** Trains the GRU model on historical data.
- **Prediction:** Predicts the next price (tomorrow's closing price) for each input sequence.
- **Persistence:** Easily save and load trained models.

---

## Sentiment Analysis

The project also includes a sentiment analysis pipeline (see `sentiment_model/webscraper/`) for collecting and analyzing YouTube transcripts to inform trading decisions.


---

## Example

See `main.py` for a complete example of pulling data, training, predicting, and un-normalizing the output.

---

## Contact

For questions or contributions, please open an issue or pull request on GitHub.