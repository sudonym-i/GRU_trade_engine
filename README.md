
# GRU Trade Engine

**Deployed at https://discord.gg/zfavbh9pF**

GRU Trade Engine is an automated stock prediction and sentiment analysis system. It combines deep learning (GRU neural networks) for time series price prediction with sentiment analysis from YouTube transcripts, and can send results to Discord. The workflow is orchestrated via shell scripts for training, prediction, and notification.


## Project Structure

```
GRU_trade_engine/
├── main.py                        # Main entry point: train, predict, sentiment, Discord notification
├── run.sh                         # End-to-end workflow: prediction, scraping, sentiment, Discord
├── setup.sh                       # Interactive training/setup script
├── data/                          # Data storage (CSV, raw transcripts, outputs)
│   ├── *.csv                      # Stock price data (e.g., AAPL.csv, MSFT.csv, ORCL.csv)
│   ├── youtube_data.raw           # Scraped YouTube transcript data
│   ├── params.txt                 # Stores last-used ticker/company
│   └── output/                    # Prediction and analysis results
│       ├── gru_prediction.out     # Output from GRU prediction
│       └── sentiment_analysis.out # Output from sentiment analysis
├── algorithms/
│   ├── requirements.txt           # Python dependencies
│   ├── gru_model/
│   │   ├── __init__.py            # Package init
│   │   ├── gru_object.py          # GRU model wrapper
│   │   ├── gru_architecture.py    # GRU neural network (PyTorch)
│   │   ├── train_gru.py           # Training loop
│   │   ├── models/                # Trained model storage
│   │   └── data_pipeline/
│   │       ├── yahoo_finance_data.py # Yahoo Finance data puller
│   │       └── formatify.py           # Feature engineering, indicators
│   └── sentiment_model/
│       ├── youtube_sentiment.py   # Sentiment analysis (transformers)
│       └── web_scraper/
│           ├── main.cpp           # C++ web scraper for YouTube transcripts
│           ├── scraper.cpp        # Scraper logic
│           ├── scraper.h          # Scraper header
│           ├── CMakeLists.txt     # Build config
│           └── build/             # Built executable (webscrape)
└── README.md
```

---


## Functionality Overview

### GRU Price Prediction
- Uses a PyTorch GRU model (`GRUPredictor`) to predict future closing prices based on historical OHLCV and technical indicators (EMA, RSI, MACD, Bollinger Bands, etc.).
- Data is pulled from Yahoo Finance and processed via `formatify.py`.
- Training and prediction are managed by `main.py` and `gru_object.py`.

---


### Sentiment Analysis
- Scrapes YouTube transcripts for a given company using a custom C++ web scraper.
- Analyzes sentiment using HuggingFace transformers (`distilbert-base-uncased-finetuned-sst-2-english`).
- Outputs average sentiment score and entry count.

---


### Discord Notification
- Results (price prediction and sentiment score) can be sent to a Discord channel via webhook (configured in `.discord_webhook`).


---

## Usage

### Quickstart
1. **Setup/Install:**
    - Run `setup.sh` for interactive setup, dependency installation, and initial training.
2. **Prediction Workflow:**
    - Run `run.sh` to execute the full pipeline:
      - Predict price (`main.py --mode p`)
      - Scrape YouTube transcripts (C++ scraper)
      - Sentiment analysis (`main.py --mode s`)
      - Send results to Discord (`main.py --mode discord`)
3. **Manual Control:**
    - Use `main.py` directly for training (`--mode t`), prediction (`--mode p`), sentiment (`--mode s`), or Discord notification (`--mode discord`).

### Example CLI Usage
```bash
# Train GRU model
python3 main.py --mode t --symbol AAPL --epochs 30 --lr 0.001 --batch_size 3

# Predict
python3 main.py --mode p --symbol AAPL

# Run sentiment analysis
python3 main.py --mode s --symbol AAPL

# Send results to Discord
python3 main.py --mode discord --symbol AAPL --prediction_file data/output/gru_prediction.out --sentiment_file data/output/sentiment_analysis.out
```

---


## Notes
- All Python dependencies are listed in `algorithms/requirements.txt`.
- C++ scraper requires `libcurl` and `fmt` libraries.
- Stock data CSV files are stored in `data/`.
- Trained model files are stored in `algorithms/gru_model/models/`.
- Output files (predictions and sentiment) are stored in `data/output/`.
- Parameters for last run are stored in `data/params.txt`.
- **These dependencies can be automatically installed in the walkthrough when you run setup.sh**

## Contact
For questions or contributions, please open an issue or pull request on GitHub.
