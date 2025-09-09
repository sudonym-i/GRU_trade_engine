# GRU Trade Engine

The end goal is a server-side stock predictor that runs a prediction daily and sends subscribers a daily notification with its prediction and recommended action (buy/hold/sell) based on its time series prediction and sentiment analysis via youtube transcripts

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
