# TSR Trade Engine


## Project Structure

```
TSR_trade_engine/
├── algorithms/
│   ├── __init__.py
│   ├── gru_predictor.py      # GRU neural network model
│   └── train_gru.py          # Training script with full pipeline
├── data_pipelines/
│   ├── __init__.py
│   ├── yahoo_finance_data.py # Yahoo Finance data collection
│   └── format_for_gru.py     # Data formatting and normalization
├── data/                     # Data storage directory (CSV files)
├── integrations/             # External integration scripts
├── models/                   # Trained model storage (created during training)
├── requirements.txt          # Python dependencies
├── test.py                   # Testing utilities
└── interact.sh               # Interactive shell script
```
