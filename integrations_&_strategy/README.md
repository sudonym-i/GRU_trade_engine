# Neural Trade Engine - Automated Trading Integration

This directory contains the automated trading system that integrates with your neural trade engine backend.

## Overview

The automated trading system focuses on **a single stock at a time** and performs daily:
1. **Webscraping** for sentiment analysis of the target stock
2. **Price predictions** using a dedicated TSR model 
3. **Trading decisions** based on prediction confidence
4. **Single-stock portfolio management** with risk controls

## Files

- `automated_trader.py` - Main automated trading script
- `schedule_trader.py` - Scheduling system for daily execution
- `config.json` - Trading configuration and parameters
- `README.md` - This documentation

## Quick Start

### 1. Install Dependencies
```bash
pip install apscheduler
```

### 2. Test Single Run
```bash
cd integrations_&_strategy
python automated_trader.py --stock NVDA --dry-run
```

### 3. Start Daily Automation
```bash
python schedule_trader.py --start
```

## Configuration

Edit `config.json` to customize:

```json
{
  "target_stock": "NVDA",
  "confidence_threshold": 0.65,
  "price_change_threshold": 0.02,
  "position_size": 1.0,
  "stop_loss": -0.05,
  "take_profit": 0.10
}
```

### Key Parameters:
- **target_stock**: Single stock to focus on (e.g., "NVDA")
- **confidence_threshold**: Minimum prediction confidence (0.65 = 65%)
- **price_change_threshold**: Minimum price change to trade (0.02 = 2%)
- **position_size**: Percentage of portfolio to invest (1.0 = 95%, keeping 5% cash)
- **stop_loss**: Stop loss percentage (-0.05 = -5%)
- **take_profit**: Take profit percentage (0.10 = +10%)

## Daily Workflow

### 5:00 PM (Market Close)
1. Run webscraping for target stock: `python3 main.py webscrape --ticker NVDA`
2. Generate prediction: `python3 main.py predict --ticker NVDA`
3. Analyze prediction confidence and price change
4. Generate BUY/SELL/HOLD signal
5. Log decision to `trading_decisions.json`

### 9:30 AM (Market Open)
Execute trades based on previous day's signals (manual for now - can be automated with broker API)

## Usage Examples

### Run with Different Stock
```bash
python automated_trader.py --stock AAPL
```

### Custom Configuration
```bash
python automated_trader.py --config my_config.json
```

### Test Mode (No Portfolio Changes)
```bash
python automated_trader.py --dry-run
```

### View Scheduled Jobs
```bash
python schedule_trader.py --status
```

### Test One Trading Cycle
```bash
python schedule_trader.py --test
python schedule_trader.py --test --stock AAPL  # Test with different stock
```

## Trading Strategy

### Signal Generation Logic:
```python
if confidence > 0.65:  # High confidence predictions only
    if predicted_price > current_price * 1.02:  # >2% upside
        signal = "BUY"
    elif predicted_price < current_price * 0.98:  # >2% downside
        signal = "SELL" 
    else:
        signal = "HOLD"  # Prediction too close
else:
    signal = "HOLD"  # Low confidence
```

### Risk Management:
- **Single stock focus** (optimal for TSR model)
- **95% position sizing** (5% cash reserve)
- **5% stop loss** on position
- **10% take profit** target
- **65% confidence threshold** for trading

## Portfolio Tracking

The system maintains:
- `portfolio_state.json` - Current cash and positions
- `trading_decisions.json` - Historical decisions and reasoning
- `automated_trader.log` - Detailed execution logs

### Portfolio State Example:
```json
{
  "cash": 5000.0,
  "positions": {
    "NVDA": {
      "shares": 210,
      "avg_price": 452.38,
      "purchase_date": "2025-09-03T17:30:00"
    }
  },
  "total_value": 100000.0
}
```

## Scheduling

### Daily Schedule (Monday-Friday):
- **5:00 PM**: Run prediction and generate signal for target stock
- **8:00 PM**: Weekly model retraining for target stock (Sundays only)

### Manual Override:
Stop scheduler: `Ctrl+C` in running terminal

## Integration with Backend

The system calls your existing backend functions:
```bash
# Webscraping
python3 ../backend_&_algorithms/main.py webscrape --ticker NVDA

# Predictions  
python3 ../backend_&_algorithms/main.py predict --ticker NVDA
```

## Logs and Monitoring

### Log Files:
- `automated_trader.log` - Trading execution logs
- `scheduler.log` - Scheduling system logs
- `trading_decisions.json` - Decision history

### Monitoring Commands:
```bash
# View recent trading decisions
tail -f trading_decisions.json

# Monitor live execution
tail -f automated_trader.log

# Check scheduler status
python schedule_trader.py --status
```

## Security Notes

- **Paper Trading**: This system tracks positions but doesn't execute real trades
- **No API Keys**: No broker integration yet - purely analytical
- **Local Files**: All data stored locally for security

## Next Steps

1. **Broker Integration**: Add Interactive Brokers API for live trading
2. **SMS/Email Alerts**: Notifications for trading signals
3. **Performance Analytics**: Backtesting and performance tracking
4. **Web Dashboard**: Real-time portfolio monitoring

## Troubleshooting

### Common Issues:

**"No trained models found"**
```bash
cd ../backend_&_algorithms
python main.py train --ticker NVDA --days 730
```

**"Prediction failed"**
- Ensure Interactive Brokers Gateway is running
- Check if ticker symbols are valid
- Verify model files exist

**"Scheduling not working"**
```bash
pip install apscheduler
python schedule_trader.py --test
```

For support, check the logs and ensure all backend dependencies are installed.