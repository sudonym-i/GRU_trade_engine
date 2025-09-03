# Neural Trade Engine - Automated Trading Integration

This directory contains the automated trading system that integrates with your neural trade engine backend.

## Overview

The automated trading system focuses on **a single stock at a time** and supports **multiple trading backends**:
1. **Simulation Mode**: Paper trading with virtual portfolio (default)
2. **Interactive Brokers Paper Trading**: Real IB paper trading account
3. **Interactive Brokers Live Trading**: Real money trading through IB

Daily operations:
1. **Webscraping** for sentiment analysis of the target stock
2. **Price predictions** using a dedicated TSR model 
3. **Trading decisions** based on prediction confidence
4. **Trade execution** through selected backend (simulation/IB)
5. **Single-stock portfolio management** with risk controls

## Files

- `automated_trader.py` - Main automated trading script with IB integration
- `ib_interface.py` - Interactive Brokers API interface
- `test_ib_connection.py` - IB connection testing utility
- `schedule_trader.py` - Scheduling system for daily execution
- `config.json` - Trading configuration and parameters
- `requirements.txt` - Python dependencies including ib-insync
- `README.md` - This documentation

## Quick Start

### 1. Install Dependencies
```bash
pip install apscheduler ib-insync>=0.9.86
```

### 2. Test Different Trading Modes

**Simulation Mode (Default)**
```bash
cd integrations_&_strategy
python automated_trader.py --stock NVDA --dry-run
```

**Interactive Brokers Paper Trading**
```bash
# First test the IB connection
python test_ib_connection.py --mode paper

# Run automated trader with IB paper trading
python automated_trader.py --mode ib_paper --stock NVDA
```

**Interactive Brokers Live Trading**
```bash
# Test connection first
python test_ib_connection.py --mode live

# Run with live trading (use with caution!)
python automated_trader.py --mode ib_live --stock NVDA
```

### 3. Start Daily Automation
```bash
python schedule_trader.py --start
```

## Trading Modes

### 1. Simulation Mode (Default)
- Virtual portfolio with $10,000 starting capital
- No real money involved
- Perfect for testing strategies
- Portfolio state saved to `portfolio_state_simulation.json`

### 2. Interactive Brokers Paper Trading
- Connects to IB paper trading account (port 7496)
- Uses real market data but simulated trades
- Requires IB Gateway or TWS running
- Portfolio synced with actual IB paper account

### 3. Interactive Brokers Live Trading
- Connects to real IB trading account (port 7497) 
- **Real money trading - use with extreme caution!**
- Requires IB Gateway or TWS running
- Portfolio synced with actual IB live account

### Command Line Options
```bash
# Trading mode selection
--mode {simulation,ib_paper,ib_live}    # Default: simulation

# Interactive Brokers settings
--ib-host IB_HOST                       # Default: 127.0.0.1
--ib-client-id IB_CLIENT_ID            # Default: 1

# Safety options
--dry-run                               # Forces simulation mode
```

## Interactive Brokers Setup

### Prerequisites
1. **Interactive Brokers Account**: Active IB account (paper or live)
2. **TWS or IB Gateway**: Download from IB website
3. **API Enabled**: Enable API in TWS settings
4. **Port Configuration**:
   - Paper trading: 7496
   - Live trading: 7497

### Setup Steps
1. **Install IB Gateway/TWS**
2. **Enable API**: TWS → Configure → API → Settings → Enable API
3. **Configure Ports**: Set paper trading port to 7496, live to 7497  
4. **Test Connection**:
   ```bash
   python test_ib_connection.py --mode paper
   python test_ib_connection.py --mode live
   ```

### Troubleshooting IB Connection
- **Connection Refused**: Check if TWS/Gateway is running
- **API Not Enabled**: Enable API in TWS settings
- **Wrong Port**: Verify port configuration (7496 for paper, 7497 for live)
- **Firewall**: Ensure firewall allows connections to IB ports
- **Client ID Conflict**: Try different --ib-client-id if multiple apps

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

### Simulation Mode Examples
```bash
# Basic simulation run
python automated_trader.py --stock AAPL

# Dry run (simulation mode enforced)
python automated_trader.py --dry-run --stock MSFT

# Custom configuration
python automated_trader.py --config my_config.json --stock NVDA
```

### Interactive Brokers Examples
```bash
# IB paper trading
python automated_trader.py --mode ib_paper --stock TSLA

# IB live trading with custom host
python automated_trader.py --mode ib_live --stock GOOGL --ib-host 192.168.1.100

# IB paper trading with different client ID
python automated_trader.py --mode ib_paper --stock AMZN --ib-client-id 2
```

### Testing and Monitoring
```bash
# Test IB connections
python test_ib_connection.py --mode paper
python test_ib_connection.py --mode live

# View scheduled jobs
python schedule_trader.py --status

# Test one trading cycle
python schedule_trader.py --test
python schedule_trader.py --test --stock AAPL  # Test with different stock
```

### Portfolio Management
```bash
# Check simulation portfolio
cat portfolio_state_simulation.json

# Check IB paper portfolio  
cat portfolio_state_ib_paper.json

# Check IB live portfolio
cat portfolio_state_ib_live.json

# View trading decisions by mode
cat trading_decisions_simulation.json
cat trading_decisions_ib_paper.json
cat trading_decisions_ib_live.json
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

- **Multiple Trading Modes**: 
  - Simulation: Virtual trades only, no real money at risk
  - IB Paper: Real IB paper account, simulated trades with real market data
  - IB Live: **Real money trading - use extreme caution!**
- **IB API Security**: Uses secure local connection to IB Gateway/TWS
- **No Stored Credentials**: No API keys or passwords stored in files
- **Local Data**: All portfolio and decision data stored locally
- **Port Isolation**: Paper (7496) and live (7497) trading use separate ports

## Next Steps

1. ✅ **Interactive Brokers Integration**: Paper and live trading implemented
2. **SMS/Email Alerts**: Notifications for trading signals  
3. **Performance Analytics**: Backtesting and performance tracking
4. **Web Dashboard**: Real-time portfolio monitoring
5. **Risk Management**: Advanced position sizing and stop losses
6. **Multi-Timeframe**: Intraday trading capabilities

## Troubleshooting

### Common Issues:

**"No trained models found"**
```bash
cd ../backend_&_algorithms
python main.py train --ticker NVDA --days 730
```

**"Failed to connect to IB"**
```bash
# Check if TWS/Gateway is running
python test_ib_connection.py --mode paper

# Common solutions:
# 1. Start IB Gateway or TWS
# 2. Enable API in TWS settings  
# 3. Check port configuration (7496=paper, 7497=live)
# 4. Try different client ID: --ib-client-id 2
```

**"Prediction failed"**
- Check if ticker symbols are valid
- Verify model files exist in backend
- Ensure webscraping completed successfully

**"ib-insync not found"**
```bash
pip install ib-insync>=0.9.86
```

**"Scheduling not working"**
```bash
pip install apscheduler
python schedule_trader.py --test
```

### Trading Mode Specific Issues:

**Simulation Mode**: Should always work - no external dependencies
**IB Paper Mode**: Requires IB Gateway/TWS running with API enabled
**IB Live Mode**: Same as paper + requires live trading permissions

### Log Files by Mode:
- `automated_trader.log` - General execution logs
- `portfolio_state_[mode].json` - Portfolio state per mode
- `trading_decisions_[mode].json` - Decision history per mode

For support, check the logs and ensure all backend dependencies are installed.