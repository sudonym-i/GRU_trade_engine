# Financial Modeling Prep API Setup

The TSR model now uses Financial Modeling Prep (FMP) API instead of Yahoo Finance for market data.

## Getting an API Key

1. Visit [Financial Modeling Prep](https://financialmodelingprep.com/)
2. Sign up for a free account
3. Navigate to your dashboard to get your API key
4. Free tier includes 250 requests per day

## Setting Up the API Key

### Option 1: .env File (Recommended for Development)
1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` file and replace with your actual API key:
```
FMP_API_KEY=your_actual_api_key_here
```

The code will automatically load this when you import the data_pipeline module.

### Option 2: Environment Variable
```bash
export FMP_API_KEY=your_actual_api_key_here
```

### Option 3: Pass directly to DataLoader
```python
from data_pipeline import DataLoader

loader = DataLoader("AAPL", "2023-01-01", "2023-12-31", api_key="your_actual_api_key_here")
```

## Supported Features

### Data Intervals
- `1d` - Daily data (default)
- `1h` - Hourly data
- `5m` - 5-minute data
- `15m` - 15-minute data  
- `30m` - 30-minute data

### Data Fields
The API returns the same fields as the original Yahoo Finance implementation:
- `Open` - Opening price
- `High` - Highest price
- `Low` - Lowest price
- `Close` - Closing price
- `Volume` - Trading volume

### Example Usage

```python
from data_pipeline import DataLoader

# The API key will be loaded automatically from .env file
# No need to set it manually if you have a .env file

# Create DataLoader
loader = DataLoader("AAPL", "2023-01-01", "2023-12-31")

# Download data
data = loader.download()

# Access data
aapl_data = loader.get("AAPL")
print(aapl_data.head())
```

## Testing

Run the integration test to verify your setup:

```bash
python tests/test_fmp_integration.py
```

## API Limits

- **Free tier**: 250 requests per day
- **Basic plan**: 25,000 requests per month ($14/month)  
- **Professional plan**: 100,000 requests per month ($29/month)

For development and testing, the free tier should be sufficient. For production use with multiple tickers or frequent data updates, consider upgrading to a paid plan.

## Troubleshooting

### 401 Unauthorized Error
- Check that your API key is correct
- Ensure the API key is properly set in environment variable or passed to DataLoader
- Verify your API key hasn't expired

### No Data Retrieved
- Check that the ticker symbol is valid
- Verify the date range is reasonable (not too far in the past or future)
- Ensure you haven't exceeded your API rate limits

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're running Python from the correct directory