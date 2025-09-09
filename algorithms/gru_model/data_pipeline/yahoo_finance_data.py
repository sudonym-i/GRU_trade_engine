
import yfinance as yf
import pandas as pd
import datetime as dt


class YahooFinanceDataPuller:
    """Class to handle Yahoo Finance data pulling operations"""
    
    def __init__(self, data_dir: str = "market_data"):
        self.data_dir = data_dir


    def get_stock_data(self, symbol: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
              
        Args:  
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period (default: '3y' for 3 years)
            interval: Data interval (default: '1d' for daily)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """

        try:
            print(f"Fetching data for {symbol} - Period: {period}, Interval: {interval}")

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data found for symbol: {symbol}")
                return None
            
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)

            print(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    
    def save_to_csv(self, data: pd.DataFrame, symbol: str):
        """Save data to CSV file"""
        filename = f"{self.data_dir}/{symbol}_{dt.date.today().strftime('%Y%m%d')}.csv"
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    