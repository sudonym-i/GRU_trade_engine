
from train_gru import train_gru_model
from gru_architecture import GRUPredictor
from data_pipeline.formatify import format_dataframe_for_gru


class GRUModel:
    """
    Wrapper for GRU model functionality.

    Usage:
        wrapper = GRUModelWrapper(input_size, hidden_size, output_size)
        formatted_data = wrapper.format_data(raw_data)
        wrapper.train(formatted_data, epochs=10, lr=0.001)
        predictions = wrapper.predict(input_data)
    """
    def __init__(self, input_size, hidden_size, output_size):
        """

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in GRU.
            output_size (int): Number of output features.
        """
        self.predictor = GRUPredictor(input_size, hidden_size, output_size)
        self.data_dir = "./data"

    def format_data(self, raw_data):
        """
        Format raw data for GRU model input.

        Args:
            raw_data: Raw input data (e.g., DataFrame).
        Returns:
            Formatted data suitable for GRU training/prediction.
        """
        return format_dataframe_for_gru(raw_data)

    def train(self, train_data, epochs=10, lr=0.001):
        """
        Train the GRU model.

        Args:
            train_data: Formatted training data.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        train_gru_model(self.predictor, train_data[:-1], train_data[-1], epochs, lr)

    def predict(self, input_data):
        """
        Make predictions using the trained GRU model.

        Args:
            input_data: Formatted input data for prediction.
        Returns:
            Model predictions.
        """
        return self.predictor.forward(input_data)
    
    def pull_data(self, symbol: str, period: str = "3y", interval: str = "1d"):
        """
        Pull data using YahooFinanceDataPuller.

        Args:
            symbol (str): Stock ticker symbol.
            period (str): Data period (default: '3y' for 3 years).
            interval (str): Data interval (default: '1d' for daily).
        Returns:
            Fetched stock data as a DataFrame or None if failed.
        """
        from .data_pipeline.yahoo_finance_data import YahooFinanceDataPuller

        puller = YahooFinanceDataPuller()

        puller.data_dir = self.data_dir
        
        data = puller.get_stock_data(symbol, period, interval)

        puller.save_to_csv(data , symbol)

        return data
