from algorithms.gru_model.train_gru import train_gru_model
from algorithms.gru_model.gru_architecture import GRUPredictor
from algorithms.gru_model.data_pipeline.formatify import format_dataframe_for_gru
from algorithms.gru_model.data_pipeline.yahoo_finance_data import YahooFinanceDataPuller
import numpy as np
import torch

class GRUModel:
    """
    Wrapper for GRU model functionality.

    Usage:
        wrapper = GRUModelWrapper(input_size, hidden_size, output_size)
        formatted_data = wrapper.format_data(raw_data)
        wrapper.train(formatted_data, epochs=10, lr=0.001)
        predictions = wrapper.predict(input_data)
    """



    def __init__(self, input_size=5, hidden_size=64, output_size=1):
        """
        Args:
            input_size (int): Number of input features (default 5 for OHLCV: Open, High, Low, Close, Volume)
            hidden_size (int): Number of hidden units in GRU.
            output_size (int): Number of output features.
        """
        self.input_size = input_size
        self.model = GRUPredictor(self.input_size, hidden_size=64, num_layers=2, dropout=0.2)
        self.data_dir = "./data"
        self.raw_data = None
        self.input_tensor = None
        self.target_tensor = None
        self.scaler = None
        self.output_tensor = None


    def format_data(self, for_training: bool = True):
        """
        Format raw data for GRU model input.

        Args:
            raw_data: Raw input data (e.g., DataFrame).
        Returns:
            Formatted data suitable for GRU traini`ng/prediction.
        """
        if for_training:
            self.input_tensor, self.target_tensor, self.scaler, feature_cols = format_dataframe_for_gru(self.raw_data)
        else:
            self.input_tensor, _, self.scaler, feature_cols = format_dataframe_for_gru(self.raw_data)
        # Set input size from feature columns if not already set
        if self.input_size is None:
            self.input_size = len(feature_cols)

        return None


    def train(self, epochs=10, lr=0.001, batch_size=32):
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_gru_model(self.model, self.input_tensor, self.target_tensor, epochs, lr, batch_size, device=device)
        return None

    def predict(self, input_tensor = None):
        """
        Make predictions using the trained GRU model.

        Args:
            input_data: Formatted input data for prediction.
        Returns:
            Model predictions.
        """

        if input_tensor is None:
            input_tensor = self.input_tensor

        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation for inference
            self.output_tensor = self.model.forward(input_tensor)
            
        return self.output_tensor

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

        puller = YahooFinanceDataPuller()

        puller.data_dir = self.data_dir


        data = puller.get_stock_data(symbol, period, interval)

        puller.save_to_csv(data , symbol)

        self.raw_data = data

        return None;

    def un_normalize(self):
        """
        Un-normalize the model's output using the fitted scaler.
        Returns:
            Un-normalized 'Close' price(s).
        """
        normalized_data = self.output_tensor
        feature_cols = getattr(self, 'feature_cols', None)
        if feature_cols is None:
            try:
                feature_cols = self.scaler.feature_names_in_.tolist()
            except AttributeError:
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_14', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volatility_20']
        if hasattr(normalized_data, 'detach'):
            normalized_data = normalized_data.detach().cpu().numpy()
        normalized_data = np.array(normalized_data).reshape(-1, 1)
        dummy = np.zeros((normalized_data.shape[0], len(feature_cols)))
        close_idx = feature_cols.index('Close')
        dummy[:, close_idx] = normalized_data.squeeze()
        result = self.scaler.inverse_transform(dummy)[:, close_idx]
        return result

    def save_model(self, filepath : str = "algorithms/gru_model/models/cached_gru_model.pth"):
        torch.save(self.model.state_dict(), filepath)
        return None

    def load_model(self, filepath : str = "algorithms/gru_model/models/cached_gru_model.pth"):
        self.model.load_state_dict(torch.load(filepath))
        return None