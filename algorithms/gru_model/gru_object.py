from algorithms.gru_model.train_gru import train_gru_model, load_pretrained_weights
from algorithms.gru_model.gru_architecture import GRUPredictor
from algorithms.gru_model.data_pipeline.formatify import format_dataframe_for_gru
from algorithms.gru_model.data_pipeline.yahoo_finance_data import YahooFinanceDataPuller
from algorithms.gru_model.data_pipeline.multi_stock_loader import pull_multiple_stocks, format_multi_stock_data
import numpy as np
import torch

class GRUModel:
    """
    Wrapper for GRU model functionality.

    Usage:
        wrapper = GRUModel(input_size, hidden_size, output_size, config)
        formatted_data = wrapper.format_data(raw_data)
        wrapper.train(formatted_data, epochs=10, lr=0.001)
        predictions = wrapper.predict(input_data)
    """

    def __init__(self, input_size=5, hidden_size=64, output_size=1, config=None):
        """
        Args:
            input_size (int): Number of input features (default 5 for OHLCV: Open, High, Low, Close, Volume)
            hidden_size (int): Number of hidden units in GRU.
            output_size (int): Number of output features.
            config: ConfigLoader instance (optional)
        """
        self.config = config
        self.input_size = input_size

        # Get GRU config if available
        if config:
            gru_config = config.get_gru_config()
            num_layers = gru_config.get('num_layers', 2)
            dropout = gru_config.get('dropout', 0.2)
        else:
            num_layers = 2
            dropout = 0.2

        self.model = GRUPredictor(self.input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
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


    def train(self, epochs=10, lr=0.001, batch_size=32, val_tensor=None, val_target=None):
        """
        Train the GRU model on formatted data.

        Args:
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            val_tensor: Validation input tensor (optional).
            val_target: Validation target tensor (optional).

        Returns:
            Training history dictionary.
        """
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        history = train_gru_model(
            self.model, self.input_tensor, self.target_tensor,
            epochs, lr, batch_size, device=device,
            val_tensor=val_tensor, val_target=val_target
        )
        return history

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

    def pretrain_on_multiple_stocks(self, stock_symbols, period="3y", interval="1d",
                                      epochs=40, lr=0.0005, batch_size=32, validation_split=0.2):
        """
        Pre-train the model on multiple stocks for transfer learning.

        Args:
            stock_symbols: List of stock ticker symbols.
            period: Data period for each stock.
            interval: Data interval.
            epochs: Number of pre-training epochs.
            lr: Pre-training learning rate.
            batch_size: Batch size.
            validation_split: Fraction of data for validation.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print("PHASE 1: PRE-TRAINING ON MULTIPLE STOCKS")
        print(f"{'='*60}\n")

        # Pull data for multiple stocks
        stock_data = pull_multiple_stocks(
            stock_symbols,
            period=period,
            interval=interval,
            data_dir=self.data_dir
        )

        if not stock_data:
            print("Error: No stock data loaded. Cannot pre-train.")
            return None

        # Get sequence length from config
        if self.config:
            gru_config = self.config.get_gru_config()
            sequence_length = gru_config.get('sequence_length', 60)
        else:
            sequence_length = 60

        # Format data for training
        train_tensor, train_target, val_tensor, val_target, scaler_dict, feature_cols = \
            format_multi_stock_data(stock_data, sequence_length=sequence_length,
                                    validation_split=validation_split)

        # Train the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        history = train_gru_model(
            self.model, train_tensor, train_target,
            epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, val_tensor=val_tensor, val_target=val_target
        )

        print(f"\n{'='*60}")
        print("PRE-TRAINING COMPLETE")
        print(f"Final Train Loss: {history['train_losses'][-1]:.4f}")
        if history['val_losses']:
            print(f"Final Val Loss: {history['val_losses'][-1]:.4f}")
        print(f"{'='*60}\n")

        return history

    def finetune_on_target_stock(self, symbol, period="3y", interval="1d",
                                  epochs=25, lr=0.0001, batch_size=32,
                                  pretrained_path=None, validation_split=0.2):
        """
        Fine-tune a pre-trained model on a specific target stock.

        Args:
            symbol: Target stock ticker symbol.
            period: Data period.
            interval: Data interval.
            epochs: Number of fine-tuning epochs.
            lr: Fine-tuning learning rate (typically lower than pre-training).
            batch_size: Batch size.
            pretrained_path: Path to pre-trained weights (optional).
            validation_split: Fraction of data for validation.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 2: FINE-TUNING ON TARGET STOCK ({symbol})")
        print(f"{'='*60}\n")

        # Load pre-trained weights if provided
        if pretrained_path:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = load_pretrained_weights(self.model, pretrained_path, device)

        # Pull and format data for target stock
        self.pull_data(symbol=symbol, period=period, interval=interval)
        self.format_data(for_training=True)

        # Split into train and validation
        total_samples = len(self.input_tensor)
        split_idx = int(total_samples * (1 - validation_split))

        train_tensor = self.input_tensor[:split_idx]
        train_target = self.target_tensor[:split_idx]
        val_tensor = self.input_tensor[split_idx:]
        val_target = self.target_tensor[split_idx:]

        print(f"Training samples: {len(train_tensor)}")
        print(f"Validation samples: {len(val_tensor)}\n")

        # Fine-tune the model with lower learning rate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        history = train_gru_model(
            self.model, train_tensor, train_target,
            epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, val_tensor=val_tensor, val_target=val_target
        )

        # Update stored tensors to full dataset for prediction
        self.input_tensor = torch.cat([train_tensor, val_tensor], dim=0)
        self.target_tensor = torch.cat([train_target, val_target], dim=0)

        print(f"\n{'='*60}")
        print("FINE-TUNING COMPLETE")
        print(f"Final Train Loss: {history['train_losses'][-1]:.4f}")
        if history['val_losses']:
            print(f"Final Val Loss: {history['val_losses'][-1]:.4f}")
        print(f"{'='*60}\n")

        return history