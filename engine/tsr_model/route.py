import json
import os
try:
    from .model import GRUPredictor
    from .data_pipeline import make_dataset
    from .train import train_gru_predictor
    from .trade import simulate_trading
except ImportError:
    from model import GRUPredictor
    from data_pipeline import make_dataset
    from train import train_gru_predictor
    from trade import simulate_trading


def train_model():
    """
    Trains the GRU predictor model using configuration from the master config file.
    
    Returns:
        model: Trained GRUPredictor model
    """
    # Get the path to the master config file relative to this module
    # Current file is in src/tsr_model/, config is in root/
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
    
    try:
        with open(config_path, "r") as fil:
            master_config = json.load(fil)
        
        # Extract training configuration from the master config
        train_config = master_config["tsr_model"]["training"]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Master config file not found at: {config_path}")
    except KeyError as e:
        raise KeyError(f"Missing configuration key in master config: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in master config file: {e}")

    # Extract training parameters from the nested configuration
    train_tickers = train_config["tickers"]
    train_start = train_config["start"]
    train_end = train_config["end"]
    train_seq_length = train_config["seq_length"]
    train_interval = train_config["interval"]
    train_epochs = train_config["epochs"]
    train_batch_size = train_config["batch_size"]
    train_lr = train_config["lr"]

    print(" -------------- TRAINING --------------")
    
    print(f"Tickers: {train_tickers}, Start: {train_start}, End: {train_end}, Interval: {train_interval}")

    dataset, input_dim = make_dataset(train_tickers, train_start, train_end, train_seq_length, train_interval, normalize=True)
    model = GRUPredictor(input_dim=input_dim)
    train_gru_predictor(model, dataset, epochs=train_epochs, batch_size=train_batch_size, lr=train_lr)

    return model




def test_run_model(model):
    """
    Tests the GRU predictor model using configuration from the master config file.
    
    Args:
        model: Trained GRUPredictor model to test
    """
    # Get the path to the master config file relative to this module
    # Current file is in src/tsr_model/, config is in root/
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
    
    try:
        with open(config_path, "r") as fil:
            master_config = json.load(fil)
        
        # Extract testing configuration from the master config
        test_config = master_config["tsr_model"]["testing"]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Master config file not found at: {config_path}")
    except KeyError as e:
        raise KeyError(f"Missing configuration key in master config: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in master config file: {e}")

    # Extract testing parameters from the nested configuration
    test_tickers = test_config["tickers"]
    test_start = test_config["start"]
    test_end = test_config["end"]
    test_seq_length = test_config["seq_length"]
    test_interval = test_config["interval"]
    test_epochs = test_config["epochs"]
    test_batch_size = test_config["batch_size"]
    test_lr = test_config["lr"]

    print(" -------------- TESTING --------------")

    print(f"Tickers: {test_tickers}, Start: {test_start}, End: {test_end}, Interval: {test_interval}")

    testDataset, input_dim = make_dataset(test_tickers, test_start, test_end, test_seq_length, test_interval, normalize=False)
    simulate_trading(model, testDataset, testDataset.tensors[1].numpy())

    return