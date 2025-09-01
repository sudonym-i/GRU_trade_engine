#!/usr/bin/env python3
"""
Comprehensive Test Suite for TSR (Time Series Regression) Model Components

Tests model.py, data_pipeline.py, utils.py, train.py, trade.py, route.py, and visualizations.py
to ensure proper functionality of the time series prediction and trading system.

Author: ML Trading Bot Project
Purpose: Validate TSR model implementation
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
import warnings
from pathlib import Path
import logging

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for dependencies
TORCH_AVAILABLE = True
PANDAS_AVAILABLE = True
REQUESTS_AVAILABLE = True
MATPLOTLIB_AVAILABLE = True

try:
    import torch
    import torch.nn as nn
    import numpy as np
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/NumPy not available. Some tests will be skipped.")

try:
    import pandas as pd
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not available. Some tests will be skipped.")

try:
    import requests
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Some tests will be skipped.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Visualization libraries not available. Some tests will be skipped.")

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

class TestGRUModel(unittest.TestCase):
    """Test cases for model.py - GRU Predictor"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from model import GRUPredictor
        self.GRUPredictor = GRUPredictor
    
    def test_gru_model_initialization(self):
        """Test GRU model initialization."""
        input_dim = 8
        hidden_dim = 64
        num_layers = 2
        
        model = self.GRUPredictor(input_dim, hidden_dim, num_layers)
        
        self.assertIsInstance(model.gru, nn.GRU)
        self.assertIsInstance(model.fc1, nn.Linear)
        self.assertIsInstance(model.fc2, nn.Linear)
        self.assertIsInstance(model.dropout, nn.Dropout)
        
        # Check dimensions
        self.assertEqual(model.gru.input_size, input_dim)
        self.assertEqual(model.gru.hidden_size, hidden_dim)
        self.assertEqual(model.gru.num_layers, num_layers)
        self.assertEqual(model.fc1.in_features, hidden_dim)
        self.assertEqual(model.fc1.out_features, 64)
        self.assertEqual(model.fc2.in_features, 64)
        self.assertEqual(model.fc2.out_features, 1)
    
    def test_gru_model_forward_pass(self):
        """Test GRU model forward pass."""
        batch_size = 4
        seq_length = 10
        input_dim = 6
        
        model = self.GRUPredictor(input_dim)
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_gru_model_training_mode(self):
        """Test model can switch between training and evaluation modes."""
        model = self.GRUPredictor(input_dim=5)
        
        # Test training mode
        model.train()
        self.assertTrue(model.training)
        
        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)


class TestDataPipeline(unittest.TestCase):
    """Test cases for data_pipeline.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not (PANDAS_AVAILABLE and REQUESTS_AVAILABLE):
            self.skipTest("Required dependencies not available")
        
        from data_pipeline import DataLoader, make_dataset
        self.DataLoader = DataLoader
        self.make_dataset = make_dataset
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        # Test single ticker with mock API key
        loader = self.DataLoader("AAPL", "2023-01-01", "2023-01-31", api_key="test_key")
        self.assertEqual(loader.tickers, ["AAPL"])
        self.assertEqual(loader.start, "2023-01-01")
        self.assertEqual(loader.end, "2023-01-31")
        self.assertEqual(loader.api_key, "test_key")
        
        # Test multiple tickers
        tickers = ["AAPL", "GOOGL", "MSFT"]
        loader = self.DataLoader(tickers, "2023-01-01", "2023-01-31", api_key="test_key")
        self.assertEqual(loader.tickers, tickers)
    
    @patch('requests.get')
    def test_data_loader_download_success(self, mock_requests_get):
        """Test successful data download."""
        # Mock FMP API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'historical': [
                {'date': '2023-01-03', 'open': 100, 'high': 105, 'low': 99, 'close': 104, 'volume': 1000},
                {'date': '2023-01-02', 'open': 101, 'high': 106, 'low': 100, 'close': 105, 'volume': 1100},
                {'date': '2023-01-01', 'open': 102, 'high': 107, 'low': 101, 'close': 106, 'volume': 1200}
            ]
        }
        mock_requests_get.return_value = mock_response
        
        loader = self.DataLoader("AAPL", "2023-01-01", "2023-01-31", api_key="test_key")
        data = loader.download()
        
        self.assertIn("AAPL", data)
        self.assertIsInstance(data["AAPL"], pd.DataFrame)
        mock_requests_get.assert_called_once()
    
    @patch('requests.get')
    def test_data_loader_download_empty(self, mock_requests_get):
        """Test handling of empty data download."""
        # Mock empty response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'historical': []}
        mock_requests_get.return_value = mock_response
        
        loader = self.DataLoader("INVALID", "2023-01-01", "2023-01-31", api_key="test_key")
        data = loader.download()
        
        self.assertNotIn("INVALID", data)
    
    def test_data_loader_get_methods(self):
        """Test data retrieval methods."""
        loader = self.DataLoader("AAPL", "2023-01-01", "2023-01-31", api_key="test_key")
        
        # Mock some data
        mock_data = pd.DataFrame({'Close': [100, 101, 102]})
        loader.data = {"AAPL": mock_data}
        
        # Test get method
        retrieved_data = loader.get("AAPL")
        pd.testing.assert_frame_equal(retrieved_data, mock_data)
        
        # Test get_all method
        all_data = loader.get_all()
        self.assertEqual(all_data, loader.data)


class TestUtils(unittest.TestCase):
    """Test cases for utils.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not (PANDAS_AVAILABLE and TORCH_AVAILABLE):
            self.skipTest("Required dependencies not available")
        
        from utils import add_technical_indicators, create_sequences
        self.add_technical_indicators = add_technical_indicators
        self.create_sequences = create_sequences
    
    def test_add_technical_indicators(self):
        """Test technical indicators calculation."""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = [100 + i + np.random.randn() * 0.5 for i in range(50)]
        
        df = pd.DataFrame({
            'Close': prices,
            'Open': [p - 1 for p in prices],
            'High': [p + 2 for p in prices],
            'Low': [p - 2 for p in prices],
            'Volume': [1000] * 50
        }, index=dates)
        
        df_with_indicators = self.add_technical_indicators(df.copy())
        
        # Check that indicators were added
        self.assertIn('SMA_14', df_with_indicators.columns)
        self.assertIn('RSI_14', df_with_indicators.columns)
        self.assertIn('MACD', df_with_indicators.columns)
        
        # Check RSI bounds (should be between 0 and 100)
        rsi_values = df_with_indicators['RSI_14'].dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())
        
        # Check SMA values are reasonable
        sma_values = df_with_indicators['SMA_14'].dropna()
        self.assertTrue(len(sma_values) > 0)
    
    def test_create_sequences(self):
        """Test sequence creation for time series."""
        # Create test data
        df = pd.DataFrame({
            'Close': [i for i in range(20)],
            'Volume': [1000] * 20,
            'Feature1': [i * 2 for i in range(20)]
        })
        
        seq_length = 5
        X, y = self.create_sequences(df, seq_length)
        
        # Check shapes
        expected_samples = len(df) - seq_length
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], seq_length)
        self.assertEqual(X.shape[2], len(df.columns))
        self.assertEqual(y.shape[0], expected_samples)
        
        # Check that targets are correct (should be Close values)
        for i in range(expected_samples):
            expected_target = df.iloc[i + seq_length]['Close']
            self.assertEqual(y[i], expected_target)
    
    def test_create_sequences_edge_cases(self):
        """Test create_sequences with edge cases."""
        # Test with small dataframe
        small_df = pd.DataFrame({'Close': [1, 2, 3]})
        
        # Sequence length larger than data
        X, y = self.create_sequences(small_df, seq_length=5)
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)
        
        # Sequence length equal to data length
        X, y = self.create_sequences(small_df, seq_length=3)
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)


class TestTraining(unittest.TestCase):
    """Test cases for train.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from train import train_gru_predictor
        from model import GRUPredictor
        self.train_gru_predictor = train_gru_predictor
        self.GRUPredictor = GRUPredictor
    
    def test_train_gru_predictor_setup(self):
        """Test training function setup and execution."""
        # Create a simple model and dataset
        model = self.GRUPredictor(input_dim=3, hidden_dim=16, num_layers=1)
        
        # Create dummy dataset
        X = torch.randn(20, 5, 3)  # 20 samples, seq_len=5, input_dim=3
        y = torch.randn(20, 1)     # 20 targets
        dataset = torch.utils.data.TensorDataset(X, y)
        
        # Mock visualization to avoid display issues in tests
        with patch('train.plot_training_loss'):
            losses = self.train_gru_predictor(
                model=model,
                dataset=dataset,
                epochs=2,
                batch_size=8,
                lr=1e-3,
                plot_loss=False
            )
        
        # Check that losses were returned
        self.assertEqual(len(losses), 2)  # 2 epochs
        self.assertIsInstance(losses[0], float)
        self.assertTrue(all(loss >= 0 for loss in losses))


class TestTradingSimulator(unittest.TestCase):
    """Test cases for trade.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from trade import TradingSimulator
        self.TradingSimulator = TradingSimulator
    
    def test_trading_simulator_initialization(self):
        """Test TradingSimulator initialization."""
        initial_balance = 5000
        simulator = self.TradingSimulator(initial_balance)
        
        self.assertEqual(simulator.initial_balance, initial_balance)
        self.assertEqual(simulator.balance, initial_balance)
        self.assertEqual(simulator.position, 0)
        self.assertEqual(len(simulator.trades), 0)
    
    def test_trading_simulator_reset(self):
        """Test simulator reset functionality."""
        simulator = self.TradingSimulator(1000)
        
        # Modify state
        simulator.balance = 1500
        simulator.position = 10
        simulator.trades = ['trade1', 'trade2']
        
        # Reset
        simulator.reset()
        
        self.assertEqual(simulator.balance, 1000)
        self.assertEqual(simulator.position, 0)
        self.assertEqual(len(simulator.trades), 0)
    
    def test_trading_simulator_buy_action(self):
        """Test buy trading action."""
        simulator = self.TradingSimulator(1000)
        
        current_price = 100
        predicted_price = 110  # Higher than current, should trigger buy
        
        with patch('builtins.print'):  # Suppress print statements
            trade_info = simulator.step(current_price, predicted_price)
        
        # Should have bought shares
        expected_shares = 1000 / 100  # balance / price
        self.assertAlmostEqual(simulator.position, expected_shares, places=4)
        self.assertEqual(simulator.balance, 0)
        self.assertEqual(trade_info['action'], 'buy')
        self.assertEqual(trade_info['buy_price'], current_price)
    
    def test_trading_simulator_sell_action(self):
        """Test sell trading action."""
        simulator = self.TradingSimulator(1000)
        
        # First buy some shares
        buy_price = 100
        predicted_buy_price = 110
        with patch('builtins.print'):
            simulator.step(buy_price, predicted_buy_price)
        
        # Now sell
        current_price = 120
        predicted_price = 110  # Lower than current, should trigger sell
        
        with patch('builtins.print'):
            trade_info = simulator.step(current_price, predicted_price)
        
        # Should have sold all shares
        self.assertEqual(simulator.position, 0)
        expected_balance = (1000 / buy_price) * current_price  # shares * sell_price
        self.assertAlmostEqual(simulator.balance, expected_balance, places=2)
        self.assertEqual(trade_info['action'], 'sell')
        self.assertEqual(trade_info['sell_price'], current_price)
    
    def test_trading_simulator_hold_action(self):
        """Test hold trading action."""
        simulator = self.TradingSimulator(1000)
        
        current_price = 100
        predicted_price = 99  # Lower than current, but no position to sell
        
        with patch('builtins.print'):
            trade_info = simulator.step(current_price, predicted_price)
        
        # Should hold (no action)
        self.assertEqual(simulator.balance, 1000)
        self.assertEqual(simulator.position, 0)
        self.assertEqual(trade_info['action'], 'hold')


class TestRoute(unittest.TestCase):
    """Test cases for route.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")
        
        # Create mock config
        self.mock_config = {
            "tsr_model": {
                "training": {
                    "tickers": ["AAPL"],
                    "start": "2023-01-01",
                    "end": "2023-01-31",
                    "seq_length": 10,
                    "interval": "1d",
                    "epochs": 2,
                    "batch_size": 16,
                    "lr": 1e-3
                },
                "trading": {
                    "tickers": ["AAPL"],
                    "start": "2023-02-01",
                    "end": "2023-02-28",
                    "seq_length": 10,
                    "interval": "1d",
                    "initial_balance": 1000
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.mock_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('route.make_dataset')
    @patch('route.train_gru_predictor')
    def test_train_model_function(self, mock_train_gru, mock_make_dataset):
        """Test train_model function."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from route import train_model
        from model import GRUPredictor
        
        # Mock dataset creation
        mock_dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 5, 6), 
            torch.randn(10, 1)
        )
        mock_make_dataset.return_value = (mock_dataset, 6)
        
        # Mock training
        mock_train_gru.return_value = [0.1, 0.05]
        
        # Patch config file path
        with patch('route.os.path.dirname') as mock_dirname:
            # Mock the path resolution to use our temp config
            mock_dirname.return_value = os.path.dirname(self.config_path)
            
            with patch('route.os.path.join', return_value=self.config_path):
                with patch('builtins.print'):  # Suppress prints
                    model = train_model()
        
        # Check that functions were called
        mock_make_dataset.assert_called_once()
        mock_train_gru.assert_called_once()
        self.assertIsInstance(model, GRUPredictor)
    
    def test_config_file_not_found(self):
        """Test handling of missing config file."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from route import train_model
        
        with patch('route.os.path.join', return_value="nonexistent_config.json"):
            with self.assertRaises(FileNotFoundError):
                train_model()
    
    def test_invalid_config_structure(self):
        """Test handling of invalid config structure."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from route import train_model
        
        # Create invalid config
        invalid_config = {"wrong_structure": {}}
        invalid_config_path = os.path.join(self.temp_dir, "invalid_config.json")
        
        with open(invalid_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        with patch('route.os.path.join', return_value=invalid_config_path):
            with self.assertRaises(KeyError):
                train_model()


class TestVisualizations(unittest.TestCase):
    """Test cases for visualizations.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MATPLOTLIB_AVAILABLE:
            self.skipTest("Visualization libraries not available")
        
        from visualizations import plot_training_loss
        self.plot_training_loss = plot_training_loss
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_training_loss(self, mock_show, mock_savefig):
        """Test training loss plotting."""
        losses = [0.5, 0.3, 0.2, 0.15, 0.1]
        
        # Test with custom save path
        save_path = "test_loss.png"
        
        with patch('builtins.print'):  # Suppress print statements
            self.plot_training_loss(losses, save_path)
        
        # Check that savefig was called
        mock_savefig.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('os.system')  # Mock system call for opening image
    def test_plot_training_loss_auto_save(self, mock_system, mock_savefig):
        """Test training loss plotting with auto-save."""
        losses = [0.5, 0.3, 0.2]
        
        with patch('builtins.print'):
            self.plot_training_loss(losses)  # No save_path provided
        
        # Should still save the plot
        mock_savefig.assert_called()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete TSR system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_component_imports(self):
        """Test that all components can be imported together."""
        try:
            if TORCH_AVAILABLE:
                from model import GRUPredictor
                from train import train_gru_predictor
                from trade import TradingSimulator
                from route import train_model
            
            if PANDAS_AVAILABLE:
                from data_pipeline import DataLoader, make_dataset
                from utils import add_technical_indicators, create_sequences
            
            if MATPLOTLIB_AVAILABLE:
                from visualizations import plot_training_loss
            
            integration_success = True
            
        except ImportError as e:
            integration_success = False
            print(f"Integration import failed: {e}")
        
        self.assertTrue(integration_success)
    
    @unittest.skipUnless(TORCH_AVAILABLE and PANDAS_AVAILABLE, "Required dependencies not available")
    def test_data_flow_compatibility(self):
        """Test that data flows correctly between components."""
        from utils import create_sequences
        import pandas as pd
        import torch
        
        # Create test dataframe
        df = pd.DataFrame({
            'Close': [100 + i for i in range(20)],
            'Volume': [1000] * 20,
            'SMA_14': [100 + i for i in range(20)]
        })
        
        # Test sequence creation
        X, y = create_sequences(df, seq_length=5)
        
        # Test that sequences can be converted to PyTorch dataset
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), 
            torch.FloatTensor(y).unsqueeze(1)
        )
        
        # Test that dataset works with DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Check that we can iterate through the dataloader
        batch_count = 0
        for batch_X, batch_y in dataloader:
            self.assertEqual(batch_X.shape[0], min(2, len(dataset)))  # batch_size or remaining
            self.assertEqual(batch_y.shape[0], min(2, len(dataset)))
            batch_count += 1
        
        self.assertGreater(batch_count, 0)


class TestRunner:
    """Custom test runner with detailed reporting for TSR model."""
    
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
    
    def run_test_suite(self):
        """Run the complete TSR model test suite."""
        print("=" * 70)
        print("TSR (TIME SERIES REGRESSION) MODEL TEST SUITE")
        print("=" * 70)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        test_classes = [
            TestGRUModel,
            TestDataPipeline,
            TestUtils,
            TestTraining,
            TestTradingSimulator,
            TestRoute,
            TestVisualizations,
            TestIntegration
        ]
        
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            buffer=True
        )
        
        print(f"\nRunning {suite.countTestCases()} tests...")
        print("-" * 70)
        
        result = runner.run(suite)
        
        # Print summary
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"Tests run: {result.testsRun}")
        print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.failures:
            print(f"\nFAILED TESTS:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print(f"\nERROR TESTS:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        # Environment and dependency info
        print(f"\nENVIRONMENT:")
        print(f"  - PyTorch Available: {TORCH_AVAILABLE}")
        print(f"  - Pandas Available: {PANDAS_AVAILABLE}")
        print(f"  - Requests Available: {REQUESTS_AVAILABLE}")
        print(f"  - Matplotlib Available: {MATPLOTLIB_AVAILABLE}")
        print(f"  - Python Version: {sys.version}")
        
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                       result.testsRun * 100) if result.testsRun > 0 else 0
        print(f"  - Success Rate: {success_rate:.1f}%")
        
        print("=" * 70)
        
        return result.wasSuccessful()


def main():
    """Main function to run TSR model tests."""
    print("TSR Model Test Suite")
    print("Testing model.py, data_pipeline.py, utils.py, train.py, trade.py, route.py, and visualizations.py")
    
    # Check dependencies
    missing_deps = []
    if not TORCH_AVAILABLE:
        missing_deps.append("torch")
    if not PANDAS_AVAILABLE:
        missing_deps.append("pandas")
    if not REQUESTS_AVAILABLE:
        missing_deps.append("requests")
    if not MATPLOTLIB_AVAILABLE:
        missing_deps.extend(["matplotlib", "seaborn", "plotly"])
    
    if missing_deps:
        print(f"\nWarning: Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests may be skipped. Install with:")
        print("pip install torch pandas requests matplotlib seaborn plotly")
    
    # Run tests
    test_runner = TestRunner()
    success = test_runner.run_test_suite()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()