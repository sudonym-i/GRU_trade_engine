#!/usr/bin/env python3
"""
Comprehensive Test Suite for Sentiment Model Components

Tests tokenize_pipeline.py, model.py, and route.py functionality
to ensure proper integration and performance of the sentiment analysis system.

Author: ML Trading Bot Project
Purpose: Validate sentiment model implementation
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch and transformers not available. Some tests will be skipped.")

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing

class TestTokenizePipeline(unittest.TestCase):
    """Test cases for tokenize_pipeline.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from tokenize_pipeline import TokenizationConfig, TextPreprocessor, TokenizationPipeline
        self.TokenizationConfig = TokenizationConfig
        self.TextPreprocessor = TextPreprocessor
        self.TokenizationPipeline = TokenizationPipeline
        
        # Create test data
        self.test_text = """
        1→This is a test transcript about NVIDIA stock.
        2→The company's performance has been excellent this quarter.
        3→Market sentiment seems very positive regarding their AI chips.
        4→Some investors are concerned about valuation levels.
        """
        
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.test_raw_file = Path(self.temp_dir) / "test.raw"
        with open(self.test_raw_file, 'w') as f:
            f.write(self.test_text)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tokenization_config_initialization(self):
        """Test TokenizationConfig initialization."""
        config = self.TokenizationConfig()
        
        self.assertEqual(config.tokenizer_model, "bert-base-uncased")
        self.assertEqual(config.max_sequence_length, 128)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.random_seed, 42)
    
    def test_text_preprocessor(self):
        """Test TextPreprocessor functionality."""
        preprocessor = self.TextPreprocessor()
        
        # Test text cleaning
        dirty_text = "1→This is NVIDIA with 50 % gains!!!"
        cleaned = preprocessor.clean_text(dirty_text)
        
        self.assertIn("NVIDIA", cleaned)  # Should normalize ticker
        self.assertIn("50%", cleaned)     # Should normalize percentage
        self.assertNotIn("1→", cleaned)   # Should remove line numbers
    
    def test_text_segmentation(self):
        """Test text segmentation."""
        preprocessor = self.TextPreprocessor()
        
        long_text = "This is a long sentence. " * 10
        segments = preprocessor.segment_text(long_text, min_length=50)
        
        self.assertGreater(len(segments), 0)
        for segment in segments:
            self.assertGreaterEqual(len(segment), 50)
    
    @patch('tokenize_pipeline.AutoTokenizer.from_pretrained')
    def test_tokenization_pipeline_initialization(self, mock_tokenizer):
        """Test TokenizationPipeline initialization."""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        config = self.TokenizationConfig()
        config.output_dir = self.temp_dir
        
        pipeline = self.TokenizationPipeline(config)
        
        self.assertIsNotNone(pipeline.tokenizer)
        self.assertIsInstance(pipeline.preprocessor, self.TextPreprocessor)
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        with patch('tokenize_pipeline.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            config = self.TokenizationConfig()
            config.output_dir = self.temp_dir
            config.min_text_length = 30
            
            pipeline = self.TokenizationPipeline(config)
            segments = pipeline.preprocess_data(self.test_text)
            
            self.assertGreater(len(segments), 0)
            for segment in segments:
                self.assertGreaterEqual(len(segment), 30)


class TestBertSentimentModel(unittest.TestCase):
    """Test cases for model.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        from model import ModelConfig, BertSentimentModel, SentimentTrainer
        self.ModelConfig = ModelConfig
        self.BertSentimentModel = BertSentimentModel
        self.SentimentTrainer = SentimentTrainer
    
    def test_model_config_initialization(self):
        """Test ModelConfig initialization."""
        config = self.ModelConfig()
        
        self.assertEqual(config.bert_model_name, "bert-base-uncased")
        self.assertEqual(config.num_classes, 3)
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.hidden_dim, 768)
    
    @patch('model.BertModel.from_pretrained')
    def test_bert_model_initialization(self, mock_bert):
        """Test BertSentimentModel initialization."""
        # Mock BERT model
        mock_bert_instance = MagicMock()
        mock_bert_instance.config.hidden_size = 768
        mock_bert.return_value = mock_bert_instance
        
        config = self.ModelConfig()
        model = self.BertSentimentModel(config)
        
        self.assertEqual(model.num_classes, 3)
        self.assertIsNotNone(model.classifier)
        self.assertIsNotNone(model.dropout)
    
    @patch('model.BertModel.from_pretrained')
    def test_model_forward_pass(self, mock_bert):
        """Test model forward pass."""
        # Mock BERT model outputs
        mock_bert_instance = MagicMock()
        mock_pooler_output = torch.randn(2, 768)  # batch_size=2, hidden_dim=768
        mock_bert_instance.return_value.pooler_output = mock_pooler_output
        mock_bert.return_value = mock_bert_instance
        
        config = self.ModelConfig()
        model = self.BertSentimentModel(config)
        
        # Mock forward pass of BERT
        with patch.object(model.bert, '__call__', return_value=MagicMock(pooler_output=mock_pooler_output)):
            # Test input tensors
            batch_size, seq_len = 2, 128
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Check output shape
            self.assertEqual(logits.shape, (batch_size, config.num_classes))
    
    def test_model_predictions(self):
        """Test model prediction functionality."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        with patch('model.BertModel.from_pretrained'):
            config = self.ModelConfig()
            model = self.BertSentimentModel(config)
            
            # Test logits
            batch_size = 3
            logits = torch.randn(batch_size, config.num_classes)
            
            predictions, probabilities = model.get_predictions(logits)
            
            self.assertEqual(predictions.shape, (batch_size,))
            self.assertEqual(probabilities.shape, (batch_size, config.num_classes))
            
            # Check probabilities sum to 1
            prob_sums = torch.sum(probabilities, dim=1)
            self.assertTrue(torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6))


class TestSentimentRoute(unittest.TestCase):
    """Test cases for route.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        from route import pull_from_web, analyze_sentiment
        self.pull_from_web = pull_from_web
        self.analyze_sentiment = analyze_sentiment
        
        # Create temporary test environment
        self.temp_dir = tempfile.mkdtemp()
        self.test_youtube_file = Path(self.temp_dir) / "youtube.raw"
        
        # Create test YouTube data
        test_youtube_content = """
        This is test YouTube transcript data about NVIDIA.
        The stock performance has been remarkable this year.
        Many investors are optimistic about future growth.
        However, some analysts express concerns about valuation.
        The AI chip market continues to expand rapidly.
        """
        
        with open(self.test_youtube_file, 'w') as f:
            f.write(test_youtube_content)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    @patch('os.chdir')
    @patch('pathlib.Path.exists')
    def test_pull_from_web_success(self, mock_exists, mock_chdir, mock_subprocess):
        """Test successful web scraping execution."""
        # Mock file system
        mock_exists.return_value = True
        
        # Mock subprocess success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Scraping completed successfully"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Mock glob for output files
        with patch('pathlib.Path.glob') as mock_glob:
            mock_file = MagicMock()
            mock_file.name = "youtube_nvidia.raw"
            mock_glob.return_value = [mock_file]
            
            result = self.pull_from_web()
            
            self.assertTrue(result["success"])
            self.assertEqual(result["exit_code"], 0)
            self.assertTrue(result["data_collected"])
    
    @patch('pathlib.Path.exists')
    def test_pull_from_web_missing_executable(self, mock_exists):
        """Test web scraping with missing executable."""
        mock_exists.return_value = False
        
        result = self.pull_from_web()
        
        self.assertFalse(result["success"])
        self.assertEqual(result["exit_code"], -1)
        self.assertFalse(result["data_collected"])
    
    @patch('pathlib.Path.exists')
    def test_analyze_sentiment_missing_file(self, mock_exists):
        """Test sentiment analysis with missing YouTube file."""
        mock_exists.return_value = False
        
        result = self.analyze_sentiment()
        
        self.assertFalse(result["success"])
        self.assertEqual(len(result["predictions"]), 0)
    
    @patch('route.Path')
    def test_analyze_sentiment_with_mock_data(self, mock_path):
        """Test sentiment analysis with mock data."""
        # Mock file system
        mock_youtube_path = MagicMock()
        mock_youtube_path.exists.return_value = True
        mock_path.return_value.resolve.return_value.parent = Path(self.temp_dir)
        mock_path.return_value.__truediv__ = lambda self, other: self.temp_dir / other if other == "youtube.raw" else MagicMock()
        
        # Mock file reading
        test_content = "This is positive news about NVIDIA stock performance."
        
        with patch('builtins.open', mock_open_multiple_files({str(self.test_youtube_file): test_content})):
            with patch('route.TokenizationPipeline') as mock_pipeline_class:
                mock_pipeline = MagicMock()
                mock_pipeline.preprocess_data.return_value = ["This is positive news about NVIDIA stock performance."]
                mock_pipeline_class.return_value = mock_pipeline
                
                with patch('route.Path.exists', return_value=False):  # No trained model
                    result = self.analyze_sentiment()
                    
                    self.assertTrue(result["success"])
                    self.assertGreater(len(result["predictions"]), 0)
                    self.assertIn("statistics", result)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete sentiment analysis system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config_data = {
            "sentiment_model": {
                "tokenization": {
                    "tokenizer_model": "bert-base-uncased",
                    "train_split": 0.8,
                    "batch_size": 8
                },
                "model": {
                    "learning_rate": 1e-5,
                    "num_epochs": 2
                }
            }
        }
        
        self.config_file = Path(self.temp_dir) / "config.json"
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config_data, f)
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_config_loading_integration(self):
        """Test configuration loading across modules."""
        from tokenize_pipeline import TokenizationConfig
        from model import ModelConfig
        
        # Test with mock config path
        with patch.object(TokenizationConfig, 'config_path', str(self.config_file)):
            tokenize_config = TokenizationConfig()
            self.assertEqual(tokenize_config.batch_size, 8)
        
        with patch.object(ModelConfig, 'config_path', str(self.config_file)):
            model_config = ModelConfig()
            self.assertEqual(model_config.learning_rate, 1e-5)
    
    def test_data_flow_integration(self):
        """Test data flow between components."""
        # This test would require more complex mocking
        # For now, just verify that components can be imported together
        try:
            from tokenize_pipeline import TokenizationPipeline
            from model import BertSentimentModel
            from route import analyze_sentiment
            integration_success = True
        except ImportError:
            integration_success = False
        
        self.assertTrue(integration_success)


def mock_open_multiple_files(files_dict):
    """Helper function to mock multiple file opens."""
    def mock_open_func(*args, **kwargs):
        filename = args[0] if args else kwargs.get('file')
        if filename in files_dict:
            from unittest.mock import mock_open
            return mock_open(read_data=files_dict[filename]).return_value
        raise FileNotFoundError(f"No mock data for {filename}")
    return mock_open_func


class TestRunner:
    """Custom test runner with detailed reporting."""
    
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
    
    def run_test_suite(self):
        """Run the complete test suite."""
        print("=" * 60)
        print("SENTIMENT MODEL TEST SUITE")
        print("=" * 60)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        test_classes = [
            TestTokenizePipeline,
            TestBertSentimentModel,
            TestSentimentRoute,
            TestIntegration
        ]
        
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run tests with custom result handler
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            buffer=True
        )
        
        print(f"\nRunning {suite.countTestCases()} tests...")
        print("-" * 60)
        
        result = runner.run(suite)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
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
        
        # Environment info
        print(f"\nENVIRONMENT:")
        print(f"  - PyTorch Available: {TORCH_AVAILABLE}")
        print(f"  - Python Version: {sys.version}")
        
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                       result.testsRun * 100) if result.testsRun > 0 else 0
        print(f"  - Success Rate: {success_rate:.1f}%")
        
        print("=" * 60)
        
        return result.wasSuccessful()


def main():
    """Main function to run tests."""
    print("Sentiment Model Test Suite")
    print("Testing tokenize_pipeline.py, model.py, and route.py")
    
    # Check dependencies
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("sklearn")
    
    if missing_deps:
        print(f"\nWarning: Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests may be skipped. Install with:")
        print("pip install torch transformers scikit-learn")
    
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