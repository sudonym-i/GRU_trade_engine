#!/usr/bin/env python3
"""
Test run script for the Unified Stock Prediction Model

This script performs a complete test of the unified model pipeline:
1. Creates a small dataset with sample tickers
2. Trains both standard and adaptive models
3. Evaluates performance and generates predictions
4. Saves results and model artifacts

Usage:
    python test_run.py
    python test_run.py --quick  # For faster testing
    python test_run.py --tickers AAPL MSFT --days 365
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports - no need for path manipulation since everything is in unified_model

from engine.unified_model.data_pipelines.integrated_data_pipeline import UnifiedDataPipeline
from integrated_model import UnifiedStockPredictor, AdaptiveUnifiedPredictor
from train import UnifiedTrainer, train_unified_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_run.log')
    ]
)
logger = logging.getLogger(__name__)


class UnifiedModelTester:
    """Comprehensive tester for the unified stock prediction model."""
    
    def __init__(self, output_dir: str = "test_results"):
        """Initialize tester with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results storage
        self.results = {
            'test_info': {},
            'data_info': {},
            'model_performance': {},
            'predictions': {}
        }
    
    def test_data_pipeline(self, tickers: list, start_date: str, end_date: str) -> bool:
        """
        Test the unified data pipeline.
        
        Returns:
            bool: True if pipeline test passes
        """
        logger.info("=== Testing Data Pipeline ===")
        
        try:
            # Initialize pipeline
            pipeline = UnifiedDataPipeline()
            
            # Test feature info for each ticker
            for ticker in tickers:
                logger.info(f"Testing feature info for {ticker}...")
                feature_info = pipeline.get_feature_info(ticker, start_date, end_date)
                
                logger.info(f"  Price features: {feature_info['price_features']}")
                logger.info(f"  Financial features: {len(feature_info['financial_features'])}")
                logger.info(f"  Total features: {feature_info['total_features']}")
                
                self.results['data_info'][ticker] = feature_info
            
            # Create unified dataset
            logger.info("Creating unified dataset...")
            dataset = pipeline.create_unified_dataset(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                seq_length=30,  # Shorter for testing
                normalize=True
            )
            
            logger.info(f"Dataset created successfully: {len(dataset)} samples")
            
            # Test data sample
            X_sample, y_sample = dataset[0]
            logger.info(f"Sample shapes: X={X_sample.shape}, y={y_sample.shape}")
            
            # Store dataset info
            self.results['data_info']['dataset'] = {
                'total_samples': len(dataset),
                'sequence_length': X_sample.shape[0],
                'features_per_timestep': X_sample.shape[1],
                'target_shape': y_sample.shape
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Data pipeline test failed: {e}")
            return False
    
    def test_model_architecture(self) -> bool:
        """Test model architectures."""
        logger.info("=== Testing Model Architecture ===")
        
        try:
            # Test parameters
            price_features = 4
            financial_features = 13
            seq_length = 30
            batch_size = 8
            
            # Test standard model
            logger.info("Testing UnifiedStockPredictor...")
            standard_model = UnifiedStockPredictor(
                price_features=price_features,
                financial_features=financial_features,
                hidden_dim=64,  # Smaller for testing
                num_layers=2,
                use_attention=True
            )
            
            # Test forward pass
            x = torch.randn(batch_size, seq_length, price_features + financial_features)
            
            with torch.no_grad():
                output = standard_model(x)
                logger.info(f"Standard model output shape: {output.shape}")
                
                # Test confidence prediction
                confidence = standard_model.predict_with_confidence(x[:1], num_samples=3)
                logger.info(f"Confidence prediction keys: {list(confidence.keys())}")
            
            # Test adaptive model
            logger.info("Testing AdaptiveUnifiedPredictor...")
            adaptive_model = AdaptiveUnifiedPredictor(
                price_features=price_features,
                financial_features=financial_features,
                hidden_dim=64,
                num_layers=2
            )
            
            with torch.no_grad():
                adaptive_output = adaptive_model(x)
                logger.info(f"Adaptive model output shape: {adaptive_output.shape}")
            
            # Store model info
            self.results['model_performance']['architectures'] = {
                'standard_params': sum(p.numel() for p in standard_model.parameters()),
                'adaptive_params': sum(p.numel() for p in adaptive_model.parameters()),
                'test_input_shape': list(x.shape),
                'test_output_shape': list(output.shape)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Model architecture test failed: {e}")
            return False
    
    def test_training_pipeline(self, tickers: list, start_date: str, end_date: str,
                             quick_mode: bool = False) -> bool:
        """Test the training pipeline."""
        logger.info("=== Testing Training Pipeline ===")
        
        try:
            # Training parameters (reduced for testing)
            training_params = {
                'epochs': 3 if quick_mode else 10,
                'batch_size': 16,
                'learning_rate': 1e-3,
                'seq_length': 30,
                'hidden_dim': 64,
                'num_layers': 2,
                'use_attention': True,
                'plot_loss': False
            }
            
            logger.info(f"Training parameters: {training_params}")
            
            # Train standard model
            logger.info("Training standard model...")
            standard_trainer = train_unified_model(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                model_type="standard",
                **training_params
            )
            
            # Store training results
            std_history = standard_trainer.training_history
            self.results['model_performance']['standard'] = {
                'final_train_loss': std_history['train_loss'][-1],
                'final_val_loss': std_history['val_loss'][-1],
                'epochs_trained': len(std_history['train_loss']),
                'min_val_loss': min(std_history['val_loss'])
            }
            
            logger.info("Standard model training completed")
            logger.info(f"  Final train loss: {std_history['train_loss'][-1]:.6f}")
            logger.info(f"  Final val loss: {std_history['val_loss'][-1]:.6f}")
            
            if not quick_mode:
                # Train adaptive model
                logger.info("Training adaptive model...")
                adaptive_trainer = train_unified_model(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    model_type="adaptive",
                    **training_params
                )
                
                # Store adaptive results
                adp_history = adaptive_trainer.training_history
                self.results['model_performance']['adaptive'] = {
                    'final_train_loss': adp_history['train_loss'][-1],
                    'final_val_loss': adp_history['val_loss'][-1],
                    'epochs_trained': len(adp_history['train_loss']),
                    'min_val_loss': min(adp_history['val_loss'])
                }
                
                logger.info("Adaptive model training completed")
                logger.info(f"  Final train loss: {adp_history['train_loss'][-1]:.6f}")
                logger.info(f"  Final val loss: {adp_history['val_loss'][-1]:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline test failed: {e}")
            return False
    
    def test_prediction_inference(self, tickers: list, start_date: str, end_date: str) -> bool:
        """Test model prediction and inference."""
        logger.info("=== Testing Prediction Inference ===")
        
        try:
            # Create small dataset for inference testing
            pipeline = UnifiedDataPipeline()
            dataset = pipeline.create_unified_dataset(
                tickers=tickers[:1],  # Just first ticker
                start_date=start_date,
                end_date=end_date,
                seq_length=30,
                normalize=True
            )
            
            # Get sample data
            X_sample, y_actual = dataset[0]
            X_batch = X_sample.unsqueeze(0)  # Add batch dimension
            
            # Create and test model
            model = UnifiedStockPredictor(
                price_features=4,
                financial_features=X_sample.shape[1] - 4,
                hidden_dim=64,
                num_layers=2
            )
            
            # Test inference
            model.eval()
            with torch.no_grad():
                # Standard prediction
                prediction = model(X_batch)
                
                # Confidence prediction
                confidence_result = model.predict_with_confidence(X_batch, num_samples=5)
            
            # Store prediction results
            self.results['predictions']['sample_test'] = {
                'actual_target': float(y_actual.item()),
                'predicted_value': float(prediction.item()),
                'prediction_error': float(abs(prediction.item() - y_actual.item())),
                'confidence_std': float(confidence_result['std'].item()),
                'confidence_interval': [
                    float(confidence_result['ci_lower'].item()),
                    float(confidence_result['ci_upper'].item())
                ]
            }
            
            logger.info("Prediction inference test completed")
            logger.info(f"  Actual: {y_actual.item():.4f}")
            logger.info(f"  Predicted: {prediction.item():.4f}")
            logger.info(f"  Error: {abs(prediction.item() - y_actual.item()):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Prediction inference test failed: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("=== Generating Test Report ===")
        
        # Add test metadata
        self.results['test_info'] = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Save results to JSON
        report_path = self.output_dir / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = self.output_dir / 'test_summary.md'
        with open(summary_path, 'w') as f:
            f.write("# Unified Model Test Report\n\n")
            f.write(f"**Test Date:** {self.results['test_info']['timestamp']}\n\n")
            
            # Data info
            f.write("## Data Pipeline Results\n")
            if 'dataset' in self.results['data_info']:
                dataset_info = self.results['data_info']['dataset']
                f.write(f"- Total samples: {dataset_info['total_samples']}\n")
                f.write(f"- Sequence length: {dataset_info['sequence_length']}\n")
                f.write(f"- Features per timestep: {dataset_info['features_per_timestep']}\n\n")
            
            # Model performance
            f.write("## Model Performance\n")
            if 'standard' in self.results['model_performance']:
                std_perf = self.results['model_performance']['standard']
                f.write(f"**Standard Model:**\n")
                f.write(f"- Final validation loss: {std_perf['final_val_loss']:.6f}\n")
                f.write(f"- Epochs trained: {std_perf['epochs_trained']}\n")
                f.write(f"- Best validation loss: {std_perf['min_val_loss']:.6f}\n\n")
            
            if 'adaptive' in self.results['model_performance']:
                adp_perf = self.results['model_performance']['adaptive']
                f.write(f"**Adaptive Model:**\n")
                f.write(f"- Final validation loss: {adp_perf['final_val_loss']:.6f}\n")
                f.write(f"- Epochs trained: {adp_perf['epochs_trained']}\n")
                f.write(f"- Best validation loss: {adp_perf['min_val_loss']:.6f}\n\n")
            
            # Prediction test
            f.write("## Prediction Test\n")
            if 'sample_test' in self.results['predictions']:
                pred_test = self.results['predictions']['sample_test']
                f.write(f"- Prediction error: {pred_test['prediction_error']:.4f}\n")
                f.write(f"- Confidence interval: [{pred_test['confidence_interval'][0]:.4f}, {pred_test['confidence_interval'][1]:.4f}]\n")
        
        logger.info(f"Test report saved to {report_path}")
        logger.info(f"Test summary saved to {summary_path}")
    
    def run_full_test(self, tickers: list, start_date: str, end_date: str, 
                     quick_mode: bool = False) -> bool:
        """Run the complete test suite."""
        logger.info("=== Starting Full Test Suite ===")
        logger.info(f"Tickers: {tickers}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Quick mode: {quick_mode}")
        
        test_results = []
        
        # Test 1: Data Pipeline
        test_results.append(self.test_data_pipeline(tickers, start_date, end_date))
        
        # Test 2: Model Architecture  
        test_results.append(self.test_model_architecture())
        
        # Test 3: Training Pipeline
        test_results.append(self.test_training_pipeline(tickers, start_date, end_date, quick_mode))
        
        # Test 4: Prediction Inference
        test_results.append(self.test_prediction_inference(tickers, start_date, end_date))
        
        # Generate report
        self.generate_test_report()
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        logger.info(f"=== Test Suite Complete ===")
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed!")
            return True
        else:
            logger.warning("❌ Some tests failed. Check logs for details.")
            return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Test the Unified Stock Prediction Model")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT'], 
                       help='Stock tickers to test with')
    parser.add_argument('--days', type=int, default=365*2,
                       help='Number of days of historical data to use')
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode (fewer epochs, faster testing)')
    parser.add_argument('--output-dir', default='test_results',
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    # Check for API key
    if not os.getenv('FMP_API_KEY'):
        logger.error("FMP_API_KEY environment variable not set!")
        logger.error("Please set your Financial Modeling Prep API key:")
        logger.error("export FMP_API_KEY=your_api_key_here")
        return False
    
    # Run tests
    tester = UnifiedModelTester(output_dir=args.output_dir)
    
    success = tester.run_full_test(
        tickers=args.tickers,
        start_date=start_date,
        end_date=end_date,
        quick_mode=args.quick
    )
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with unexpected error: {e}")
        sys.exit(1)