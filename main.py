#!/usr/bin/env python3
"""
Neural Trade Engine - Main Application

Example usage of the neural trade engine package combining sentiment analysis
and unified stock prediction models.

Usage:
    python main.py --help
    python main.py train --tickers AAPL MSFT --days 730
    python main.py predict --ticker AAPL --model models/latest_model.pth
    python main.py sentiment --text "Stock market looks bullish today"
    python main.py paper-trade --balance 50000
    python main.py simulate --tickers AAPL MSFT --strategy momentum --days 30
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Import engine functions
try:
    from engine import (
        train_model, 
        predict_price, 
        get_model_info, 
        list_available_models,
        analyze_sentiment,
        pull_from_web,
        create_trading_engine
    )
except ImportError as e:
    print(f"❌ Failed to import engine functions: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)



# Add current directory to path for engine imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color utilities for terminal output
class Colors:
    # Standard colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    @staticmethod
    def red(text): return f"{Colors.RED}{text}{Colors.RESET}"
    
    @staticmethod
    def green(text): return f"{Colors.GREEN}{text}{Colors.RESET}"
    
    @staticmethod
    def yellow(text): return f"{Colors.YELLOW}{text}{Colors.RESET}"
    
    @staticmethod
    def blue(text): return f"{Colors.BLUE}{text}{Colors.RESET}"
    
    @staticmethod
    def magenta(text): return f"{Colors.MAGENTA}{text}{Colors.RESET}"
    
    @staticmethod
    def cyan(text): return f"{Colors.CYAN}{text}{Colors.RESET}"
    
    @staticmethod
    def bold(text): return f"{Colors.BOLD}{text}{Colors.RESET}"
    
    @staticmethod
    def dim(text): return f"{Colors.DIM}{text}{Colors.RESET}"



def cmd_train(args):
    """Train a unified stock prediction model."""
    # Set logging levels to show only training progress
    import logging
    logging.getLogger('engine.unified_model.api').setLevel(logging.ERROR)
    logging.getLogger('engine.unified_model.data_pipelines').setLevel(logging.ERROR)
    logging.getLogger('engine.unified_model.train').setLevel(logging.INFO)
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    print(Colors.bold(Colors.cyan("Training Neural Trade Engine")))
    print("─" * 40)
    print(f"{Colors.blue('Tickers:')}     {', '.join(args.tickers)}")
    print(f"{Colors.blue('Data Range:')}  {start_date} to {end_date} ({args.days} days)")
    print(f"{Colors.blue('Model:')}      {args.model_type.title()}")
    print(f"{Colors.blue('Epochs:')}     {args.epochs}")
    print(f"{Colors.blue('Batch Size:')}  {args.batch_size}")
    print(f"{Colors.blue('Learning:')}    {args.learning_rate}")
    print()
    
    try:
        model_path = train_model(
            tickers=args.tickers,
            start_date=start_date,
            end_date=end_date,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print(f"\n{Colors.green('SUCCESS:')} Training completed successfully!")
        print(f"{Colors.magenta('Model saved to:')} {Path(model_path).name}")
        
        # Show model info
        try:
            info = get_model_info(model_path)
            print(f"{Colors.green('Final validation loss:')} {info['validation_loss']:.6f}")
        except:
            pass  # Skip if model info fails
        
    except Exception as e:
        print(f"\n{Colors.red('ERROR:')} Training failed: {e}")
        if "No data was successfully processed" in str(e):
            print(Colors.yellow("TIP: Try different tickers or increase the data range (--days)"))
        return 1
    
    return 0


def cmd_predict(args):
    """Make price predictions using a trained model with sentiment analysis."""
    # Set logging level to reduce noise
    import logging
    logging.getLogger('engine.unified_model').setLevel(logging.ERROR)
    logging.getLogger('engine').setLevel(logging.ERROR)
    
    # Find model file
    if args.model:
        model_path = args.model
        model_name = Path(model_path).name
    else:
        # Use latest model
        models = list_available_models()
        if not models:
            print(Colors.red("ERROR: No trained models found. Train a model first with:"))
            print(Colors.dim("   python main.py train --tickers AAPL MSFT --days 365"))
            return 1
        model_path = models[0]['file_path']
        model_name = Path(model_path).name
    
    if not os.path.exists(model_path):
        print(Colors.red(f"ERROR: Model file not found: {model_path}"))
        return 1
    
    print(f"{Colors.cyan('Predicting')} {Colors.bold(args.ticker.upper())} {Colors.cyan('price using')} {Colors.magenta(model_name)}...")
    
    # Pull sentiment data for the ticker
    print(f"{Colors.cyan('Gathering sentiment data for')} {Colors.bold(args.ticker.upper())}...")
    try:
        sentiment_result = pull_from_web(args.ticker)
        if sentiment_result.get('success') and sentiment_result.get('data_collected'):
            print(f"{Colors.green('✓')} Sentiment data collected successfully")
            
            # Analyze sentiment from the scraped data
            print(Colors.cyan("Analyzing sentiment..."))
            sentiment_analysis = analyze_sentiment()
            
            if sentiment_analysis.get('success'):
                stats = sentiment_analysis.get('statistics', {})
                overall_sentiment = stats.get('overall_sentiment', 'neutral')
                sentiment_confidence = stats.get('sentiment_confidence', 0)
                
                print(f"{Colors.green('✓')} Sentiment analysis complete")
                print(f"  {Colors.blue('Overall Sentiment:')} {overall_sentiment.title()}")
                print(f"  {Colors.blue('Confidence:')} {sentiment_confidence:.1%}")
            else:
                print(f"{Colors.yellow('⚠')} Sentiment analysis failed: {sentiment_analysis.get('message', 'Unknown error')}")
                overall_sentiment = 'neutral'
                sentiment_confidence = 0.5
        else:
            print(f"{Colors.yellow('⚠')} Could not gather sentiment data: {sentiment_result.get('message', 'Unknown error')}")
            overall_sentiment = 'neutral'
            sentiment_confidence = 0.5
            
    except Exception as e:
        print(f"{Colors.yellow('⚠')} Sentiment analysis error: {e}")
        overall_sentiment = 'neutral' 
        sentiment_confidence = 0.5
    
    # Make price prediction
    print(Colors.cyan("Making price prediction..."))
    try:
        result = predict_price(
            ticker=args.ticker,
            model_path=model_path,
            include_confidence=args.confidence
        )
        
        print(f"\n{Colors.bold(Colors.blue(result['ticker'] + ' Price Prediction'))}")
        print("─" * 50)
        print(f"{Colors.green('Predicted Price:')}  ${result['predicted_price']:.2f}")
        
        if args.confidence and 'confidence_interval' in result:
            ci_low, ci_high = result['confidence_interval']
            print(f"{Colors.yellow('95% Confidence:')}   ${ci_low:.2f} - ${ci_high:.2f}")
            print(f"{Colors.magenta('Uncertainty:')}      {result['uncertainty']:.1%}")
        
        print(f"{Colors.blue('Prediction Date:')}  {result['prediction_date']}")
        print()
        print(f"{Colors.cyan('Sentiment Analysis:')}")
        print(f"  {Colors.green('Overall Sentiment:')} {overall_sentiment.title()}")
        print(f"  {Colors.green('Confidence:')} {sentiment_confidence:.1%}")
        
        # Provide sentiment-based guidance
        if overall_sentiment == 'positive' and sentiment_confidence > 0.6:
            print(f"{Colors.green('📈 Sentiment suggests bullish market mood')}")
        elif overall_sentiment == 'negative' and sentiment_confidence > 0.6:
            print(f"{Colors.red('📉 Sentiment suggests bearish market mood')}")
        else:
            print(f"{Colors.yellow('📊 Mixed or neutral sentiment detected')}")
        
    except Exception as e:
        print(Colors.red(f"ERROR: Prediction failed: {e}"))
        if "No data available" in str(e) or "No data was successfully processed" in str(e):
            print(Colors.yellow("TIP: Please check that the ticker symbol is valid (e.g. AAPL, MSFT, TSLA)"))
        return 1
    
    return 0


def cmd_sentiment(args):
    """Analyze sentiment of text."""
    # Set logging level to reduce noise
    import logging
    logging.getLogger('engine').setLevel(logging.ERROR)
    
    try:
        if args.url:
            print(f"{Colors.cyan('Analyzing sentiment from:')} {args.url}")
            content = pull_from_web(args.url)
            text_to_analyze = content.get('content', args.url) if isinstance(content, dict) else str(content)
        else:
            print(Colors.cyan("Analyzing text sentiment..."))
            text_to_analyze = args.text
        
        result = analyze_sentiment(text_to_analyze)
        
        print(f"\n{Colors.bold(Colors.blue('Sentiment Analysis'))}")
        print("─" * 40)
        print(f"{Colors.blue('Text Preview:')} {text_to_analyze[:80]}{'...' if len(text_to_analyze) > 80 else ''}")
        
        if isinstance(result, dict):
            sentiment = result.get('sentiment', 'Unknown').title()
            confidence = result.get('confidence', 0)
            print(f"{Colors.green('Sentiment:')}   {sentiment}")
            print(f"{Colors.yellow('Confidence:')}  {confidence:.1%}")
        else:
            print(f"{Colors.green('Result:')}      {result}")
        
    except Exception as e:
        print(Colors.red(f"ERROR: Sentiment analysis failed: {e}"))
        return 1
    
    return 0


def cmd_list_models(args):
    """List available trained models."""
    models = list_available_models()
    
    if not models:
        print(Colors.yellow("No trained models found."))
        print("Train a model first with:")
        print(Colors.dim("   python main.py train --tickers AAPL MSFT --days 365"))
        return 0
    
    print(f"\n{Colors.bold(Colors.blue('Available Models'))}")
    print("─" * 40)
    
    for i, model in enumerate(models, 1):
        print(f"{Colors.green(f'{i}.')} {Colors.bold(Path(model['file_path']).name)}")
        print(f"   {Colors.blue('Path:')} {model['file_path']}")
        if 'size' in model:
            print(f"   {Colors.blue('Size:')} {model['size']}")
        if 'modified' in model:
            print(f"   {Colors.blue('Modified:')} {model['modified']}")
        print()
    
    return 0


def cmd_info(args):
    """Show model information."""
    if not os.path.exists(args.model):
        print(Colors.red(f"ERROR: Model file not found: {args.model}"))
        return 1
    
    try:
        info = get_model_info(args.model)
        model_name = Path(args.model).name
        
        print(f"\n{Colors.bold(Colors.blue(f'Model Information: {model_name}'))}")
        print("─" * 50)
        
        for key, value in info.items():
            if key == 'file_path':
                continue
            
            key_display = key.replace('_', ' ').title()
            
            if isinstance(value, float):
                if key in ['validation_loss', 'training_loss']:
                    print(f"{Colors.green(key_display + ':')} {value:.6f}")
                else:
                    print(f"{Colors.green(key_display + ':')} {value:.4f}")
            elif isinstance(value, int):
                print(f"{Colors.green(key_display + ':')} {value}")
            else:
                print(f"{Colors.green(key_display + ':')} {value}")
        
        print(f"\n{Colors.blue('File Path:')} {args.model}")
        
    except Exception as e:
        print(Colors.red(f"ERROR: Could not read model info: {e}"))
        return 1
    
    return 0


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Neural Trade Engine - AI-powered stock prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a prediction model')
    train_parser.add_argument('--tickers', nargs='+', default=['AAPL'], 
                             help='Stock tickers to train on')
    train_parser.add_argument('--days', type=int, default=730,
                             help='Days of historical data (default: 730)')
    train_parser.add_argument('--model-type', choices=['standard', 'adaptive'], 
                             default='standard', help='Model architecture type')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-3,
                             help='Learning rate')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict stock price')
    predict_parser.add_argument('--ticker', required=True,
                               help='Stock ticker to predict')
    predict_parser.add_argument('--model', 
                               help='Model file path (uses latest if not specified)')
    predict_parser.add_argument('--confidence', action='store_true',
                               help='Include confidence intervals')
    
    # Sentiment command
    sentiment_parser = subparsers.add_parser('sentiment', help='Analyze text sentiment')
    sentiment_group = sentiment_parser.add_mutually_exclusive_group(required=True)
    sentiment_group.add_argument('--text', help='Text to analyze')
    sentiment_group.add_argument('--url', help='URL to extract and analyze')
    
    # List models command
    list_parser = subparsers.add_parser('models', help='List available trained models')
    
    # Model info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', required=True, help='Model file path')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper-trade', help='Start paper trading session')
    paper_parser.add_argument('--balance', type=float, default=100000,
                             help='Starting balance (default: $100,000)')
    paper_parser.add_argument('--save', help='Save portfolio state to file')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run automated trading simulation')
    sim_parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT'], 
                           help='Stock tickers to trade')
    sim_parser.add_argument('--balance', type=float, default=1000,
                           help='Starting balance (default: $1,000)')
    sim_parser.add_argument('--days', type=int, default=30,
                           help='Simulation duration in days')
    sim_parser.add_argument('--strategy', choices=['buy_hold', 'momentum', 'ml_prediction'],
                           default='buy_hold', help='Trading strategy to use')
    sim_parser.add_argument('--model', help='Model path for ML strategy')
    sim_parser.add_argument('--frequency', type=float, default=6,
                           help='Update frequency in hours (default: 6)')
    sim_parser.add_argument('--realtime', action='store_true',
                           help='Run in real-time (slower but more realistic)')
    sim_parser.add_argument('--save', help='Save simulation results to JSON file')
    sim_parser.add_argument('--export-csv', help='Export trade history to CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Check for API key
    if args.command in ['train', 'predict', 'paper-trade', 'simulate'] and not os.getenv('FMP_API_KEY'):
        print(Colors.red("ERROR: FMP_API_KEY environment variable not set!"))
        print("Please set your Financial Modeling Prep API key:")
        print(Colors.dim("export FMP_API_KEY=your_api_key_here"))
        return 1
    
    # Execute command
    if args.command == 'train':
        return cmd_train(args)
    elif args.command == 'predict':
        return cmd_predict(args)
    elif args.command == 'sentiment':
        return cmd_sentiment(args)
    elif args.command == 'models':
        return cmd_list_models(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        print(Colors.red(f"ERROR: Unknown command: {args.command}"))
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(Colors.yellow("\nInterrupted by user"))
        sys.exit(1)
    except Exception as e:
        print(Colors.red(f"ERROR: Unexpected error: {e}"))
        sys.exit(1)