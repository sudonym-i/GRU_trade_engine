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

# Add current directory to path for engine imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def cmd_train(args):
    """Train a unified stock prediction model."""
    print(f"🚀 Training model on {len(args.tickers)} tickers...")
    print(f"📊 Tickers: {', '.join(args.tickers)}")
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    print(f"📅 Training data: {start_date} to {end_date}")
    print(f"⚙️  Model type: {args.model_type}")
    print(f"🏋️  Epochs: {args.epochs}")
    
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
        
        print(f"✅ Training completed successfully!")
        print(f"💾 Model saved to: {model_path}")
        
        # Show model info
        info = get_model_info(model_path)
        print(f"📈 Final validation loss: {info['validation_loss']:.6f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


def cmd_predict(args):
    """Make price predictions using a trained model."""
    print(f"🔮 Predicting price for {args.ticker}...")
    
    # Find model file
    if args.model:
        model_path = args.model
    else:
        # Use latest model
        models = list_available_models()
        if not models:
            print("❌ No trained models found. Train a model first.")
            return 1
        model_path = models[0]['file_path']
        print(f"📁 Using latest model: {Path(model_path).name}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return 1
    
    try:
        result = predict_price(
            ticker=args.ticker,
            model_path=model_path,
            include_confidence=args.confidence
        )
        
        print(f"✅ Prediction for {result['ticker']}:")
        print(f"💰 Predicted price: ${result['predicted_price']:.2f}")
        
        if args.confidence and 'confidence_interval' in result:
            ci_low, ci_high = result['confidence_interval']
            print(f"📊 95% confidence interval: ${ci_low:.2f} - ${ci_high:.2f}")
            print(f"🎯 Uncertainty: {result['uncertainty']:.2%}")
        
        print(f"📅 Prediction date: {result['prediction_date']}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return 1
    
    return 0


def cmd_sentiment(args):
    """Analyze sentiment of text."""
    print(f"🧠 Analyzing sentiment...")
    
    try:
        if args.url:
            print(f"🌐 Extracting content from: {args.url}")
            content = pull_from_web(args.url)
            text_to_analyze = content.get('content', args.url) if isinstance(content, dict) else str(content)
        else:
            text_to_analyze = args.text
        
        result = analyze_sentiment(text_to_analyze)
        
        print(f"✅ Sentiment analysis result:")
        print(f"📝 Text: {text_to_analyze[:100]}...")
        if isinstance(result, dict):
            print(f"😊 Sentiment: {result.get('sentiment', 'Unknown')}")
            print(f"🎯 Confidence: {result.get('confidence', 0):.2%}")
        else:
            print(f"😊 Sentiment result: {result}")
        
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        return 1
    
    return 0


def cmd_list_models(args):
    """List all available trained models."""
    print("📋 Available trained models:")
    
    try:
        models = list_available_models()
        
        if not models:
            print("❌ No trained models found.")
            return 0
        
        for i, model in enumerate(models, 1):
            print(f"\n{i}. {model['file_name']}")
            print(f"   📊 Type: {model['model_class']}")
            print(f"   📅 Created: {model['timestamp']}")
            print(f"   📈 Validation loss: {model['validation_loss']}")
            print(f"   💾 Size: {model['file_size_mb']:.1f} MB")
            
            if 'training_history' in model:
                history = model['training_history']
                print(f"   🏋️  Epochs: {history['epochs_trained']}")
                print(f"   🎯 Best loss: {history['best_val_loss']:.6f}")
        
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return 1
    
    return 0


def cmd_info(args):
    """Show information about a specific model."""
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    try:
        info = get_model_info(args.model)
        
        print(f"📊 Model Information: {Path(args.model).name}")
        print(f"   📋 Type: {info['model_class']}")
        print(f"   📅 Created: {info['timestamp']}")
        print(f"   🏋️  Training epoch: {info['training_epoch']}")
        print(f"   📈 Validation loss: {info['validation_loss']}")
        print(f"   💾 File size: {info['file_size_mb']:.1f} MB")
        
        if 'training_history' in info:
            history = info['training_history']
            print(f"\n📈 Training History:")
            print(f"   🏁 Epochs completed: {history['epochs_trained']}")
            print(f"   📊 Final train loss: {history['final_train_loss']:.6f}")
            print(f"   📊 Final val loss: {history['final_val_loss']:.6f}")
            print(f"   🎯 Best val loss: {history['best_val_loss']:.6f}")
        
        print(f"\n⚙️  Model Configuration:")
        config = info['model_config']
        for key, value in config.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Failed to get model info: {e}")
        return 1
    
    return 0


def cmd_paper_trade(args):
    """Start paper trading session."""
    print(f"📊 Starting paper trading with ${args.balance:,.2f} balance...")
    
    try:
        # Create trading engine
        engine = create_trading_engine(initial_balance=args.balance)
        
        print(f"✅ Paper trading engine created")
        print(f"💰 Initial balance: ${engine.get_portfolio_summary()['total_equity']:,.2f}")
        print(f"📈 Market is {'open' if engine.is_market_open() else 'closed'}")
        
        # Interactive trading session
        print("\n📋 Available commands:")
        print("  buy <symbol> <quantity> [price]   - Place buy order")
        print("  sell <symbol> <quantity> [price]  - Place sell order") 
        print("  portfolio                        - Show portfolio")
        print("  positions                        - Show positions")
        print("  orders                           - Show orders")
        print("  update                           - Update market data")
        print("  quit                             - Exit")
        
        while True:
            try:
                cmd = input("\n💼 > ").strip().lower()
                
                if cmd == 'quit':
                    break
                elif cmd == 'portfolio':
                    summary = engine.get_portfolio_summary()
                    print(f"💰 Total Equity: ${summary['total_equity']:,.2f}")
                    print(f"💵 Cash: ${summary['cash_balance']:,.2f}")
                    print(f"📊 P&L: ${summary['total_return']:,.2f} ({summary['total_return_percent']:.2f}%)")
                
                elif cmd == 'positions':
                    positions = engine.get_positions()
                    if positions:
                        print(f"📈 Positions ({len(positions)}):")
                        for pos in positions:
                            print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['average_cost']:.2f} "
                                  f"(${pos['current_price']:.2f}, P&L: ${pos['unrealized_pnl']:.2f})")
                    else:
                        print("📭 No positions")
                
                elif cmd == 'orders':
                    orders = engine.get_orders('active')
                    if orders:
                        print(f"📋 Active Orders ({len(orders)}):")
                        for order in orders:
                            print(f"  {order['order_id']}: {order['side'].upper()} {order['quantity']} "
                                  f"{order['symbol']} @ {order['order_type'].upper()}")
                    else:
                        print("📭 No active orders")
                
                elif cmd == 'update':
                    print("🔄 Updating market data...")
                    engine.update_market_data()
                    print("✅ Market data updated")
                
                elif cmd.startswith('buy ') or cmd.startswith('sell '):
                    parts = cmd.split()
                    if len(parts) < 3:
                        print("❌ Usage: buy/sell <symbol> <quantity> [price]")
                        continue
                    
                    action = parts[0]
                    symbol = parts[1].upper()
                    quantity = int(parts[2])
                    
                    if len(parts) > 3:
                        # Limit order
                        price = float(parts[3])
                        if action == 'buy':
                            order_id = engine.buy(symbol, quantity, "limit", limit_price=price)
                        else:
                            order_id = engine.sell(symbol, quantity, "limit", limit_price=price)
                        print(f"✅ Placed limit {action} order: {order_id}")
                    else:
                        # Market order
                        if action == 'buy':
                            order_id = engine.buy(symbol, quantity, "market")
                        else:
                            order_id = engine.sell(symbol, quantity, "market")
                        print(f"✅ Placed market {action} order: {order_id}")
                
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Show final summary
        final_summary = engine.get_portfolio_summary()
        print(f"\n📊 Final Summary:")
        print(f"💰 Final Equity: ${final_summary['total_equity']:,.2f}")
        print(f"📈 Total Return: ${final_summary['total_return']:,.2f} ({final_summary['total_return_percent']:.2f}%)")
        
        # Save portfolio state
        if args.save:
            engine.save_state(args.save)
            print(f"💾 Portfolio saved to {args.save}")
        
    except Exception as e:
        print(f"❌ Paper trading failed: {e}")
        return 1
    
    return 0


def cmd_simulate(args):
    """Run automated trading simulation."""
    print(f"🤖 Starting trading simulation...")
    print(f"💰 Initial balance: ${args.balance:,.2f}")
    print(f"📊 Tickers: {', '.join(args.tickers)}")
    print(f"⏱️  Duration: {args.days} days")
    print(f"🎯 Strategy: {args.strategy}")
    
    try:
        # Create trading engine
        engine = create_trading_engine(initial_balance=args.balance)
        
        # Import trading strategies
        from engine.trading_simulation import BuyAndHoldStrategy, MomentumStrategy, StrategyManager
        
        # Set up strategy
        if args.strategy == "buy_hold":
            # Equal allocation across all tickers
            allocation = {ticker: 1.0/len(args.tickers) for ticker in args.tickers}
            strategy = BuyAndHoldStrategy(allocation, rebalance_threshold=0.05)
        elif args.strategy == "momentum":
            strategy = MomentumStrategy(
                lookback_days=20,
                momentum_threshold=0.05,
                stop_loss_pct=0.10,
                take_profit_pct=0.20
            )
        elif args.strategy == "ml_prediction":
            # ML strategy using trained model
            if not args.model:
                print("❌ ML strategy requires --model parameter")
                return 1
            
            def ml_predictor(symbol):
                try:
                    return predict_price(symbol, args.model, include_confidence=True)
                except Exception as e:
                    print(f"⚠️ Prediction failed for {symbol}: {e}")
                    return None
            
            from engine.trading_simulation.strategies import MLPredictionStrategy
            strategy = MLPredictionStrategy(
                model_predictor=ml_predictor,
                confidence_threshold=0.6,
                position_size_pct=0.1
            )
        else:
            print(f"❌ Unknown strategy: {args.strategy}")
            return 1
        
        strategy.activate()
        
        # Simulation parameters
        simulation_days = args.days
        update_frequency = args.frequency  # hours
        current_day = 0
        
        print(f"\n🚀 Starting simulation...")
        print(f"📈 Market status: {'Open' if engine.is_market_open() else 'Closed'}")
        
        # Initial market data update
        engine.update_market_data()
        
        # Track performance over time
        daily_performance = []
        
        # Simulation loop
        import time
        from datetime import datetime, timedelta
        
        while current_day < simulation_days:
            try:
                # Update market data
                engine.update_market_data()
                
                # Get current portfolio state
                portfolio_summary = engine.get_portfolio_summary()
                positions = {pos['symbol']: pos for pos in engine.get_positions()}
                
                # Get market data for strategy
                market_data = {}
                for ticker in args.tickers:
                    try:
                        current_price = engine.market_data.get_current_price(ticker)
                        if current_price:
                            market_data[ticker] = {
                                'current_price': current_price,
                                'symbol': ticker
                            }
                    except Exception as e:
                        print(f"⚠️ Failed to get price for {ticker}: {e}")
                        continue
                
                # Generate signals for each ticker
                signals_executed = 0
                for ticker in args.tickers:
                    if ticker not in market_data:
                        continue
                    
                    # Get strategy signal
                    signal = strategy.generate_signal(
                        symbol=ticker,
                        market_data=market_data[ticker],
                        portfolio_data={
                            **portfolio_summary,
                            'positions': positions
                        }
                    )
                    
                    # Execute signal
                    if signal['action'] == 'buy' and signal.get('quantity', 0) > 0:
                        try:
                            order_id = engine.buy(
                                ticker, 
                                signal['quantity'], 
                                signal.get('order_type', 'market'),
                                limit_price=signal.get('limit_price')
                            )
                            print(f"📈 BUY: {signal['quantity']} {ticker} - {signal.get('reason', '')}")
                            signals_executed += 1
                        except Exception as e:
                            print(f"❌ Buy order failed for {ticker}: {e}")
                    
                    elif signal['action'] == 'sell' and signal.get('quantity', 0) > 0:
                        try:
                            order_id = engine.sell(
                                ticker,
                                signal['quantity'],
                                signal.get('order_type', 'market'),
                                limit_price=signal.get('limit_price')
                            )
                            print(f"📉 SELL: {signal['quantity']} {ticker} - {signal.get('reason', '')}")
                            signals_executed += 1
                        except Exception as e:
                            print(f"❌ Sell order failed for {ticker}: {e}")
                
                # Record daily performance
                current_performance = {
                    'day': current_day,
                    'total_equity': portfolio_summary['total_equity'],
                    'total_return': portfolio_summary['total_return'],
                    'total_return_percent': portfolio_summary['total_return_percent'],
                    'cash_balance': portfolio_summary['cash_balance'],
                    'positions_count': len(engine.get_positions()),
                    'signals_executed': signals_executed
                }
                daily_performance.append(current_performance)
                
                # Progress update
                if current_day % max(1, simulation_days // 10) == 0 or signals_executed > 0:
                    print(f"📅 Day {current_day}: Equity ${portfolio_summary['total_equity']:,.2f} "
                          f"({portfolio_summary['total_return_percent']:+.2f}%) - "
                          f"{signals_executed} trades")
                
                current_day += 1
                
                # Sleep between updates (for demo purposes, adjust as needed)
                if args.realtime:
                    time.sleep(update_frequency * 3600)  # Convert hours to seconds
                else:
                    # Fast simulation mode
                    time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n⏹️  Simulation interrupted by user")
                break
            except Exception as e:
                print(f"❌ Simulation error on day {current_day}: {e}")
                continue
        
        # Final results
        final_summary = engine.get_portfolio_summary()
        performance_metrics = engine.get_performance_metrics()
        
        print(f"\n🏁 Simulation Complete!")
        print(f"=" * 50)
        print(f"📊 Final Results:")
        print(f"💰 Final Equity: ${final_summary['total_equity']:,.2f}")
        print(f"📈 Total Return: ${final_summary['total_return']:,.2f} ({final_summary['total_return_percent']:+.2f}%)")
        print(f"💵 Cash Remaining: ${final_summary['cash_balance']:,.2f}")
        print(f"📋 Total Trades: {performance_metrics['total_trades']}")
        print(f"🎯 Win Rate: {performance_metrics['win_rate']:.1f}%")
        print(f"💡 Average Win: ${performance_metrics['average_win']:.2f}")
        print(f"💸 Average Loss: ${performance_metrics['average_loss']:.2f}")
        print(f"💳 Total Fees: ${performance_metrics['total_fees']:.2f}")
        
        # Show final positions
        final_positions = engine.get_positions()
        if final_positions:
            print(f"\n📈 Final Positions:")
            for pos in final_positions:
                print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['average_cost']:.2f} "
                      f"(Current: ${pos['current_price']:.2f}, "
                      f"P&L: ${pos['unrealized_pnl']:+.2f})")
        
        # Save results
        if args.save:
            # Save detailed results
            results = {
                'simulation_parameters': {
                    'tickers': args.tickers,
                    'strategy': args.strategy,
                    'initial_balance': args.balance,
                    'simulation_days': args.days,
                    'update_frequency': args.frequency
                },
                'final_summary': final_summary,
                'performance_metrics': performance_metrics,
                'daily_performance': daily_performance,
                'final_positions': final_positions,
                'trade_history': engine.get_trade_history()
            }
            
            import json
            with open(args.save, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {args.save}")
        
        # Export trade history
        if args.export_csv:
            engine.export_trades(args.export_csv)
            print(f"📊 Trade history exported to {args.export_csv}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


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
    sim_parser.add_argument('--balance', type=float, default=100000,
                           help='Starting balance (default: $100,000)')
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
        print("❌ FMP_API_KEY environment variable not set!")
        print("Please set your Financial Modeling Prep API key:")
        print("export FMP_API_KEY=your_api_key_here")
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
    elif args.command == 'paper-trade':
        return cmd_paper_trade(args)
    elif args.command == 'simulate':
        return cmd_simulate(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)