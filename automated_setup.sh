
echo "Neural_Trade_Engine"
echo "===================="
echo ""
echo ""
echo "Setting up the environment..."


cd backend_\&_algorithms

chmod +x install.sh
./install.sh

cd engine/sentiment_model

python3 download_dataset.py

echo ""
echo "Training sentiment model..."
echo ""

python3 train_with_labeled_data.py

cd ../../


echo ""
echo "===================================="
echo ""
read -p "What ticker do you want to algorithmically trade?: " ticker
read -p "Enter semantic name (EX: GOOG would be -> google): " semantic_name

python3 main.py webscrape --ticker "$semantic_name"

python3 main.py train --ticker "$ticker"

cd ../integrations_\&_strategy
nano config.json

read -p "Start automated trading now? (y/n): " start
if [ "$start" == "y" || "$start" == "Y" ]; then
    echo ""
    echo "Select Trading Mode:"
    echo "1) Simulation Mode (paper trading with virtual portfolio - recommended for testing)"
    echo "2) Interactive Brokers Paper Trading (requires IB Gateway/TWS running)"
    echo "3) Interactive Brokers Live Trading (real money - use with extreme caution!)"
    echo ""
    read -p "Enter your choice (1-3): " mode_choice
    
    case $mode_choice in
        1)
            echo "Starting automated trading in SIMULATION mode..."
            echo "Portfolio will use virtual $10,000 starting capital."
            python automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            ;;
        2)
            echo "Starting automated trading in IB PAPER TRADING mode..."
            echo "Testing IB connection first..."
            python test_ib_connection.py --mode paper
            if [ $? -eq 0 ]; then
                python automated_trader.py --mode ib_paper --stock "$ticker" --semantic-name "$semantic_name"
            else
                echo "IB connection test failed. Please check:"
                echo "1. IB Gateway or TWS is running"
                echo "2. API is enabled in IB settings"
                echo "3. Paper trading port 7496 is configured"
                echo "Falling back to simulation mode..."
                python automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            fi
            ;;
        3)
            echo "⚠️  WARNING: You selected LIVE TRADING mode with real money!"
            echo "This will execute actual trades in your Interactive Brokers account."
            read -p "Are you absolutely sure you want to continue? (yes/no): " confirm
            if [ "$confirm" == "yes" ]; then
                echo "Testing IB live connection first..."
                python test_ib_connection.py --mode live
                if [ $? -eq 0 ]; then
                    echo "Starting LIVE TRADING - Good luck!"
                    python automated_trader.py --mode ib_live --stock "$ticker" --semantic-name "$semantic_name"
                else
                    echo "IB live connection test failed. Please check:"
                    echo "1. IB Gateway or TWS is running"
                    echo "2. API is enabled in IB settings" 
                    echo "3. Live trading port 7497 is configured"
                    echo "4. You have live trading permissions"
                    echo "Aborting live trading for safety."
                fi
            else
                echo "Live trading cancelled. Falling back to simulation mode..."
                python automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            fi
            ;;
        *)
            echo "Invalid choice. Defaulting to simulation mode..."
            python automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            ;;
    esac
else
    echo "Setup complete. You can start trading later with:"
    echo "  Simulation:     python automated_trader.py --mode simulation --stock $ticker --semantic-name $semantic_name"
    echo "  IB Paper:       python automated_trader.py --mode ib_paper --stock $ticker --semantic-name $semantic_name"
    echo "  IB Live:        python automated_trader.py --mode ib_live --stock $ticker --semantic-name $semantic_name"
fi
