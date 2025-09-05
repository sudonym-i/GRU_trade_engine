
echo "Neural_Trade_Engine"
echo "===================="
echo ""
echo ""



# May need to automate model downloading as well

echo "Setting up Python virtual environment.."
echo ""
echo ""
sudo apt install python3-full

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing Python dependencies.."
echo ""
echo ""
pip install -r backend_\&_algorithms/engine/requirements.txt
echo ""
echo ""
echo "Setting up C++ environment.."
echo ""
echo ""
mkdir backend_\&_algorithms/engine/sentiment_model/web_scraper/build
sudo apt-get update && sudo apt upgrade
sudo apt-get install libgtest-dev
sudo apt-get install cmake
sudo apt install libfmt-dev 
sudo apt install curl
sudo apt install libcurl4-gnutls-dev
echo ""
echo "Compiling c++"
echo ""
cd backend_\&_algorithms/engine/sentiment_model/web_scraper/build
cmake ..
make
chmod +x webscrape.exe && echo "C++ build successful"
echo ""
cd ../..
echo ""
echo ""
echo "Setup complete."
echo ""
echo ""

echo ""
echo "===================================="
echo ""
read -p "What ticker do you want to algorithmically trade?: " ticker
read -p "Enter semantic name (EX: GOOG would be -> google): " semantic_name
echo ""
echo ""
read -p "Train sentiment model?: (y/n): " train_models
echo ""

if [ "$train_models" == "y" ] || [ "$train_models" == "Y" ]; then

    ../../../.venv/bin/python3 download_dataset.py
    ../../../.venv/bin/python3 tokenize_pipeline.py
    echo ""
    echo "Training sentiment model..."
    echo ""

    ../../../.venv/bin/python3 train_with_labeled_data.py

fi

cd ..

../.venv/bin/python3 main.py webscrape --ticker "$semantic_name"
echo ""
echo ""
read -p "Train TSR model? (y/n): " train_tsr
echo ""
echo ""
if [ "$train_tsr" == "y" || "$train_tsr" == "Y" ]; then
    echo "Training TSR model for $ticker..."
    ../.venv/bin/python3 main.py train --ticker "$ticker"
fi

cd ../integrations_\&_strategy
# Virtual environment activated via direct python path
nano config.json
echo ""
echo ""
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
            ../.venv/bin/python3 automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            ;;
        2)
            echo "Starting automated trading in IB PAPER TRADING mode..."
            echo "Testing IB connection first..."
            ../.venv/bin/python3 test_ib_connection.py --mode paper
            if [ $? -eq 0 ]; then
                ../.venv/bin/python3 automated_trader.py --mode ib_paper --stock "$ticker" --semantic-name "$semantic_name"
            else
                echo "IB connection test failed. Please check:"
                echo "1. IB Gateway or TWS is running"
                echo "2. API is enabled in IB settings"
                echo "3. Paper trading port 7496 is configured"
                echo "Falling back to simulation mode..."
                ../.venv/bin/python3 automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            fi
            ;;
        3)
            echo "⚠️  WARNING: You selected LIVE TRADING mode with real money!"
            echo "This will execute actual trades in your Interactive Brokers account."
            read -p "Are you absolutely sure you want to continue? (yes/no): " confirm
            if [ "$confirm" == "yes" ]; then
                echo "Testing IB live connection first..."
                ../.venv/bin/python3 test_ib_connection.py --mode live
                if [ $? -eq 0 ]; then
                    echo "Starting LIVE TRADING - Good luck!"
                    ../.venv/bin/python3 automated_trader.py --mode ib_live --stock "$ticker" --semantic-name "$semantic_name"
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
                ../.venv/bin/python3 automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            fi
            ;;
        *)
            echo "Invalid choice. Defaulting to simulation mode..."
            ../.venv/bin/python3 automated_trader.py --mode simulation --stock "$ticker" --semantic-name "$semantic_name"
            ;;
    esac
else
    echo "Setup complete. You can start trading later with:"
    echo "  Simulation:     ../.venv/bin/python3 automated_trader.py --mode simulation --stock $ticker --semantic-name $semantic_name"
    echo "  IB Paper:       ../.venv/bin/python3 automated_trader.py --mode ib_paper --stock $ticker --semantic-name $semantic_name"
    echo "  IB Live:        ../.venv/bin/python3 automated_trader.py --mode ib_live --stock $ticker --semantic-name $semantic_name"
fi
