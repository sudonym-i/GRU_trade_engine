
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

read -p "What ticker do you want to algorithmically trade?: " ticker
read -p "enter semantic name (for sentiment): " semantic_name

python3 main.py webscrape --ticker "$semantic_name"

python3 main.py train --ticker "$ticker"

cd ../integrations_\&_strategy
nano config.json

read -p "Start automated trading now? (y/n): " start
if [ "$start" == "y" || "$start" == "Y" ]; then
    python schedule_trader.py --start
fi
else
    echo "Setup complete."
fi
