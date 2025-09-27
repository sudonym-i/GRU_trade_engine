# This imports all of the ML objects, allowing us to control what we do wth these models
from algorithms.gru_model.gru_object import GRUModel
from algorithms.sentiment_model.youtube_sentiment import YouTubeSentimentAnalyzer

import torch
import argparse

import requests
import os



# ============================
# testing data
symbol = 'MSFT'
input_size = 12  # OHLCV
hidden_size = 2048
output_size = 1
sequence_length = 150
# ============================


# Discord webhook URL (read from .discord_webhook file if present, else env variable)
def get_discord_webhook_url():
    try:
        with open('.discord_webhook', 'r') as f:
            return f.read().strip()
    except Exception:
        return os.getenv("DISCORD_WEBHOOK_URL", "")

DISCORD_WEBHOOK_URL = get_discord_webhook_url()

def send_discord_message(message: str):
    """
    Send a plain text message to Discord using a webhook.
    """
    data = {"content": message}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Discord message: {e}")
        return False
    return True


def print_device_info():
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA device:', torch.cuda.get_device_name(0))
    else:
        print('No CUDA device detected. Running on CPU or integrated graphics.')



def main():
    print_device_info()


    parser = argparse.ArgumentParser(description="GRU Trade Engine")
    parser.add_argument('--mode', choices=['t', 'p', 's', 'discord'], required=True, help="Mode: t=train, p=predict, s=skip GRU model, discord=send results to Discord")
    parser.add_argument('--symbol', type=str, default=symbol, help="Stock symbol")
    parser.add_argument('--epochs', type=int, default=30, help="Epochs for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training")
    parser.add_argument('--prediction_file', type=str, help="Path to GRU prediction output file (for Discord mode)")
    parser.add_argument('--sentiment_file', type=str, help="Path to sentiment analysis output file (for Discord mode)")
    args = parser.parse_args()

    mode = args.mode
    symbol_arg = args.symbol

    if mode == 'discord':
        # Read prediction and sentiment output files
        prediction_text = ""
        sentiment_text = ""
        if args.prediction_file:
            try:
                with open(args.prediction_file, 'r') as f:
                    prediction_text = f.read().strip()
            except Exception as e:
                prediction_text = f"Error reading prediction file: {e}"
        if args.sentiment_file:
            try:
                with open(args.sentiment_file, 'r') as f:
                    sentiment_text = f.read().strip()
            except Exception as e:
                sentiment_text = f"Error reading sentiment file: {e}"
        # Format message
        message = f"📈 Price Prediction:\n{prediction_text}\n\n📰 Sentiment Analysis:\n{sentiment_text}"
        # Send to Discord
        success = send_discord_message(message)
        if success:
            print("Discord message sent successfully.")
        else:
            print("Failed to send Discord message.")
        return

    if mode != 's' and mode == 't':
        # ---------- TRAIN MODEL -------------
        gru_model = GRUModel(input_size, hidden_size, output_size)
        gru_model.data_dir = "data"
        gru_model.pull_data(symbol=symbol_arg, period="max")
        gru_model.format_data()
        gru_model.train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        gru_model.save_model(f"./algorithms/gru_model/models/{symbol_arg}_gru_model.pth")

    elif mode != 's' and mode == 'p':
        # -------------- PREDICT ------------
        gru_model = GRUModel(input_size, hidden_size, output_size)
        gru_model.data_dir = "data"
        gru_model.load_model(f"algorithms/gru_model/models/{symbol_arg}_gru_model.pth")
        gru_model.pull_data(symbol=symbol_arg, period="3mo")
        gru_model.format_data()
        gru_model.predict()
        price_prediction = gru_model.un_normalize()[-1]
        print(f"\nThe past 10 day window: \n{gru_model.raw_data['Close'].tail(10).values}")
        print(f"\nPredicted future closing price: {price_prediction}")

    def run_youtube_sentiment():
        analyzer = YouTubeSentimentAnalyzer()
        data_path = "./data/youtube_data.raw"
        result = analyzer.analyze_file(data_path)
        print(f"Average sentiment score: {result['average_score']}")
        print(f"Processed {result['num_entries']} entries.")

    run_youtube_sentiment()

#testing
if __name__ == "__main__":
    main()
