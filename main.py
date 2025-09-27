# This imports all of the ML objects, allowing us to control what we do wth these models
from algorithms.gru_model.gru_object import GRUModel
from algorithms.sentiment_model.youtube_sentiment import YouTubeSentimentAnalyzer

import torch
import argparse



# ============================
# testing data

symbol = 'ORCL'
input_size = 12  # OHLCV
hidden_size = 2048
output_size = 1
sequence_length = 150

# ============================


def print_device_info():
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA device:', torch.cuda.get_device_name(0))
    else:
        print('No CUDA device detected. Running on CPU or integrated graphics.')



def main():
    print_device_info()

    parser = argparse.ArgumentParser(description="GRU Trade Engine")
    parser.add_argument('--mode', choices=['t', 'p', 's'], required=True, help="Mode: t=train, p=predict, s=skip GRU model")
    parser.add_argument('--symbol', type=str, default=symbol, help="Stock symbol")
    parser.add_argument('--epochs', type=int, default=30, help="Epochs for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training")
    args = parser.parse_args()

    mode = args.mode
    symbol_arg = args.symbol

    if mode != 's' and mode == 't':
        # ---------- TRAIN MODEL -------------
        gru_model = GRUModel(input_size, hidden_size, output_size)
        gru_model.data_dir = "./data"
        gru_model.pull_data(symbol=symbol_arg, period="1y")
        gru_model.format_data()
        gru_model.train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        gru_model.save_model(f"./algorithms/gru_model/models/{symbol_arg}_gru_model.pth")

    elif mode != 's' and mode == 'p':
        # -------------- TEST MODEL ------------
        gru_model = GRUModel(input_size, hidden_size, output_size)
        gru_model.data_dir = "./data"
        gru_model.load_model(f"./algorithms/gru_model/models/{symbol_arg}_gru_model.pth")
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