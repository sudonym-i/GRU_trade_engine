# This imports all of the ML objects, allowing us to control what we do wth these models
from algorithms.gru_model.gru_object import GRUModel
from algorithms.sentiment_model.youtube_sentiment import YouTubeSentimentAnalyzer
from config_loader import get_config

import torch
import argparse
import requests
import os

def run_youtube_sentiment(config):
        analyzer = YouTubeSentimentAnalyzer(config)
        data_path = config.get('paths', 'youtube_data')
        result = analyzer.analyze_file(data_path)
        print(f"\n**Average sentiment score: {round(result['average_score'], 2)}/1**\n")


# Discord webhook URL (read from config, .discord_webhook file, or env variable)
def get_discord_webhook_url(config):
    # Try config first
    webhook_url = config.get('discord', 'webhook_url')
    if webhook_url:
        return webhook_url

    # Try .discord_webhook file
    webhook_file = config.get('paths', 'discord_webhook_file', default='.discord_webhook')
    try:
        with open(webhook_file, 'r') as f:
            return f.read().strip()
    except Exception:
        pass

    # Fall back to environment variable
    return os.getenv("DISCORD_WEBHOOK_URL", "")

def send_discord_message(message: str, webhook_url: str):
    """
    Send a plain text message to Discord using a webhook.
    """
    data = {"content": message}
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Discord message: {e}")
        return False
    return True


def main():
    # Load configuration
    config = get_config()

    # Parse command line arguments (only for mode override)
    parser = argparse.ArgumentParser(description="GRU Trade Engine - Config-driven ML trading system")
    parser.add_argument('--mode', choices=['t', 'p', 's', 'discord', 'pretrain', 'finetune'],
                        help="Mode: t=train, p=predict, s=sentiment only, pretrain=pre-train on multiple stocks, finetune=fine-tune on target stock, discord=send to Discord (overrides config)")
    parser.add_argument('--config', type=str, default='config.json', help="Path to config file")
    args = parser.parse_args()

    # Use mode from args if provided, otherwise use from config
    mode = args.mode if args.mode else config.get('execution', 'mode', default='predict')

    # Get all configuration values
    stock_config = config.get_stock_config()
    gru_config = config.get_gru_config()
    training_config = config.get_training_config()
    paths_config = config.get_paths_config()

    symbol_arg = stock_config.get('ticker', 'MSFT')
    input_size = gru_config.get('input_size', 12)
    hidden_size = gru_config.get('hidden_size', 2048)
    output_size = gru_config.get('output_size', 1)

    if mode == 'discord':
        # Read prediction and sentiment output files from config
        prediction_text = ""
        sentiment_text = ""
        prediction_file = paths_config.get('prediction_output')
        sentiment_file = paths_config.get('sentiment_output')

        if prediction_file:
            try:
                with open(prediction_file, 'r') as f:
                    prediction_text = f.read().strip()
            except Exception as e:
                prediction_text = f"Error reading prediction file: {e}"
        if sentiment_file:
            try:
                with open(sentiment_file, 'r') as f:
                    sentiment_text = f.read().strip()
            except Exception as e:
                sentiment_text = f"Error reading sentiment file: {e}"

        # Format message
        message = f" Price Prediction:\n{prediction_text}\n\n\n YouTube Sentiment Analysis:\n{sentiment_text}"

        # Get webhook URL and send to Discord
        webhook_url = get_discord_webhook_url(config)
        success = send_discord_message(message, webhook_url)
        if success:
            print("Discord message sent successfully.")
        else:
            print("Failed to send Discord message.")
        return

    if mode == 'pretrain':
        # ---------- PRE-TRAIN ON MULTIPLE STOCKS -------------
        print(f"\n{'='*60}")
        print("STARTING PRE-TRAINING WORKFLOW")
        print(f"{'='*60}\n")

        gru_model = GRUModel(input_size, hidden_size, output_size, config)
        gru_model.data_dir = paths_config.get('data_dir', 'data')

        # Get pre-training configuration
        pretrain_stocks = training_config.get('pretrain_stocks', ['AMD', 'NVDA', 'INTC', 'TSM'])
        pretrain_epochs = training_config.get('pretrain_epochs', 40)
        pretrain_lr = training_config.get('pretrain_lr', 0.0005)
        batch_size = training_config.get('batch_size', 32)
        validation_split = training_config.get('validation_split', 0.2)
        training_period = stock_config.get('training_period', 'max')

        # Pre-train on multiple stocks
        gru_model.pretrain_on_multiple_stocks(
            stock_symbols=pretrain_stocks,
            period=training_period,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            batch_size=batch_size,
            validation_split=validation_split
        )

        # Save pre-trained model
        pretrained_path = training_config.get('pretrained_model_path',
                                               'algorithms/gru_model/models/pretrained_gru.pth')
        gru_model.save_model(pretrained_path)
        print(f"\n✓ Pre-trained model saved to: {pretrained_path}\n")

    elif mode == 'finetune':
        # ---------- FINE-TUNE ON TARGET STOCK -------------
        print(f"\n{'='*60}")
        print("STARTING FINE-TUNING WORKFLOW")
        print(f"{'='*60}\n")

        gru_model = GRUModel(input_size, hidden_size, output_size, config)
        gru_model.data_dir = paths_config.get('data_dir', 'data')

        # Get fine-tuning configuration
        finetune_epochs = training_config.get('finetune_epochs', 25)
        finetune_lr = training_config.get('finetune_lr', 0.0001)
        batch_size = training_config.get('batch_size', 32)
        validation_split = training_config.get('validation_split', 0.2)
        training_period = stock_config.get('training_period', 'max')
        pretrained_path = training_config.get('pretrained_model_path',
                                               'algorithms/gru_model/models/pretrained_gru.pth')

        # Fine-tune on target stock
        gru_model.finetune_on_target_stock(
            symbol=symbol_arg,
            period=training_period,
            epochs=finetune_epochs,
            lr=finetune_lr,
            batch_size=batch_size,
            pretrained_path=pretrained_path,
            validation_split=validation_split
        )

        # Save fine-tuned model
        model_path = gru_config.get('model_path', 'algorithms/gru_model/models/gru_model.pth')
        gru_model.save_model(model_path)
        print(f"\n✓ Fine-tuned model saved to: {model_path}\n")

    elif mode != 's' and mode == 't':
        # ---------- STANDARD TRAIN MODEL (SINGLE STOCK) -------------
        gru_model = GRUModel(input_size, hidden_size, output_size, config)
        gru_model.data_dir = paths_config.get('data_dir', 'data')
        gru_model.pull_data(
            symbol=symbol_arg,
            period=stock_config.get('training_period', 'max')
        )
        gru_model.format_data()
        gru_model.train(
            epochs=training_config.get('epochs', 30),
            lr=training_config.get('learning_rate', 0.001),
            batch_size=training_config.get('batch_size', 1)
        )
        model_path = gru_config.get('model_path', 'algorithms/gru_model/models/gru_model.pth')
        gru_model.save_model(model_path)

    elif mode != 's' and mode == 'p':
        # -------------- PREDICT ------------
        gru_model = GRUModel(input_size, hidden_size, output_size, config)
        gru_model.data_dir = paths_config.get('data_dir', 'data')
        model_path = gru_config.get('model_path', 'algorithms/gru_model/models/gru_model.pth')
        gru_model.load_model(model_path)
        gru_model.pull_data(
            symbol=symbol_arg,
            period=stock_config.get('data_period', '3mo')
        )
        # Use existing scaler from loaded model for prediction
        gru_model.format_data(for_training=False, use_existing_scaler=True)

        # Predict only from the most recent sequence
        gru_model.predict(predict_last_only=True)

        # DEBUG: Show normalized prediction value
        normalized_prediction = gru_model.output_tensor.item()
        print(f"\n[DEBUG] Normalized prediction value: {normalized_prediction:.6f}")

        # Get the prediction (only one value now)
        price_prediction = gru_model.un_normalize()[0]

        # Get current price and recent history
        current_price = gru_model.raw_data['Close'].iloc[-1]
        last_10_prices = gru_model.raw_data['Close'].tail(10).values

        # DEBUG: Show scaler range
        if gru_model.scaler is not None:
            close_idx = 3  # 'Close' is at index 3 in OHLCV
            scaler_min = gru_model.scaler.data_min_[close_idx]
            scaler_max = gru_model.scaler.data_max_[close_idx]
            print(f"[DEBUG] Scaler range for 'Close': ${scaler_min:.2f} - ${scaler_max:.2f}")
            print(f"[DEBUG] Current price normalized: {(current_price - scaler_min) / (scaler_max - scaler_min):.6f}")

        print(f"\nRecent price history (last 10 days): \n")
        for price in last_10_prices:
            print(f" -> {round(price, 2)}")

        print(f"\n")
        print(f"Current closing price: ${round(current_price, 2)}")
        print(f"**Predicted next-day closing price: ${round(price_prediction, 2)}**")

        # Show prediction change
        change = price_prediction - current_price
        change_pct = (change / current_price) * 100
        direction = "↑" if change > 0 else "↓"
        print(f"Predicted change: {direction} ${abs(round(change, 2))} ({round(change_pct, 2)}%)")
        print()

    if mode == 's':
        run_youtube_sentiment(config)

#testing
if __name__ == "__main__":
    main()
