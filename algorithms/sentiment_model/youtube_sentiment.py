import torch
from transformers import pipeline

# Load prebuilt sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def read_youtube_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def main():
    data_path = "/home/isaac/projects/GRU_trade_engine/data/youtube_data.raw"
    texts = read_youtube_data(data_path)
    results = []
    for text in texts:
        truncated_text = text[:512]
        try:
            result = sentiment_analyzer(truncated_text)[0]
            results.append({"score": result["score"]})
        except Exception as e:
            print(f"Error processing text: {e}\nText: {truncated_text[:100]}...")
    # Print results
    final_result = 0;

    for r in results:
        final_result += r['score']
    
    final_result = final_result / len(results) if results else 0
    
    print(f"{final_result}")
    print(f"Processed {len(results)} entries.")

if __name__ == "__main__":
    main()
