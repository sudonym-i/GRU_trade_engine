
import torch
from transformers import pipeline

class YouTubeSentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

    def read_youtube_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    def analyze_file(self, file_path):
        texts = self.read_youtube_data(file_path)
        results = []
        for text in texts:
            truncated_text = text[:512]
            try:
                result = self.sentiment_analyzer(truncated_text)[0]
                results.append({"score": result["score"]})
            except Exception as e:
                print(f"Error processing text: {e}\nText: {truncated_text[:100]}...")
        final_result = sum(r['score'] for r in results) / len(results) if results else 0
        return {
            "average_score": final_result,
            "num_entries": len(results),
            "results": results
        }

    def analyze_text(self, text):
        truncated_text = text[:512]
        try:
            result = self.sentiment_analyzer(truncated_text)[0]
            return result
        except Exception as e:
            print(f"Error processing text: {e}\nText: {truncated_text[:100]}...")
            return None
