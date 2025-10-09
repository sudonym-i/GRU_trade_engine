
import torch
from transformers import pipeline

class YouTubeSentimentAnalyzer:
    def __init__(self, config=None, model_name=None):
        """
        Initialize sentiment analyzer.

        Args:
            config: ConfigLoader instance (optional)
            model_name: Model name override (optional)
        """
        self.config = config

        # Get model name from config or use provided/default
        if model_name is None and config:
            sentiment_config = config.get_sentiment_config()
            model_name = sentiment_config.get('model_name', 'distilbert-base-uncased-finetuned-sst-2-english')
        elif model_name is None:
            model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

        self.model_name = model_name
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

        # Get max text length from config
        if config:
            sentiment_config = config.get_sentiment_config()
            self.max_text_length = sentiment_config.get('max_text_length', 512)
        else:
            self.max_text_length = 512

    def read_youtube_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    def analyze_file(self, file_path):
        texts = self.read_youtube_data(file_path)
        results = []
        for text in texts:
            truncated_text = text[:self.max_text_length]
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
        truncated_text = text[:self.max_text_length]
        try:
            result = self.sentiment_analyzer(truncated_text)[0]
            return result
        except Exception as e:
            print(f"Error processing text: {e}\nText: {truncated_text[:100]}...")
            return None
