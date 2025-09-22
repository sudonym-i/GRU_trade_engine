
# Train a logistic regression sentiment model using formatted YouTube data


# Fix import path for local and package usage
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from algorithms.sentiment_model.format_youtube_data import format_youtube_data
    from algorithms.sentiment_model.logistic_model import LogisticSentimentModel
except ImportError:
    from format_youtube_data import format_youtube_data
    from logistic_model import LogisticSentimentModel

def load_labels(label_path, num_samples):
	"""
	Dummy label loader. Replace with actual label loading logic.
	For demonstration, generates random binary labels.
	"""
	# Example: return torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)
	# TODO: Replace with actual label loading from file
	return torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)


def train_sentiment_model(
	features_path,
	labels_path,
	vectorizer_path,
	model_save_path,
	epochs=20,
	lr=0.01
):
	"""
	Trains a logistic regression sentiment model using pre-extracted features and labels (e.g., from Sentiment140).
	Args:
		features_path: Path to features (npy).
		labels_path: Path to labels (npy).
		vectorizer_path: Path to vectorizer pickle file.
		model_save_path: Path to save trained model.
		epochs: Training epochs.
		lr: Learning rate.
	Returns:
		model: Trained model.
	"""
	# Load features and labels
	features_np = np.load(features_path)
	labels_np = np.load(labels_path)
	features_tensor = torch.tensor(features_np, dtype=torch.float32)
	labels = torch.tensor(labels_np, dtype=torch.float32)
	if labels.ndim == 1:
		labels = labels.unsqueeze(1)
	# Load vectorizer for later inference compatibility
	with open(vectorizer_path, 'rb') as f:
		vectorizer = pickle.load(f)

	input_dim = features_tensor.shape[1]
	model = LogisticSentimentModel(input_dim)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		outputs = model(features_tensor)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		if (epoch + 1) % 5 == 0 or epoch == 0:
			print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

	torch.save(model.state_dict(), model_save_path)
	print(f"Model saved to {model_save_path}")
	return model, vectorizer

if __name__ == "__main__":
    # Example usage for CLI: train on Sentiment140 data
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, '../../data'))
    train_sentiment_model(
        features_path=os.path.join(data_dir, 'twitter_features.npy'),
        labels_path=os.path.join(data_dir, 'twitter_labels.npy'),
        vectorizer_path=os.path.join(data_dir, 'twitter_vectorizer.pkl'),
        model_save_path=os.path.join(data_dir, 'sentiment_logistic_model.pt'),
        epochs=20,
        lr=0.01
    )
