# Train a logistic regression sentiment model using formatted YouTube data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from algorithms.sentiment_model.format_youtube_data import format_youtube_data
from algorithms.sentiment_model.logistic_model import LogisticSentimentModel


def load_labels(label_path, num_samples):
	"""
	Dummy label loader. Replace with actual label loading logic.
	For demonstration, generates random binary labels.
	"""
	# Example: return torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)
	# TODO: Replace with actual label loading from file
	return torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)

def train_sentiment_model(
	raw_path,
	vectorizer_path,
	features_path,
	model_save_path,
	label_path=None,
	max_features=1000,
	epochs=20,
	lr=0.01
):
	"""
	Trains a logistic regression sentiment model using formatted YouTube data.
	Args:
		raw_path: Path to raw YouTube data.
		vectorizer_path: Path to vectorizer pickle file.
		features_path: Path to save features (npy).
		model_save_path: Path to save trained model.
		label_path: Path to labels (optional, uses dummy if None).
		max_features: Number of TF-IDF features.
		epochs: Training epochs.
		lr: Learning rate.
	Returns:
		model: Trained model.
	"""
	features_tensor, vectorizer, samples = format_youtube_data(
		raw_path=raw_path,
		output_features_path=features_path,
		output_vectorizer_path=vectorizer_path,
		max_features=max_features,
		use_existing_vectorizer=True
	)

	if label_path is not None and os.path.exists(label_path):
		labels_np = np.load(label_path)
		labels = torch.tensor(labels_np, dtype=torch.float32)
		if labels.ndim == 1:
			labels = labels.unsqueeze(1)
	else:
		labels = load_labels(label_path, features_tensor.shape[0])

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
	return model

if __name__ == "__main__":
	# Example usage for CLI
	base_dir = os.path.dirname(__file__)
	train_sentiment_model(
		raw_path=os.path.join(base_dir, '../../data/youtube_data.raw'),
		vectorizer_path=os.path.join(base_dir, 'youtube_vectorizer.pkl'),
		features_path=os.path.join(base_dir, 'youtube_features.npy'),
		model_save_path=os.path.join(base_dir, 'sentiment_logistic_model.pt'),
		label_path=os.path.join(base_dir, '../../data/youtube_labels.npy'),
		max_features=1000,
		epochs=20,
		lr=0.01
	)
