def train_gru_model(model, train_tensor, target_tensor, epochs=10, lr=0.001):
	"""
	Train a GRUStockPredictor model using formatted data.

	Args:
		model: GRUStockPredictor instance.
		train_tensor: Input tensor of shape (num_samples, sequence_length, num_features).
		target_tensor: Target tensor of shape (num_samples, 1).
		epochs (int): Number of training epochs.
		lr (float): Learning rate.
	"""
	import torch
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.MSELoss()
	epochs = int(epochs)
	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		output = model(train_tensor)
		loss = criterion(output, target_tensor)
		loss.backward()
		optimizer.step()
		print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
