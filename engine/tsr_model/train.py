import torch
from torch.utils.data import DataLoader as TorchDataLoader
try:
    from .visualizations import plot_training_loss
except ImportError:
    from visualizations import plot_training_loss


def train_gru_predictor(model, dataset, epochs=10, batch_size=32, lr=1e-3, plot_loss=True):
	print(f"[INFO] Starting training for {epochs} epochs, batch size {batch_size}, learning rate {lr}")
	loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	model.train()
	
	losses = []
	
	for epoch in range(epochs):
		total_loss = 0
		batch_count = 0
		for X_batch, y_batch in loader:
			optimizer.zero_grad()
			output = model(X_batch)
			loss = criterion(output, y_batch)
			
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * X_batch.size(0)
			batch_count += 1
		avg_loss = total_loss / len(dataset)
		losses.append(avg_loss)
		print(f"[INFO] Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
	
	print("[INFO] Training complete.")

	torch.save(model.state_dict(), "engine/tsr_model/cached_models/model.pth")
	
	if plot_loss and losses:
		plot_training_loss(losses)
	
	return losses

