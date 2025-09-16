def train_gru_model(model, train_tensor, target_tensor, epochs=10, lr=0.001, batch_size=32):
    """
    Train a GRUStockPredictor model using formatted data.

    Args:
        model: GRUStockPredictor instance.
        train_tensor: Input tensor of shape (num_samples, sequence_length, num_features).
        target_tensor: Target tensor of shape (num_samples, 1).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    dataset = TensorDataset(train_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    epochs = int(epochs)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

    return None