def train_gru_model(model, train_tensor, target_tensor, epochs=10, lr=0.001, batch_size=32, device=None,
                    val_tensor=None, val_target=None, verbose=True):
    """
    Train a GRUStockPredictor model using formatted data.

    Args:
        model: GRUStockPredictor instance.
        train_tensor: Input tensor of shape (num_samples, sequence_length, num_features).
        target_tensor: Target tensor of shape (num_samples, 1).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training.
        device: PyTorch device (CPU/GPU).
        val_tensor: Validation input tensor (optional).
        val_target: Validation target tensor (optional).
        verbose (bool): Print training progress.

    Returns:
        Dictionary with training history (train_losses, val_losses)
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    train_tensor = train_tensor.to(device)
    target_tensor = target_tensor.to(device)

    # Move validation data to device if provided
    if val_tensor is not None and val_target is not None:
        val_tensor = val_tensor.to(device)
        val_target = val_target.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    dataset = TensorDataset(train_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epochs = int(epochs)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(loader)
        train_losses.append(avg_train_loss)

        # Validation
        if val_tensor is not None and val_target is not None:
            model.eval()
            with torch.no_grad():
                val_output = model(val_tensor)
                val_loss = criterion(val_output, val_target).item()
                val_losses.append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def load_pretrained_weights(model, pretrained_path, device=None):
    """
    Load pre-trained weights into a model.

    Args:
        model: GRU model instance.
        pretrained_path: Path to pre-trained model weights.
        device: PyTorch device.

    Returns:
        Model with loaded weights.
    """
    import torch
    import os

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(pretrained_path):
        print(f"Warning: Pre-trained model not found at {pretrained_path}")
        return model

    try:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"✓ Successfully loaded pre-trained weights from {pretrained_path}")
    except Exception as e:
        print(f"✗ Error loading pre-trained weights: {e}")

    return model