def train_gru_model(model, train_tensor, target_tensor, epochs=10, lr=0.001, batch_size=32, device=None,
                    val_tensor=None, val_target=None, verbose=True, loss_type='directional', loss_kwargs=None):
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
        loss_type (str): Type of loss function ('mse', 'directional', 'weighted_mse', 'huber_directional').
        loss_kwargs (dict): Additional arguments for the loss function.

    Returns:
        Dictionary with training history (train_losses, val_losses)
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from algorithms.gru_model.loss_functions import get_loss_function

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

    # Get loss function
    if loss_kwargs is None:
        loss_kwargs = {}
    criterion = get_loss_function(loss_type, **loss_kwargs)

    # Extract previous values (last value from sequence) for directional loss
    # Shape: (num_samples, num_features) -> we need the 'Close' price (index 3)
    previous_values = train_tensor[:, -1, 3].unsqueeze(1)  # Index 3 is 'Close' in OHLCV

    dataset = TensorDataset(train_tensor, target_tensor, previous_values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epochs = int(epochs)
    train_losses = []
    val_losses = []

    # Check if loss function supports directional mode
    use_directional = hasattr(criterion, 'direction_weight')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data in loader:
            batch_x, batch_y = batch_data[0], batch_data[1]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)

            # Use directional loss if available
            if use_directional and len(batch_data) > 2:
                batch_prev = batch_data[2].to(device)
                loss = criterion(output, batch_y, batch_prev)
            else:
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

                # Use directional loss for validation if available
                if use_directional:
                    val_prev = val_tensor[:, -1, 3].unsqueeze(1)
                    val_loss = criterion(val_output, val_target, val_prev).item()
                else:
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