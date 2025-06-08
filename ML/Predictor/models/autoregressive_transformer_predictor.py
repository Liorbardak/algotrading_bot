import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoregressiveTimeSeriesDecoder(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True  # Using batch_first=True for easier handling
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, input_size)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, tgt, memory=None):
        """
        tgt: [batch_size, seq_len, input_size] - target sequence
        memory: [batch_size, memory_len, d_model] - encoder memory (optional)
        """
        batch_size, seq_len, _ = tgt.shape

        # Embed and add positional encoding
        tgt_embedded = self.embedding(tgt) + self.positional_encoding[:seq_len]

        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)

        # If no memory provided, use self-attention only (decoder-only)
        if memory is None:
            memory = torch.zeros(batch_size, 1, self.d_model).to(tgt.device)

        # Apply transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask
        )

        # Project to output size
        output = self.output_proj(output)

        return output


def create_training_batch(data, seq_len, pred_len, batch_size):
    """
    Create training batches for autoregressive training

    Args:
        data: [total_timesteps, features] - full time series data
        seq_len: length of input sequence
        pred_len: length of prediction sequence
        batch_size: number of samples per batch

    Returns:
        inputs: [batch_size, seq_len, features] - input sequences
        targets: [batch_size, seq_len + pred_len, features] - full sequences for teacher forcing
        labels: [batch_size, seq_len + pred_len, features] - shifted targets for loss calculation
    """
    total_len = seq_len + pred_len
    samples = []

    # Create samples
    for i in range(len(data) - total_len + 1):
        samples.append(data[i:i + total_len])

    # Randomly sample batch
    indices = np.random.choice(len(samples), batch_size, replace=True)
    batch_samples = [samples[i] for i in indices]
    batch_samples = torch.stack(batch_samples)

    # Split into inputs and targets
    inputs = batch_samples[:, :seq_len]  # Historical data
    full_sequence = batch_samples[:, :]  # Historical + future data

    # For autoregressive training:
    # - targets: what we feed to the decoder (includes start token/historical data)
    # - labels: what we want to predict (shifted by 1)
    targets = full_sequence[:, :-1]  # Everything except last timestep
    labels = full_sequence[:, 1:]  # Everything except first timestep (shifted)

    return inputs, targets, labels


def train_step(model, data, seq_len, pred_len, batch_size, optimizer, criterion):
    """Single training step"""
    model.train()

    # Create batch
    inputs, targets, labels = create_training_batch(data, seq_len, pred_len, batch_size)

    # Forward pass
    # In pure decoder-only setup, we feed the targets (with teacher forcing)
    outputs = model(targets)

    # Calculate loss
    # We typically only calculate loss on the prediction part
    pred_outputs = outputs[:, seq_len:]  # Only prediction timesteps
    pred_labels = labels[:, seq_len:]  # Only prediction labels

    loss = criterion(pred_outputs.reshape(-1, pred_outputs.size(-1)),
                     pred_labels.reshape(-1, pred_labels.size(-1)))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def inference(model, initial_sequence, pred_len):
    """Autoregressive inference"""
    model.eval()

    with torch.no_grad():
        current_sequence = initial_sequence.clone()

        for _ in range(pred_len):
            # Predict next timestep
            output = model(current_sequence)
            next_timestep = output[:, -1:, :]  # Last timestep prediction

            # Append to sequence
            current_sequence = torch.cat([current_sequence, next_timestep], dim=1)

        # Return only the predicted part
        return current_sequence[:, -pred_len:, :]


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    torch.manual_seed(42)
    time_steps = 1000
    features = 1
    data = torch.sin(torch.linspace(0, 40 * np.pi, time_steps)).unsqueeze(1)

    # Model parameters
    seq_len = 50  # Historical sequence length
    pred_len = 10  # Prediction length
    batch_size = 32

    # Initialize model
    model = AutoregressiveTimeSeriesDecoder(
        input_size=features,
        d_model=64,
        nhead=4,
        num_layers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    print("Training autoregressive time series model...")
    for epoch in range(100):
        loss = train_step(model, data, seq_len, pred_len, batch_size, optimizer, criterion)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Test inference
    test_input = data[:seq_len].unsqueeze(0)  # [1, seq_len, features]
    predictions = inference(model, test_input, pred_len)

    print(f"Input shape: {test_input.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print("Training completed!")