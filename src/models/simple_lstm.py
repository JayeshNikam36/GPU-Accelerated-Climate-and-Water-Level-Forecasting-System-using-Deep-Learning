# src/models/simple_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Simple dataset for time-series data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleLSTM(nn.Module):
    """
    LSTM with self-attention for better long-range dependency capture.
    """
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Self-attention layer after LSTM
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,  # hidden_size must be divisible by num_heads (128/8=16)
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Self-attention: query=key=value = lstm_out
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Take the output of the last timestep
        out = self.fc(attn_out[:, -1, :])  # (batch, output_size)
        return out.squeeze(-1)  # (batch,)

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    config_path: str = "configs/models/lstm_config.yaml",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the LSTM model using config from YAML.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    train_cfg = config['training']

    dataset = TimeSeriesDataset(X, y)

    train_size = int((1 - train_cfg['val_split']) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg['batch_size'], shuffle=False)

    model = SimpleLSTM(
        input_size=model_cfg['input_size'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        dropout=model_cfg['dropout'],
        output_size=model_cfg['output_size']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=1e-5)

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Training on {device} | Train: {len(train_ds)}, Val: {len(val_ds)}")

    for epoch in range(train_cfg['epochs']):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_ds)

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{train_cfg['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_lstm_model.pth")
            logger.info(f"New best val loss: {val_loss:.6f} — saved model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_cfg['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info("Training complete")
    return model

def predict_future(model, last_sequence: np.ndarray, steps: int = 12, device='cuda'):
    """
    Recursively predict future steps autoregressively.
    
    Args:
        last_sequence: np.ndarray of shape (1, lookback, features)
        steps: number of future steps to predict
    
    Returns:
        np.ndarray of shape (steps,)
    """
    model.eval()
    predictions = []
    current = torch.tensor(last_sequence, dtype=torch.float32).to(device)
    n_features = current.shape[2]
    with torch.no_grad():
        for _ in range(steps):
            pred = model(current).item()  # scalar prediction
            predictions.append(pred)
            # Shift window: remove oldest timestep, append new prediction
            new_step = torch.zeros((1, 1, n_features), device=device)
            new_step[0, 0, 0] = pred 
            
            # Shift window: remove oldest timestep, append new prediction vector
            current = torch.cat((current[:, 1:, :], new_step), dim=1)

    return np.array(predictions)

def load_model(file_path="best_lstm_model.pth", input_size=3):
    """Load saved model weights."""
    model = SimpleLSTM(input_size=input_size)
    model.load_state_dict(torch.load(file_path, map_location='cuda'))
    model.to('cuda')
    model.eval()
    logger.info(f"Loaded model from {file_path}")
    return model