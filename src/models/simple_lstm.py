# src/models/simple_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])   # take last timestep
        return out.squeeze(-1)         # (batch,)

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    val_split: float = 0.2,
    patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    dataset = TimeSeriesDataset(X, y)

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = SimpleLSTM(input_size=X.shape[-1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Training on {device} | Train: {len(train_ds)}, Val: {len(val_ds)}")

    for epoch in range(epochs):
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
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_lstm_model.pth")
            logger.info(f"New best val loss: {val_loss:.6f} — model saved")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info("Training complete")
    return model

def predict_future(model, last_sequence: np.ndarray, steps: int = 12, device='cuda'):
    """
    Recursively predict future steps using the last lookback sequence.
    last_sequence: shape (1, lookback, n_features)
    """
    model.eval()
    predictions = []
    # Ensure input is (1, seq_len, n_features)
    current = torch.tensor(last_sequence, dtype=torch.float32).to(device)
    n_features = last_sequence.shape[2]

    with torch.no_grad():
        for _ in range(steps):
            # 1. Predict next gage height
            pred = model(current).item() 
            predictions.append(pred)
            
            # 2. Create a new feature vector for this timestep
            # We put the prediction in index 0 (gage_height) 
            # and pad the rest with 0 or the last known value
            new_features = torch.zeros((1, 1, n_features), device=device)
            new_features[0, 0, 0] = pred 
            # Optional: carry over last known wind/precip if desired:
            # new_features[0, 0, 1:] = current[0, -1, 1:] 

            # 3. Shift the window: Drop first, append new
            current = torch.cat((current[:, 1:, :], new_features), dim=1)

    return np.array(predictions)

def load_model(file_path="best_lstm_model.pth", input_size=3):
    model = SimpleLSTM(input_size=input_size)
    model.load_state_dict(torch.load(file_path, map_location='cuda'))
    model.to('cuda')
    model.eval()
    logger.info(f"Loaded model from {file_path}")
    return model