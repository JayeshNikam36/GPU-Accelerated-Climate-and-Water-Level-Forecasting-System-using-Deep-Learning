# src/models/simple_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from src.data_acquisition.utils import logger

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # take last timestep
        return out

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.0003,
    val_split: float = 0.2,
    patience: int = 10,
    min_delta: float = 0.0001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    dataset = TimeSeriesDataset(X, y)

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SimpleLSTM(input_size=X.shape[-1], output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # No verbose argument — compatible with older PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=4
    )

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Training LSTM on {device} | Train: {len(train_ds)}, Val: {len(val_ds)}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss += criterion(output.squeeze(), batch_y).item()

        avg_val_loss = val_loss / len(val_loader)

        # Step scheduler
        scheduler.step(avg_val_loss)

        # Optional: print learning rate change if desired
        if scheduler.optimizer.param_groups[0]['lr'] < lr * 0.9:
            logger.info(f"Learning rate reduced to {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_lstm_model.pth")
            logger.info(f"New best val loss: {best_val_loss:.6f} → saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info("Training complete")
    return model