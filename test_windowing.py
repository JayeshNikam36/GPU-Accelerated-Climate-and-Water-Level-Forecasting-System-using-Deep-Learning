# test_windowing.py
from src.preprocessing.loader import DataLoader
from src.preprocessing.windowing import create_sequences
from src.preprocessing.normalization import GPUNormalizer
import logging
import cudf
import cupy as cp
import torch
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("Full Preprocessing + Training + Evaluation Pipeline (GPU)")
    print("=" * 80)

    # CONFIG
    usgs_file   = "data/raw/usgs/01388500_2025-01-01_to_2026-01-20.json"
    noaa_file   = "data/raw/noaa/USW00014734_2025-01-01_to_2026-01-20.json"
    nasa_folder = "data/raw/nasa/GPM_3IMERGDF"

    lookback   = 96
    horizon    = 12              # ← forecasting 12 steps ahead
    stride     = 12              # reasonable speed
    batch_size = 64
    epochs     = 100
    val_split  = 0.2
    patience   = 10

    # Load
    loader = DataLoader(use_gpu=True)

    print("Loading USGS and NOAA...")
    dfs = loader.load_all(usgs_file=usgs_file, noaa_file=noaa_file)

    print(f"USGS shape: {dfs['usgs'].shape}")
    if 'noaa' in dfs:
        print(f"NOAA shape: {dfs['noaa'].shape}")

    print("\nLoading NASA precipitation files...")
    dfs['nasa'] = loader.load_nasa_multi(nasa_folder)
    if 'nasa' in dfs:
        print(f"NASA combined shape: {dfs['nasa'].shape}")

    # Merge
    print("\nMerging sources (forward-fill)...")
    merged_df = loader.merge_sources(dfs['usgs'], dfs.get('noaa'), dfs.get('nasa'))
    print("Merged shape:", merged_df.shape)
    print("Columns:", list(merged_df.columns))

    # Windowing - predict MEAN of next 12 hours
    print(f"\nCreating sequences (predict MEAN of next {horizon} steps)...")
    X, y = create_sequences(
        merged_df,
        target_col='gage_height_ft',
        timestamp_col='timestamp',
        lookback=lookback,
        horizon=48,
        feature_cols=['gage_height_ft', 'precip_mm', 'AWND'],           # ← pure autoregressive for now
        stride=stride
    )
    print("X shape:", X.shape)
    print("y shape (mean over horizon):", y.shape)

    # Normalization (per feature - here only 1)
    print("\nNormalizing each feature independently...")
    X_norm = cp.zeros_like(X)
    for f in range(X.shape[2]):
        col = X[:, :, f].ravel()[:, None]
        col_df = cudf.DataFrame(col, columns=[f'f{f}'])
        norm = GPUNormalizer(method='minmax')
        norm.fit(col_df, col_df.columns)
        norm_col = norm.transform(col_df, col_df.columns).values.ravel()
        X_norm[:, :, f] = norm_col.reshape(X.shape[0], X.shape[1])

    print("Normalized X shape:", X_norm.shape)
    print("First window sample:\n", X_norm[0][:5].get())

    # Training
    print("\nStarting LSTM training...")
    from src.models.simple_lstm import train_model

    X_cpu = X_norm.get()
    y_cpu = y.get() if hasattr(y, 'get') else y

    model = train_model(
        X_cpu, y_cpu,
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
        patience=patience,
        device='cuda'
    )

    # Evaluation
    print("\nEvaluating on validation set...")
    model.eval()

    val_size = int(val_split * len(y))
    X_val = X[-val_size:]
    y_val = y[-val_size:]

    with torch.no_grad():
        X_val_tensor = torch.as_tensor(X_val.get(), dtype=torch.float32).to('cuda')
        preds_val = model(X_val_tensor).squeeze().cpu().numpy()

    actual_val = y_val.get() if hasattr(y_val, 'get') else y_val
    actual_val = np.asarray(actual_val).ravel()
    preds_val  = preds_val.ravel()

    # Denormalization
    target_normalizer = GPUNormalizer(method='minmax')
    target_data = merged_df[['gage_height_ft']].astype('float32')
    target_normalizer.fit(target_data, columns=['gage_height_ft'])

    pred_df   = cudf.DataFrame(preds_val.reshape(-1,1), columns=['gage_height_ft'])
    pred_denorm = target_normalizer.inverse_transform(pred_df, ['gage_height_ft'])['gage_height_ft'].to_numpy()

    actual_df   = cudf.DataFrame(actual_val.reshape(-1,1), columns=['gage_height_ft'])
    actual_denorm = target_normalizer.inverse_transform(actual_df, ['gage_height_ft'])['gage_height_ft'].to_numpy()

    # Metrics
    rmse_norm = np.sqrt(np.mean((preds_val - actual_val)**2))
    mae_norm  = np.mean(np.abs(preds_val - actual_val))
    rmse_orig = np.sqrt(np.mean((pred_denorm - actual_denorm)**2))
    mae_orig  = np.mean(np.abs(pred_denorm - actual_denorm))

    print(f"Val RMSE (norm): {rmse_norm:.6f}   MAE (norm): {mae_norm:.6f}")
    print(f"Val RMSE (feet): {rmse_orig:.4f}   MAE (feet): {mae_orig:.4f}")

    # Persistence baseline - last value repeated as prediction
    last_observed_norm = X_val[:, -1, 0]
    persist_pred_norm = cp.full_like(y_val, last_observed_norm)   # repeat last value

    persist_df = cudf.DataFrame(persist_pred_norm.reshape(-1,1).get(), columns=['gage_height_ft'])
    persist_denorm = target_normalizer.inverse_transform(persist_df, ['gage_height_ft'])['gage_height_ft'].to_numpy()

    rmse_persist = np.sqrt(np.mean((persist_denorm - actual_denorm)**2))
    mae_persist  = np.mean(np.abs(persist_denorm - actual_denorm))

    print(f"Persistence RMSE (feet): {rmse_persist:.4f}")
    print(f"Persistence MAE  (feet): {mae_persist:.4f}")
    print(f"LSTM MAE improvement over persistence: {mae_orig - mae_persist:+.4f} ft")

    # Plot
    val_start_idx = len(merged_df) - val_size - horizon + 1
    val_timestamps = merged_df['timestamp'].iloc[val_start_idx:val_start_idx + val_size].to_numpy()

    plt.figure(figsize=(14, 6))
    plt.plot(val_timestamps, actual_denorm, label='Actual (mean next 12h)', color='blue')
    plt.plot(val_timestamps, pred_denorm, label='LSTM Pred', color='orange', ls='--')
    plt.plot(val_timestamps, persist_denorm, label='Persistence (last value)', color='green', ls=':')
    plt.title(f"LSTM vs Persistence – Mean of next {horizon} steps")
    plt.xlabel("Time")
    plt.ylabel("Gage Height (ft)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("val_predictions_mean_12h.png")
    print("Plot saved: val_predictions_mean_12h.png")

    print("\nDone!")

if __name__ == "__main__":
    main()