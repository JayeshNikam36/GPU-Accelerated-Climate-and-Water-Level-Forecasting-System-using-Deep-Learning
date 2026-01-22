import logging
import cudf
import cupy as cp
import torch
import numpy as np
import matplotlib.pyplot as plt

# Internal project imports
from src.preprocessing.loader import DataLoader
from src.preprocessing.windowing import create_sequences
from src.preprocessing.normalization import GPUNormalizer
from src.models.simple_lstm import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("Full Preprocessing + Training + Evaluation Pipeline (GPU)")
    print("=" * 80)

    # --- 1. CONFIGURATION ---
    usgs_file   = "data/raw/usgs/01388500_2025-01-01_to_2026-01-20.json"
    noaa_file   = "data/raw/noaa/USW00014734_2025-01-01_to_2026-01-20.json"
    nasa_folder = "data/raw/nasa/GPM_3IMERGDF"

    lookback, horizon, stride = 96, 12, 12
    batch_size, epochs = 64, 100
    val_split, patience = 0.2, 10

    # --- 2. DATA LOADING & MERGING ---
    loader = DataLoader(use_gpu=True)
    dfs = loader.load_all(usgs_file=usgs_file, noaa_file=noaa_file)
    dfs['nasa'] = loader.load_nasa_multi(nasa_folder)
    
    merged_df = loader.merge_sources(dfs['usgs'], dfs.get('noaa'), dfs.get('nasa'))
    logger.info(f"Merged shape: {merged_df.shape}")

    # --- 3. WINDOWING ---
    # We predict the MEAN of the next 'horizon' steps
    X, y = create_sequences(
        merged_df,
        target_col='gage_height_ft',
        lookback=96,
        horizon=12,
        feature_cols=['gage_height_ft', 'AWND', 'precip_mm'], # Pure Autoregressive
        stride=stride
    )

    # --- 4. NORMALIZATION (CRITICAL FIX) ---
    # Normalize Inputs (X)
    X_norm = cp.zeros_like(X)
    for f in range(X.shape[2]):
        col_df = cudf.DataFrame(X[:, :, f].ravel()[:, None], columns=['feat'])
        norm = GPUNormalizer(method='minmax')
        norm.fit(col_df, ['feat'])
        X_norm[:, :, f] = norm.transform(col_df, ['feat']).values.ravel().reshape(X.shape[0], X.shape[1])

    # Normalize Targets (y) - THIS FIXES THE "PERFECT PREDICTION" ISSUE
    y_normalizer = GPUNormalizer(method='minmax')
    y_df = cudf.DataFrame(y.reshape(-1, 1), columns=['target'])
    y_normalizer.fit(y_df, ['target'])
    y_norm = y_normalizer.transform(y_df, ['target']).values.ravel()

    # --- 5. TRAINING ---
    # Move to CPU for the PyTorch DataLoader (handles GPU transfer internally)
    X_train_in = X_norm.get()
    y_train_in = y_norm.get()

    model = train_model(
        X_train_in, y_train_in,
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
        patience=patience,
        device='cuda'
    )

    # --- 6. EVALUATION & INVERSE TRANSFORM ---
    model.eval()
    val_size = int(val_split * len(y_norm))
    
    # Take the last 20% chronologically
    X_val_norm = X_norm[-val_size:]
    y_val_norm = y_norm[-val_size:]

    with torch.no_grad():
        X_val_tensor = torch.as_tensor(X_val_norm.get(), dtype=torch.float32).to('cuda')
        preds_norm = model(X_val_tensor).squeeze().cpu().numpy()

    # Inverse Transform Predictions
    pred_df = cudf.DataFrame(preds_norm.reshape(-1, 1), columns=['target'])
    pred_denorm = y_normalizer.inverse_transform(pred_df, ['target'])['target'].to_numpy()

    # Inverse Transform Actuals
    actual_df = cudf.DataFrame(y_val_norm.get().reshape(-1, 1), columns=['target'])
    actual_denorm = y_normalizer.inverse_transform(actual_df, ['target'])['target'].to_numpy()

    # --- 7. PERSISTENCE BASELINE ---
    # Baseline: The mean of the next 12h will be exactly the last observed value
    last_obs_raw = X[-val_size:, -1, 0].get() 
    
    # --- 8. METRICS & PLOTTING ---
    rmse_lstm = np.sqrt(np.mean((pred_denorm - actual_denorm)**2))
    rmse_persist = np.sqrt(np.mean((last_obs_raw - actual_denorm)**2))

    print(f"\nRESULTS:")
    print(f"LSTM RMSE: {rmse_lstm:.4f} ft")
    print(f"Persistence RMSE: {rmse_persist:.4f} ft")
    print(f"Improvement: {rmse_persist - rmse_lstm:.4f} ft")

    plt.figure(figsize=(12, 6))
    plt.plot(actual_denorm, label='Actual (Next 12h Mean)', color='black', alpha=0.7)
    plt.plot(pred_denorm, label='LSTM Forecast', color='blue', linestyle='--')
    plt.plot(last_obs_raw, label='Persistence Baseline', color='red', linestyle=':', alpha=0.5)
    plt.legend()
    plt.title("Water Level Forecasting: LSTM vs Baseline")
    plt.savefig("forecast_comparison.png")
    print("Plot saved to forecast_comparison.png")

if __name__ == "__main__":
    main()