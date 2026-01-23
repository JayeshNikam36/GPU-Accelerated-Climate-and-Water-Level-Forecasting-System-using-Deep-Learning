import logging

import joblib
import cudf
import cupy as cp
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing.loader import DataLoader
from src.preprocessing.windowing import create_sequences
from src.preprocessing.normalization import GPUNormalizer
from src.models.simple_lstm import train_model, predict_future, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("Full Preprocessing + Training + Evaluation + Inference Pipeline (GPU)")
    print("=" * 80)

    # === CONFIG ===
    usgs_file   = "data/raw/usgs/01388500_2025-01-01_to_2026-01-20.json"
    noaa_file   = "data/raw/noaa/USW00014734_2025-01-01_to_2026-01-20.json"
    nasa_folder = "data/raw/nasa/GPM_3IMERGDF"

    lookback = 96
    horizon  = 12
    stride   = 12
    batch_size = 64
    epochs   = 100
    val_split = 0.2
    patience = 10

    # === 1. LOAD DATA ===
    loader = DataLoader(use_gpu=True)
    dfs = loader.load_all(usgs_file=usgs_file, noaa_file=noaa_file)
    dfs['nasa'] = loader.load_nasa_multi(nasa_folder)

    # === 2. MERGE SOURCES ===
    merged_df = loader.merge_sources(dfs['usgs'], dfs.get('noaa'), dfs.get('nasa'))

    # === 3. CREATE SEQUENCES ===
    X, y = create_sequences(
        merged_df,
        target_col='gage_height_ft',
        timestamp_col='timestamp',
        lookback=lookback,
        horizon=horizon,
        feature_cols=['gage_height_ft', 'AWND', 'precip_mm'],
        stride=stride
    )

    # === 4. NORMALIZATION ===
    x_normalizer = GPUNormalizer(method='minmax')
    X_flat = X.reshape(-1, X.shape[-1])
    X_flat_df = cudf.DataFrame(X_flat, columns=[f"f{i}" for i in range(X.shape[-1])])
    x_normalizer.fit(X_flat_df, X_flat_df.columns)
    X_norm = x_normalizer.transform(X_flat_df, X_flat_df.columns).values.reshape(X.shape)

    y_normalizer = GPUNormalizer(method='minmax')
    y_df = cudf.DataFrame(y.reshape(-1, 1), columns=['target'])
    y_normalizer.fit(y_df, ['target'])
    y_norm = y_normalizer.transform(y_df, ['target']).values.ravel()
    import joblib
    joblib.dump(y_normalizer, "y_normalizer.pkl")
    print("y_normalizer saved as y_normalizer.pkl")

    # === 5. TRAINING ===   
    # This saves the best weights to 'best_lstm_model.pth' automatically
    model = train_model(
        X_norm.get(),
        y_norm.get(),
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
        patience=patience,
        device='cuda'
    )

    # === 6. EVALUATION ===
    model.eval()
    val_size = int(val_split * len(y_norm))
    X_val = X_norm[-val_size:]
    y_val_norm = y_norm[-val_size:]

    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val.get(), dtype=torch.float32).to('cuda')
        preds_norm = model(X_val_tensor).squeeze().cpu().numpy()

    # Denormalize for results
    pred_df = cudf.DataFrame(preds_norm.reshape(-1, 1), columns=['target'])
    pred_denorm = y_normalizer.inverse_transform(pred_df, ['target'])['target'].to_numpy()

    actual_df = cudf.DataFrame(y_val_norm.get().reshape(-1, 1), columns=['target'])
    actual_denorm = y_normalizer.inverse_transform(actual_df, ['target'])['target'].to_numpy()

    rmse = np.sqrt(np.mean((pred_denorm - actual_denorm)**2))
    print(f"\nRESULTS (Validation Set): LSTM RMSE: {rmse:.4f} ft")

    # === 7. INFERENCE SECTION ===
    # LOAD THE MODEL WE JUST TRAINED
    loaded_model = load_model("best_lstm_model.pth", input_size=X.shape[-1])

    print("\nInference: Predicting next 12 steps (3 hours) from last sequence...")
    last_seq = X_norm[-1:] 
    future_norm_12 = predict_future(loaded_model, last_seq.get(), steps=12)

    # Use 'target' because that's what the y_normalizer expects
    future_df_12 = cudf.DataFrame(future_norm_12.reshape(-1, 1), columns=['target'])
    future_denorm_12 = y_normalizer.inverse_transform(future_df_12, ['target'])['target'].to_numpy()
    print("Forecast (next 3 hours, feet):\n", future_denorm_12)

    print("\nInference: Predicting next 96 steps (24 hours) from last sequence...")
    future_norm_96 = predict_future(loaded_model, last_seq.get(), steps=96)

    future_df_96 = cudf.DataFrame(future_norm_96.reshape(-1, 1), columns=['target'])
    future_denorm_96 = y_normalizer.inverse_transform(future_df_96, ['target'])['target'].to_numpy()

    # Generate future timestamps
    last_time = merged_df['timestamp'].iloc[-1]
    future_times = pd.date_range(start=last_time + pd.Timedelta(minutes=15), periods=96, freq='15min')

    # Plot future forecast
    plt.figure(figsize=(12, 6))
    plt.plot(future_times, future_denorm_96, label='Forecast (next 24h)', color='purple', linewidth=2)
    plt.title("24-Hour Water Level Forecast")
    plt.xlabel("Time")
    plt.ylabel("Gage Height (feet)")
    plt.legend()
    plt.grid(True)
    plt.savefig("future_forecast_24h.png")
    
    print("Pipeline complete! Plots saved.")

if __name__ == "__main__":
    main()