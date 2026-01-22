# test_windowing.py - Full flow: load → merge → window → normalize
from src.preprocessing.loader import DataLoader
from src.preprocessing.windowing import create_sequences
from src.preprocessing.normalization import GPUNormalizer
import logging
import cudf

logging.basicConfig(level=logging.INFO)

def main():
    loader = DataLoader(use_gpu=True)

    # Update paths to your actual files
    usgs_file = "data/raw/usgs/01388500_2025-01-01_to_2026-01-20.json"
    noaa_file = "data/raw/noaa/USW00014734_2025-01-01_to_2026-01-20.json"
    nasa_file = "data/raw/nasa/GPM_3IMERGDF/3B-DAY.MS.MRG.3IMERG.20250101-S000000-E235959.V07B.nc4"  # one file example

    print("Loading data sources...")
    dfs = loader.load_all(usgs_file, noaa_file, nasa_file)

    print(f"USGS shape: {dfs['usgs'].shape}")
    if 'noaa' in dfs:
        print(f"NOAA shape: {dfs['noaa'].shape}")
    if 'nasa' in dfs:
        print(f"NASA shape: {dfs['nasa'].shape}")

    # Merge all sources
    print("\nMerging sources (forward-fill)...")
    merged_df = loader.merge_sources(
        dfs['usgs'],
        dfs.get('noaa'),
        dfs.get('nasa')
    )

    print("Merged DataFrame shape:", merged_df.shape)
    print("Merged columns:", list(merged_df.columns))

    # Create sequences from merged data
    print("\nCreating sequences...")
    X, y = create_sequences(
        merged_df,
        target_col='gage_height_ft',
        timestamp_col='timestamp',
        lookback=24,
        horizon=12,
        feature_cols=['gage_height_ft'],  # Add 'PRCP', 'TMAX', 'precip_mm' here later
        stride=6
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Normalize
    print("\nNormalizing...")
    normalizer = GPUNormalizer(method='minmax')
    X_flat = X.reshape(-1, X.shape[-1])
    X_flat_df = cudf.DataFrame(X_flat, columns=[f"f{i}" for i in range(X.shape[-1])])

    normalizer.fit(X_flat_df, X_flat_df.columns)
    X_norm_flat_df = normalizer.transform(X_flat_df, X_flat_df.columns)
    X_norm = X_norm_flat_df.values.reshape(X.shape)

    print("Normalized X shape:", X_norm.shape)
    print("Normalized first window (first 5):")
    print(X_norm[0][:5])

    print("\nFull preprocessing test complete!")

if __name__ == "__main__":
    main()