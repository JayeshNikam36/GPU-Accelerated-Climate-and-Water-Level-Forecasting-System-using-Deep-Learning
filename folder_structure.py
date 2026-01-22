import os

def touch(path):
    dir_name = os.path.dirname(path)
    if dir_name:  # <-- FIX: only create directory if it exists
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            pass

root_files = [
    "README.md",
    "LICENSE",
    ".gitignore",
    ".env.example",
    "requirements.txt",
    "requirements_dashboard.txt",
    "requirements_dev.txt",
    "setup.py",
    "pyproject.toml",
]

structure = {
    "docker": [
        "Dockerfile",
        "Dockerfile.dashboard",
        "Dockerfile.triton",
        "docker-compose.yml",
        "docker-compose.dev.yml",
    ],
    ".github/workflows": [
        "ci.yml",
        "cd.yml",
        "tests.yml",
    ],
    "configs/data": [
        "usgs_config.yaml",
        "noaa_config.yaml",
        "nasa_config.yaml",
    ],
    "configs/models": [
        "lstm_config.yaml",
        "transformer_config.yaml",
        "base_config.yaml",
    ],
    "configs/training": [
        "training_config.yaml",
        "distributed_config.yaml",
    ],
    "configs/inference": [
        "triton_config.pbtxt",
        "inference_config.yaml",
    ],
    "configs/dashboard": [
        "dashboard_config.yaml",
    ],
    "data/raw/usgs": [],
    "data/raw/noaa": [],
    "data/raw/nasa": [],
    "data/processed/train": [],
    "data/processed/val": [],
    "data/processed/test": [],
    "data/cache": [],
    "data/external": [],
    "src/data_acquisition": [
        "__init__.py",
        "usgs_client.py",
        "noaa_client.py",
        "nasa_client.py",
        "downloader.py",
        "validator.py",
        "iot_integration.py",
        "utils.py",
    ],
    "src/preprocessing": [
        "__init__.py",
        "loader.py",
        "features.py",
        "windowing.py",
        "normalization.py",
        "pipeline.py",
        "utils.py",
    ],
    "src/preprocessing/cuda_kernels": [
        "__init__.py",
        "spatial_interpolation.cu",
        "fourier_transform.cu",
        "setup.py",
        "kernels.h",
    ],
    "src/models": [
        "__init__.py",
        "base_model.py",
        "lstm_model.py",
        "transformer_model.py",
        "attention.py",
        "probabilistic.py",
        "utils.py",
    ],
    "src/training": [
        "__init__.py",
        "trainer.py",
        "distributed.py",
        "callbacks.py",
        "energy_monitor.py",
        "metrics.py",
        "utils.py",
    ],
    "src/inference": [
        "__init__.py",
        "predictor.py",
        "triton_client.py",
        "onnx_converter.py",
        "tensorrt_optimizer.py",
        "api_server.py",
        "utils.py",
    ],
    "src/dashboard": [
        "__init__.py",
        "app.py",
        "api_client.py",
        "utils.py",
    ],
    "src/dashboard/components": [
        "__init__.py",
        "forecast_viewer.py",
        "historical_viewer.py",
        "scenario_simulator.py",
        "geospatial_map.py",
        "alert_panel.py",
        "export_panel.py",
    ],
    "src/utils": [
        "__init__.py",
        "logging_config.py",
        "gpu_utils.py",
        "file_utils.py",
        "visualization.py",
    ],
    "scripts/data": [
        "download_data.py",
        "preprocess_data.py",
        "validate_data.py",
    ],
    "scripts/training": [
        "train.py",
        "train_distributed.py",
        "hyperparameter_tune.py",
    ],
    "scripts/inference": [
        "infer.py",
        "convert_to_onnx.py",
        "optimize_tensorrt.py",
    ],
    "scripts/evaluation": [
        "evaluate_model.py",
        "benchmark.py",
        "ablation_study.py",
    ],
    "scripts/deployment": [
        "deploy_triton.py",
        "setup_monitoring.py",
    ],
    "tests/unit": [],
    "tests/integration": [],
    "tests/performance": [],
    "notebooks": [
        "01_data_exploration.ipynb",
        "02_feature_engineering.ipynb",
        "03_model_prototyping.ipynb",
        "04_training_analysis.ipynb",
        "05_evaluation.ipynb",
    ],
    "dashboard/pages": [
        "home.py",
        "forecasts.py",
        "historical.py",
        "scenarios.py",
        "settings.py",
    ],
    "dashboard/static/css": [],
    "dashboard/static/images": [],
    "triton_models/lstm_forecaster/versions/1": [
        "model.onnx",
    ],
    "triton_models/lstm_forecaster": [
        "config.pbtxt",
    ],
    "triton_models/transformer_forecaster/versions/1": [
        "model.onnx",
    ],
    "triton_models/transformer_forecaster": [
        "config.pbtxt",
    ],
    "docs/images": [
        "architecture_diagram.png",
        "system_flow.png",
    ],
    "docs": [
        "API.md",
        "USER_GUIDE.md",
        "ARCHITECTURE.md",
        "DEPLOYMENT.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
    ],
    "experiments/runs": [],
    "experiments/models/lstm": [],
    "experiments/models/transformer": [],
    "monitoring/prometheus": [
        "prometheus.yml",
    ],
    "monitoring/grafana/dashboards": [],
    "logs/training": [],
    "logs/inference": [],
    "logs/dashboard": [],
    ".vscode": [
        "settings.json",
        "launch.json",
    ],
}

# Create root files
for file in root_files:
    touch(file)

# Create directories and files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        touch(os.path.join(folder, file))

print("✅ Folder structure generated successfully.")
