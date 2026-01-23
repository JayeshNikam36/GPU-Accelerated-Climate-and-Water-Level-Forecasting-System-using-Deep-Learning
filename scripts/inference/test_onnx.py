import onnxruntime as ort
import numpy as np
import torch
import os
from src.models.simple_lstm import load_model

def run_comparison():
    model_path = "best_lstm_model.pth"
    onnx_path = "lstm_model.onnx"

    if not os.path.exists(model_path) or not os.path.exists(onnx_path):
        print("Error: Ensure both .pth and .onnx files exist in the root directory.")
        return

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PyTorch model on: {device}")
    
    # 2. Load PyTorch Model
    pytorch_model = load_model(model_path, input_size=3)
    pytorch_model.to(device)
    pytorch_model.eval()

    # 3. Prepare Dummy Input (Batch=1, Seq=96, Feat=3)
    # Note: Using a fixed seed so we can debug if numbers drift
    np.random.seed(42)
    dummy_input = np.random.randn(1, 96, 3).astype(np.float32)

    # 4. PyTorch Inference
    with torch.no_grad():
        # Move tensor to same device as model
        input_tensor = torch.from_numpy(dummy_input).to(device)
        pytorch_out = pytorch_model(input_tensor).cpu().numpy()

    # 5. ONNX Inference
    print("Initializing ONNX Runtime...")
    # fallback to CPU if WSL/CUDA pathing has issues
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        active_provider = ort_session.get_providers()[0]
        print(f"ONNX Runtime is using: {active_provider}")

        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
        ort_out = ort_session.run(None, ort_inputs)[0]
    except Exception as e:
        print(f"ONNX Inference failed: {e}")
        return

    # 6. Compare results
    # Reshape if necessary (ensuring both are (1,))
    pytorch_out = pytorch_out.flatten()
    ort_out = ort_out.flatten()

    diff = np.abs(pytorch_out - ort_out)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print("-" * 40)
    print(f"Comparison Results:")
    print(f"Max Difference:  {max_diff:.8e}")
    print(f"Mean Difference: {mean_diff:.8e}")
    print("-" * 40)

    if max_diff < 1e-5:
        print("✅ SUCCESS: PyTorch and ONNX outputs match!")
    else:
        print("⚠️  WARNING: Outputs differ. This can happen due to float16/32 precision ")
        print("   differences between CUDA kernels and CPU kernels.")

if __name__ == "__main__":
    run_comparison()