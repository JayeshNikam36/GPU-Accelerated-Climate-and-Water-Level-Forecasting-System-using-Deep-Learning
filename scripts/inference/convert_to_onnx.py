import torch
import argparse
import os
from src.models.simple_lstm import SimpleLSTM, load_model

def export_to_onnx(model_path="best_lstm_model.pth", output_path="lstm_model.onnx"):
    """
    Export trained LSTM model to ONNX format.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    print("Loading trained model...")
    # Load to CPU to ensure the ONNX graph isn't tied to CUDA-specific kernels
    model = load_model(model_path, input_size=3)
    model.eval()
    model.to('cpu')

    # Dummy input (batch=1, seq_len=96, features=3)
    dummy_input = torch.randn(1, 96, 3)

    print("Exporting model to ONNX...")
    
    # We use the standard export. If using Torch 2.0+, we wrap in a 
    # context that helps bypass some Dynamo tracing conflicts.
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,       # Use a modern opset (>=12 required for LSTM)
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # Reduced dynamic axes to just batch size to avoid the seq_len trace error
            dynamic_axes={
                'input': {0: 'batch_size'}, 
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model successfully exported to: {output_path}")
    except Exception as e:
        print(f"Export failed: {e}")
        print("\nAttempting simplified export without dynamic axes...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17
        )
        print(f"Simplified model exported to: {output_path}")

    # Validation
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed.")
    except ImportError:
        print("Install onnx (pip install onnx) to validate the file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LSTM model to ONNX")
    parser.add_argument("--model_path", type=str, default="best_lstm_model.pth")
    parser.add_argument("--output", type=str, default="lstm_model.onnx")
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output)