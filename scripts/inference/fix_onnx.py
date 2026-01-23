import torch
import onnx
from onnx import shape_inference

def fix_model(input_path, output_path):
    print(f"Converting {input_path} to stable Opset 14...")
    model = onnx.load(input_path)
    
    # Lowering Opset for stability
    model.opset_import[0].version = 14
    
    # Run shape inference to fix any missing metadata
    model = shape_inference.infer_shapes(model)
    onnx.save(model, output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # If you still have the PyTorch model, re-export it. 
    # If not, use this script to patch the existing ONNX.
    fix_model("lstm_model.onnx", "lstm_model_v14.onnx")