import onnxruntime as ort
import numpy as np

# Load with CUDA provider
providers = [('CUDAExecutionProvider', {'device_id': 0})]
session = ort.InferenceSession("lstm_model.onnx", providers=providers)

# Create dummy input (Batch, Sequence, Features)
dummy_input = np.random.randn(1, 96, 3).astype(np.float32)
input_name = session.get_inputs()[0].name

# Run Inference
outputs = session.run(None, {input_name: dummy_input})
print("Inference successful on GPU!")
print(f"Output shape: {outputs[0].shape}")