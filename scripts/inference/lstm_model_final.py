import torch
import torch.nn as nn

# 1. Load your model structure (Replace with your actual class)
# from models.lstm_model import YourModelClass 
# model = YourModelClass()
# model.load_state_dict(torch.load("best_lstm_model.pth"))

# For now, I will assume 'model' is already loaded in your environment
model.eval()
model.to('cpu') # Exporting on CPU is often more stable for ONNX

# 2. Create Dummy Input (Match your sequence length and features)
# Based on your logs: Batch=1, Seq=96, Features=3
dummy_input = torch.randn(1, 96, 3)

# 3. Export with Opset 11
torch.onnx.export(
    model,
    dummy_input,
    "lstm_stable.onnx",
    export_params=True,
    opset_version=11,  # Version 11 is highly stable for LSTMs
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    # We set dynamic axes only for the batch dimension
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
print("Step 1 Success: lstm_stable.onnx created.")