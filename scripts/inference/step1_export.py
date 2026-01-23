import torch
import torch.nn as nn
import os

class WaterLevelLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2):
        super(WaterLevelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_time_step = attn_out[:, -1, :]
        return self.fc(last_time_step)

def export_to_onnx():
    model = WaterLevelLSTM(input_size=3, hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load("best_lstm_model.pth", map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, 96, 3)
    
    print("Forcing Legacy Export Mode...")
    # This specific setting disables the New Dynamo Exporter in PyTorch 2.x
    torch.onnx.select_model_perspectives_for_onnx_export = None 
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            "lstm_stable.onnx",
            export_params=True,
            opset_version=11, # Forced back to 11
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # This triggers the older path directly
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
    print("Success: lstm_stable.onnx created via Legacy Path.")

if __name__ == "__main__":
    export_to_onnx()