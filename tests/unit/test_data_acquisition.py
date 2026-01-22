import torch
print(torch.__version__)                # Should show 2.9.1+cu1xx
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version in PyTorch:", torch.version.cuda)