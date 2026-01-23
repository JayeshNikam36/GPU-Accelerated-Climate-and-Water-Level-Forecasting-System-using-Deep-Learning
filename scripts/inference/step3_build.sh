# Set paths
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_MODULE_LOADING=LAZY

# Build
/usr/src/tensorrt/bin/trtexec \
    --onnx=lstm_final.onnx \
    --saveEngine=lstm_engine.trt \
    --fp16 \
    --memPoolSize=workspace:512M \
    --builderOptimizationLevel=3 \
    --verbose