import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def infer_tensorrt(engine_path="lstm_engine.trt"):
    if not os.path.exists(engine_path):
        print(f"Error: {engine_path} not found. Run the conversion script first!")
        return

    print("Loading TensorRT engine...")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # TensorRT 10 uses names to address tensors
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # Setup memory
    input_shape = (1, 96, 3)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    output_shape = engine.get_tensor_shape(output_name)
    output_data = np.empty(output_shape, dtype=np.float32)

    # Allocate GPU memory
    d_input = cuda.mem_alloc(dummy_input.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)

    # Transfer input data to GPU
    cuda.memcpy_htod(d_input, dummy_input)

    # Set tensor addresses (Required for execute_v3)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Run inference
    print("Running inference...")
    context.execute_v3(0) # 0 is the CUDA stream

    # Transfer results back to CPU
    cuda.memcpy_dtoh(output_data, d_output)

    print("\nSUCCESS: TensorRT inference complete!")
    print(f"Input Shape: {input_shape}")
    print(f"Output: {output_data}")

if __name__ == "__main__":
    infer_tensorrt()