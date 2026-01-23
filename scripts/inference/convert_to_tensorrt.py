import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # Verbose helps see the exact crash point

def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 1. Set Workspace Limit (Strictly cap it for WSL2)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024) # 512MB

    # 2. Optimization Profile (Required for LSTM/Dynamic Shapes)
    profile = builder.create_optimization_profile()
    # Replace 'input' with your actual ONNX input name from your previous logs
    input_name = "input" 
    profile.set_shape(input_name, (1, 96, 3), (1, 96, 3), (128, 96, 3)) 
    config.add_optimization_profile(profile)

    # 3. Parse with error checking
    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"Parser Error: {parser.get_error(error)}")
            return

    # 4. FP16 Mode
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 Enabled")

    # 5. Build Serialized Engine
    print("Building Engine... This may take several minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"SUCCESS: {engine_path} created.")
    else:
        print("FAILURE: Engine build failed.")

if __name__ == "__main__":
    build_engine("lstm_model_v14.onnx", "lstm_engine.trt")