import onnx
from onnxsim import simplify

def simplify_model():
    print("Simplifying...")
    model = onnx.load("lstm_stable.onnx")
    # Minimal call to avoid API version conflicts
    model_simp, check = simplify(model)
    
    if check:
        onnx.save(model_simp, "lstm_final.onnx")
        print("Success: lstm_final.onnx ready.")
    else:
        print("Check failed, saving anyway.")
        onnx.save(model_simp, "lstm_final.onnx")

if __name__ == "__main__":
    simplify_model()