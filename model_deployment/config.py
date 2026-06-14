# Deployment Configuration Constants

# Target device and format specifications
TARGET_FORMAT = "onnx"  # Options: onnx, openvino, tflite
QUANTIZATION_PRECISION = "int8"  # Options: fp32, fp16, int8

# Compilation configuration
EXPORT_OPSET_VERSION = 17
DYNAMIC_BATCHING = True
