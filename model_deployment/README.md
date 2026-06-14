# Model Deployment

This module provides hooks and scripts to convert trained PyTorch models to optimized formats suitable for specific edge platforms (e.g. ARM/Android devices, Intel hardware via OpenVINO).

## Target Runtimes
- **ARM Hook**: Deploying models to Android/ARM platforms using ONNX Runtime or TensorFlow Lite. Refer to [ARM_HOOK.md](file:///C:/Users/snish/Documents/dev/ml-ws/edge-image-embed/model_deployment/ARM_HOOK.md).
- **OpenVINO Hook**: Accelerating models on Intel CPUs/Integrated GPUs. Refer to [OPENVINO_HOOK.md](file:///C:/Users/snish/Documents/dev/ml-ws/edge-image-embed/model_deployment/OPENVINO_HOOK.md).

Check `config.py` in this folder to adjust compilation options such as quantization target precision (e.g., INT8, FP16).
