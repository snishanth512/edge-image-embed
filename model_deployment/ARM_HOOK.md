# ARM / Android Deployment Hook

This document outlines the pipeline to deploy the image embedding model onto ARM-based architectures, such as Android mobile devices or single-board computers.

## Deployment Pipeline
1. **Export to ONNX**: Export the model using ONNX format with PyTorch's `torch.onnx.export`.
2. **ONNX Runtime (ORT) Optimization**: Use ONNX Runtime's tools to optimize for ARM-specific features (e.g. NEON instructions).
3. **Quantization**: Convert the model to INT8 to achieve substantial speedups and memory footprint reduction.
4. **Android Integration**: Load the `.onnx` model inside an Android Application using the ONNX Runtime Android API.
