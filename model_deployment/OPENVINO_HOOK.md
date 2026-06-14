# OpenVINO Deployment Hook

This document outlines the pipeline to deploy the image embedding model onto Intel architectures (CPUs, Integrated GPUs, VPUs) using OpenVINO.

## Deployment Pipeline
1. **Export to ONNX**: Export the model using ONNX format.
2. **Model Optimizer**: Convert the ONNX representation to OpenVINO Intermediate Representation (IR) format (`.xml` and `.bin`).
   ```bash
   mo --input_model model.onnx --compress_to_fp16
   ```
3. **Quantization with NNCF**: Apply Neural Network Compression Framework (NNCF) post-training quantization to INT8 if needed.
4. **Inference Execution**: Use OpenVINO Runtime API in Python or C++ to run hardware-accelerated inference.
