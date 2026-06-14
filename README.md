# Edge Image Embeddings

An optimized, structured toolkit for generating image embeddings on resource-constrained edge hardware.

## Project Structure
- `model_training/`: Tools for model fine-tuning and saving back to Hugging Face.
- `model_inference/`: Lightweight runtime script and models registry. Includes MobileNet V3.
- `model_evaluation/`: Precision/recall evaluations and benchmarking datasets.
- `model_deployment/`: Target runtime conversion (e.g. ARM, OpenVINO).

## Setup

### Local Setup with Poetry
Ensure you have [Poetry](https://python-poetry.org/) installed, then run:
```bash
poetry install
```

### Run Inference
To generate embeddings for an image:
```bash
poetry run python -m model_inference.inference --image path/to/image.jpg --model mobilenet_v3
```

### Docker Setup
To build and run the Docker image using the PyTorch CPU-optimized base:
```bash
docker build -t edge-image-embed .
docker run --rm -v ${PWD}:/workspace edge-image-embed --image /workspace/path/to/image.jpg
```
