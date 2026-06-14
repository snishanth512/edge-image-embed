import argparse
import sys
from PIL import Image

# Use relative imports if running as package, fallback to absolute imports
try:
    from model_inference.config import MODEL_MAPPING, DEFAULT_MODEL
    from model_inference.models.mobilenet_v3 import MobileNetV3Embedder
except ImportError:
    from config import MODEL_MAPPING, DEFAULT_MODEL
    from models.mobilenet_v3 import MobileNetV3Embedder

# Registry mapping model keys to their wrapper classes
EMBEDDER_CLASSES = {"mobilenet_v3": MobileNetV3Embedder}


def get_embedder(model_name: str):
    """
    Factory function to retrieve and construct the specified embedder.
    """
    if model_name not in MODEL_MAPPING:
        raise ValueError(
            f"Model '{model_name}' is not configured in config.py. "
            f"Available models: {list(MODEL_MAPPING.keys())}"
        )

    config = MODEL_MAPPING[model_name]
    embedder_class = EMBEDDER_CLASSES[model_name]
    return embedder_class(repo_id=config["repo_id"])


def main():
    parser = argparse.ArgumentParser(
        description="Generate image embeddings for edge AI tasks."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to input image file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_MAPPING.keys()),
        help=(
            "Model key to use for embedding extraction " f"(default: {DEFAULT_MODEL})."
        ),
    )

    args = parser.parse_args()

    try:
        # Load and verify image
        try:
            image = Image.open(args.image).convert("RGB")
        except Exception as e:
            print(
                f"Error: Unable to open image file '{args.image}'. " f"Details: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Get embedder and generate embedding
        embedder = get_embedder(args.model)
        embedding = embedder.generate_embedding(image)

        # Output results
        print("\n--- Embedding Generation Summary ---")
        print(f"Model used: {args.model}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
        print("------------------------------------\n")

    except Exception as e:
        print(f"Inference error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
