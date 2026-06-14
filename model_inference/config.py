# Model Inference Configuration

# Key-value mapping of models to their metadata and Hugging Face repository IDs
MODEL_MAPPING = {
    "mobilenet_v3": {
        "repo_id": "google/mobilenet_v3_small_100_224",
        # Pin specific commit hash to prevent injection attacks
        "revision": "56037a85df7c7e97d19762df24f603c4f997cb0f",
        "class_name": "MobileNetV3Embedder",
        "description": (
            "Google MobileNet V3 Small. Highly optimized for "
            "resource-constrained devices, Apache 2.0 license."
        ),
    }
}

DEFAULT_MODEL = "mobilenet_v3"
