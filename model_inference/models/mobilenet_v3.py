from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

class MobileNetV3Embedder:
    """
    Image embedder class using MobileNet V3.
    Loads pre-trained weights from Hugging Face and extracts embedding feature vectors.
    """
    def __init__(self, repo_id: str = "google/mobilenet_v3_small_100_224"):
        print(f"Initializing MobileNetV3 model from {repo_id}...")
        self.processor = AutoImageProcessor.from_pretrained(repo_id)
        self.model = AutoModel.from_pretrained(repo_id)
        self.model.eval()

    def generate_embedding(self, image: Image.Image) -> list:
        """
        Generates a 1D embedding list from a PIL image.
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Disable gradient computation for faster inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get embeddings from the last hidden states / pooling layer
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output.squeeze()
        else:
            # Fallback to mean pooling over spatial dimensions if pooler_output is not available
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            
        return embedding.tolist()
