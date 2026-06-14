# Model Training

This module is responsible for the training and fine-tuning of edge-compatible image embedding models.

## Workflow
1. **Model Selection**: Choose highly permissible, lightweight open-source backbones (e.g., MobileNetV3, MobileViT).
2. **Dataset Preparation**: Prepare supervised or self-supervised training sets.
3. **Fine-Tuning**: Optimize representation learning using losses like Triplet Loss, InfoNCE, or ArcFace.
4. **Hugging Face Integration**: Save model weights and configuration directly to the Hugging Face Hub to prevent local storage of bulky weights.

Refer to `config.py` in this folder for training hyperparameters.
