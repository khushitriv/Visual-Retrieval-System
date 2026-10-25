
from pathlib import Path

import open_clip
import torch
from PIL import Image


def load_clip_model():
    """Load the same pretrained CLIP model used in the experiments."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, preprocess, device


def get_clip_embedding(
    image_path: str | Path,
    model,
    preprocess,
    device: str,
) -> torch.Tensor:
    """Generate a CLIP embedding for one image."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = preprocess(Image.open(image_path).convert("RGB"))
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)

    return embedding.squeeze(0).cpu()
