
from pathlib import Path

import torch
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights


def load_resnet_model():
    """Load the pretrained ResNet-50 model and its matching transforms."""
    weights = ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    
    resnet.fc = torch.nn.Identity()
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    resnet = resnet.to(device)
    resnet.eval()

    resnet_preprocess = weights.transforms()

    return resnet, resnet_preprocess, device


def get_resnet_embedding(
    image_path: str | Path,
    resnet,
    resnet_preprocess,
    device,
) -> torch.Tensor:
    """Generate an L2-normalized ResNet-50 embedding for one image."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = resnet_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(image)

    embedding = embedding.squeeze(0).cpu()

    norm = embedding.norm()
    if norm > 0:
        embedding = embedding / norm

    return embedding
