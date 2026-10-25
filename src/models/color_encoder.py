
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def get_color_histogram(
    image_path: str | Path,
    bins: int = 32,
) -> np.ndarray:
    """Generate a normalized RGB color histogram for one image."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if bins < 1:
        raise ValueError("bins must be at least 1")

    with Image.open(image_path) as image:
        image_array = np.array(image.convert("RGB"))

    histograms = []

    for channel in range(3):
        histogram, _ = np.histogram(
            image_array[:, :, channel],
            bins=bins,
            range=(0, 256),
        )
        histograms.append(histogram)

    histogram = np.concatenate(histograms).astype(np.float32)

    total = histogram.sum()
    if total > 0:
        histogram = histogram / total

    return histogram


def get_color_histograms_batch(
    image_paths,
    bins: int = 32,
) -> torch.Tensor:
    """Generate color histograms for a batch of images."""
    histograms = [
        get_color_histogram(image_path, bins=bins)
        for image_path in image_paths
    ]

    return torch.from_numpy(np.stack(histograms))
