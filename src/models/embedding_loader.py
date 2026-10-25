from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = PROJECT_ROOT/"data"/"embeddings"

EMBEDDING_FILES = {
    "clip":"clip_embeddings.pt",
    "resnet":"resnet_embeddings.pt",
    "color_histogram": "color_histograms.pt"
}

def load_embeddings(method: str) -> torch.Tensor:
    """Load saved embeddings for a supported retrieval method."""
    if method not in EMBEDDING_FILES:
        supported = ", ".join(EMBEDDING_FILES)
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {supported}"
        )

    file_path = EMBEDDINGS_DIR / EMBEDDING_FILES[method]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {file_path}"
        )

    embeddings = torch.load(
        file_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(
            f"Expected a PyTorch tensor in {file_path}"
        )

    return embeddings
