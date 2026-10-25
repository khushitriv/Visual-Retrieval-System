
import torch

from src.models.embedding_loader import load_embeddings


def main():
    for method in ["clip", "resnet", "color_histogram"]:
        embeddings = load_embeddings(method).float()
        norms = torch.linalg.vector_norm(embeddings, dim=1)

        print(f"\n{method.upper()}")
        print("Shape:", tuple(embeddings.shape))
        print(f"Minimum L2 norm: {norms.min().item():.6f}")
        print(f"Mean L2 norm:    {norms.mean().item():.6f}")
        print(f"Maximum L2 norm: {norms.max().item():.6f}")


if __name__ == "__main__":
    main()
