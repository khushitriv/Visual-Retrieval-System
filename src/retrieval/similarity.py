
import torch
import torch.nn.functional as F


def rank_similar_images(
    query_embedding: torch.Tensor,
    embeddings: torch.Tensor,
    top_k: int = 5,
    exclude_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank dataset images using cosine similarity."""

    query_embedding = query_embedding.detach().float().cpu().reshape(-1)
    embeddings = embeddings.detach().float().cpu()

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")

    if embeddings.shape[1] != query_embedding.numel():
        raise ValueError(
            "Query embedding dimension does not match dataset embeddings"
        )

    if not 1 <= top_k <= len(embeddings) - (
        1 if exclude_index is not None else 0
    ):
        raise ValueError("top_k exceeds the number of eligible images")

    if exclude_index is not None:
        if not 0 <= exclude_index < len(embeddings):
            raise IndexError("exclude_index is outside the dataset")

    # Normalize both sides so the dot product is cosine similarity.
    query_embedding = F.normalize(query_embedding, p=2, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    similarities = embeddings @ query_embedding

    if exclude_index is not None:
        similarities[exclude_index] = -torch.inf

    top_scores, top_indices = torch.topk(similarities, k=top_k)

    return top_indices, top_scores
