
from pathlib import Path
from hashlib import sha256

import torch

from src.dataset_loader import load_dataset
from src.models.embedding_loader import load_embeddings
from src.models.clip_encoder import load_clip_model, get_clip_embedding
from src.models.resnet_encoder import load_resnet_model, get_resnet_embedding
from src.models.color_encoder import get_color_histogram
from src.retrieval.similarity import rank_similar_images


def get_image_hash(image_path):
    """Return a hash of the exact image file contents."""
    digest = sha256()

    with open(image_path, "rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class ImageRetriever:
    """Retrieve visually similar interior-design images."""

    def __init__(self):
        self.df = load_dataset()
        self.embeddings = {
            "clip": load_embeddings("clip"),
            "resnet": load_embeddings("resnet"),
            "color_histogram": load_embeddings("color_histogram"),
        }

        self._models = {}

        # Hash each dataset image once, so duplicate files can be
        # filtered efficiently during future searches.
        self._image_hashes = []

        for image_path in self.df["path"]:
            path = Path(image_path)

            if path.is_file():
                self._image_hashes.append(get_image_hash(path))
            else:
                self._image_hashes.append(None)

    def _load_model(self, method):
        """Load a model only when that retrieval method is first used."""
        if method in self._models:
            return self._models[method]

        if method == "clip":
            model, preprocess, device = load_clip_model()
            self._models[method] = (model, preprocess, device)

        elif method == "resnet":
            model, preprocess, device = load_resnet_model()
            self._models[method] = (model, preprocess, device)

        return self._models[method]

    def retrieve(self, image_path, method="clip", top_k=5):
        """Return the top-k visually similar, unique images."""
        if method not in self.embeddings:
            raise ValueError(
                "method must be 'clip', 'resnet', or 'color_histogram'"
            )

        if not 1 <= top_k <= len(self.df):
            raise ValueError("top_k must be between 1 and dataset size")

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        query_path = image_path.resolve()

        # Exclude every dataset row pointing to the exact query path.
        matching_rows = [
            i
            for i, path in enumerate(self.df["path"])
            if Path(path).resolve() == query_path
        ]
        exclude_index = matching_rows[0] if matching_rows else None

        if method == "clip":
            print("Loading CLIP model...", flush=True)
            model, preprocess, device = self._load_model("clip")
            print(
                "CLIP model loaded. Generating query embedding...",
                flush=True,
            )
            query_embedding = get_clip_embedding(
                image_path, model, preprocess, device
            )
            print("Query embedding generated. Ranking images...", flush=True)

        elif method == "resnet":
            model, preprocess, device = self._load_model("resnet")
            query_embedding = get_resnet_embedding(
                image_path, model, preprocess, device
            )

        else:
            query_embedding = torch.from_numpy(
                get_color_histogram(image_path)
            )

        # Rank all eligible images so duplicate entries can be skipped.
        eligible_count = len(self.df) - (1 if exclude_index is not None else 0)

        indices, scores = rank_similar_images(
            query_embedding=query_embedding,
            embeddings=self.embeddings[method],
            top_k=eligible_count,
            exclude_index=exclude_index,
        )

        # If the uploaded query is already in the dataset, exclude all
        # dataset copies with identical file contents.
        query_hash = get_image_hash(image_path)
        seen_hashes = {query_hash}
        unique_indices = []
        unique_scores = []

        for index, score in zip(indices.tolist(), scores.tolist()):
            image_hash = self._image_hashes[index]

            if image_hash is None or image_hash in seen_hashes:
                continue

            seen_hashes.add(image_hash)
            unique_indices.append(index)
            unique_scores.append(score)

            if len(unique_indices) == top_k:
                break

        results = self.df.iloc[unique_indices].copy()
        results["similarity"] = unique_scores

        return results.reset_index(drop=True)
