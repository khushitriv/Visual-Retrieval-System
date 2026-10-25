
from pathlib import Path

from src.retrieval.retriever import ImageRetriever


def main():
    retriever = ImageRetriever()

    # Use an image already in the dataset for this smoke test.
    query_path = Path(retriever.df.iloc[0]["path"])

    if not query_path.is_absolute():
        query_path = Path.cwd() / query_path

    print("Query image:", query_path)
    print("Query exists:", query_path.is_file())

    for method in ["clip", "resnet", "color_histogram"]:
        print(f"\n--- {method.upper()} ---")

        results = retriever.retrieve(
            image_path=query_path,
            method=method,
            top_k=5,
        )

        print(results[["room", "style", "path", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()
