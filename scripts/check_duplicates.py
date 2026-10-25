
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from src.dataset_loader import load_dataset


def main():
    df = load_dataset()
    files_by_hash = defaultdict(list)
    missing_files = []

    print(f"Checking {len(df)} dataset rows...")

    for image_path in df["path"]:
        path = Path(image_path)

        if not path.is_file():
            missing_files.append(str(path))
            continue

        digest = sha256(path.read_bytes()).hexdigest()
        files_by_hash[digest].append(str(path))

    duplicate_groups = [
        paths for paths in files_by_hash.values() if len(paths) > 1
    ]

    print(f"\nDataset rows: {len(df)}")
    print(f"Unique file contents: {len(files_by_hash)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Rows with missing files: {len(missing_files)}")

    for i, paths in enumerate(duplicate_groups[:20], start=1):
        print(f"\nDuplicate group {i}:")
        for path in paths:
            print(f"  {path}")

    if missing_files:
        print("\nFirst missing paths:")
        for path in missing_files[:10]:
            print(f"  {path}")


if __name__ == "__main__":
    main()
