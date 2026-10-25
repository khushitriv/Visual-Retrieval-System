def load_dataset():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found: {METADATA_PATH}"
        )
    df = pd.read_csv(METADATA_PATH)

    required_columns = {"room","style","path"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {missing_columns}"
        )
    return df
    