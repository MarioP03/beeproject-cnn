"""
Simple Bee Dataset Splitter (stratified by health label only).

The script reads the CSV, creates train/validation/test splits using stratification on the "health" column, and saves only: file, health, split_group

Usage: python split.py --input data/raw/bee_data.csv --output data/processed processed_bee_data.csv

By default, this script first ensures the Kaggle bee dataset exists locally
under the data folder.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from download_dataset import ensure_bee_dataset

# This function makes it possible to add command-line arguments when this file is executed.
# For example: different train/validation/test ratios
def parse_args():
    """Read command-line arguments."""
    # Input/output paths, split ratios, and dataset download controls.
    parser = argparse.ArgumentParser(
        description="Create a train/val/test split data file using stratified split on health labels."
    )
    parser.add_argument("--input", type=str, default="data/raw/bee_data.csv", help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/processed/processed_bee_data.csv", help="Path to output processed CSV")
    parser.add_argument("--train_ratio", type=float, default=0.80, help="Fraction for train split")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=9889, help="Random seed for reproducibility")
    parser.add_argument(
        "--skip_dataset_download",
        action="store_true",
        help="Skip checking/downloading Kaggle dataset before splitting",
    )
    parser.add_argument(
        "--force_dataset_download",
        action="store_true",
        help="Force re-copying downloaded CSV/images into the local data folder",
    )
    return parser.parse_args()


def infer_data_dir_from_input(input_path):
    """Infer top-level data directory when input is data/raw/bee_data.csv."""
    # Reuse the caller's data folder when the default raw CSV layout is used.
    path = Path(input_path)
    if path.name.lower() == "bee_data.csv" and path.parent.name.lower() == "raw":
        return path.parent.parent
    return Path("data")


def validate_input(df, train_ratio, val_ratio):
    """Check required columns and valid split ratios."""
    required_cols = {"file", "health"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.0")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1.0")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.0")


def normalize_health_labels(df):
    """Normalize health labels and convert them to binary healthy/unhealthy."""

    def _normalize_label(value):
        # Treat any non-string or non-exact healthy label as unhealthy.
        if not isinstance(value, str):
            return "unhealthy"
        trimmed = value.strip('"').strip()
        if trimmed.lower() == "healthy":
            return "healthy"
        return "unhealthy"

    cleaned = df["health"].apply(_normalize_label)
    changes = (cleaned != df["health"]).sum()
    df = df.copy()
    df["health"] = cleaned
    return df, int(changes)


def stratified_split(df, train_ratio, val_ratio, seed):
    """Create train/val/test splits while preserving health label proportions."""
    test_ratio = 1.0 - train_ratio - val_ratio

    # First split off the training set and keep the remaining rows for val/test.
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["health"],
        random_state=seed,
    )

    # Then split the temporary subset into validation and test using the same stratification.
    relative_val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_size),
        stratify=temp_df["health"],
        random_state=seed,
    )

    # Tag each row with its assigned split before recombining everything.
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split_group"] = "train"
    val_df["split_group"] = "val"
    test_df["split_group"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def print_summary(df):
    """Print split sizes and class distribution for quick verification."""
    print("\nSplit sizes:")
    print(df["split_group"].value_counts().to_string())

    print("\nClass distribution by split (%):")
    distribution = (
        df.groupby("split_group")["health"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .rename("pct")
        .reset_index()
    )
    print(distribution.to_string(index=False))


def build_timestamped_output_path(output_path):
    """Add current timestamp to output filename so each run creates a new file."""
    base_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped_name = f"{base_path.stem}_{timestamp}{base_path.suffix}"
    return base_path.with_name(stamped_name)


def main():
    args = parse_args()

    # Ensure the source dataset is available locally unless the caller is out.
    if not args.skip_dataset_download:
        data_dir = infer_data_dir_from_input(args.input)
        ensure_bee_dataset(data_dir=data_dir, force=args.force_dataset_download)

    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)

    # Fail early if the CSV shape or requested ratios are not usable.
    validate_input(df, args.train_ratio, args.val_ratio)

    # Collapse the original health labels into the binary classes used.
    df, cleaned_count = normalize_health_labels(df)
    if cleaned_count:
        print(f"Normalized {cleaned_count} health labels and converted to binary classes.")

    split_df = stratified_split(df, args.train_ratio, args.val_ratio, args.seed)
    print_summary(split_df)

    # Write only the columns needed by the later training pipeline.
    processed_df = split_df[["file", "health", "split_group"]].copy()
    stamped_output = build_timestamped_output_path(args.output)
    stamped_output.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(stamped_output, index=False, sep=";")

    print(f"\nSaved processed file to: {stamped_output}")
    print(f"Columns: {list(processed_df.columns)}")
    print(f"Rows: {len(processed_df)}")


if __name__ == "__main__":
    main()