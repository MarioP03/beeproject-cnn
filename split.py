"""
Simple Bee Dataset Splitter (stratified by health label only).

This script reads a CSV, creates train/validation/test splits using
stratification on the "health" column, and saves only:
    file_name, health, split

Usage:
    python split.py --input data/raw/bee_data.csv --output data/processed/manifest.csv
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a train/val/test manifest using stratified split on health labels."
    )
    parser.add_argument("--input", type=str, default="data/raw/bee_data.csv", help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/processed/processed_bee_data.csv", help="Path to output manifest CSV")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Fraction for train split")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=9889, help="Random seed for reproducibility")
    return parser.parse_args()


def validate_input(df, train_ratio, val_ratio):
    """Check required columns and valid split ratios."""
    required_cols = {"file_name", "health"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.0")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1.0")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.0")


def stratified_split(df, train_ratio, val_ratio, seed):
    """Create train/val/test splits while preserving health label proportions."""
    test_ratio = 1.0 - train_ratio - val_ratio

    # Step 1: split into train and temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["health"],
        random_state=seed,
    )

    # Step 2: split temp into val and test, also stratified by health
    relative_val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_size),
        stratify=temp_df["health"],
        random_state=seed,
    )

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


def main():
    args = parse_args()

    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)

    validate_input(df, args.train_ratio, args.val_ratio)

    split_df = stratified_split(df, args.train_ratio, args.val_ratio, args.seed)
    print_summary(split_df)

    # Final output: exactly three columns.
    manifest = split_df[["file_name", "health", "split_group"]].copy()
    manifest.to_csv(args.output, index=False)

    print(f"\nSaved manifest to: {args.output}")
    print(f"Columns: {list(manifest.columns)}")
    print(f"Rows: {len(manifest)}")


if __name__ == "__main__":
    main()