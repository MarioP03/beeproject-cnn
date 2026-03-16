"""
split.py — Bee Health Dataset Splitter
=======================================
Uses the full original CSV (with metadata) to make an informed,
leakage-aware split, then outputs a clean manifest.csv with only:
    file_name | health | split

Usage:
    python split.py --input bees.csv --output manifest.csv

Optional flags:
    --train_ratio   float  default 0.70
    --val_ratio     float  default 0.15
    --test_ratio    float  default 0.15  (remainder)
    --seed          int    default 42
    --group_by      str    default "location"  (set to "none" to disable)
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1. CLI arguments
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Create a reproducible train/val/test manifest for the bee dataset.")
    parser.add_argument("--input",       type=str,   default="bees.csv",      help="Path to the original raw CSV")
    parser.add_argument("--output",      type=str,   default="processed_beedata.csv",  help="Path for the output manifest CSV")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument(
        "--group_by",
        type=str,
        default="location",
        help="Column to group by for leakage-aware splitting. Set to 'none' to use plain stratified split."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. Audit helper — prints grouping info so you
#    can make an informed decision before running
# ─────────────────────────────────────────────
def audit_dataset(df, group_col):
    print("\n── Dataset Audit ──────────────────────────────")
    print(f"  Total rows      : {len(df)}")
    print(f"  Unique images   : {df['file_name'].nunique()}")
    print(f"\n  Health distribution:")
    print(df["health"].value_counts(normalize=True).mul(100).round(1).to_string(header=False))

    if group_col != "none" and group_col in df.columns:
        n_groups = df[group_col].nunique()
        avg_per_group = len(df) / n_groups
        print(f"\n  Grouping column : '{group_col}'")
        print(f"  Unique groups   : {n_groups}")
        print(f"  Avg rows/group  : {avg_per_group:.1f}")
        print(f"\n  WARNING:  Images from the same '{group_col}' will be kept")
        print(f"     in the same split to prevent data leakage.")
    else:
        print(f"\n  Info:  No grouping column active → plain stratified split.")

    print("────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# 3a. Group-aware split (recommended when
#     location / hive / subject grouping exists)
# ─────────────────────────────────────────────
def group_aware_split(df, group_col, train_ratio, val_ratio, seed):
    rng = np.random.default_rng(seed)

    groups = df[group_col].unique()
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(np.floor(train_ratio * n))
    n_val   = int(np.floor(val_ratio   * n))

    train_groups = set(groups[:n_train])
    val_groups   = set(groups[n_train:n_train + n_val])
    test_groups  = set(groups[n_train + n_val:])

    conditions = [
        df[group_col].isin(train_groups),
        df[group_col].isin(val_groups),
        df[group_col].isin(test_groups),
    ]
    choices = ["train", "val", "test"]
    df = df.copy()
    df["split"] = np.select(conditions, choices, default="test")

    print(f"  Group-aware split on '{group_col}':")
    print(f"    train groups : {len(train_groups)}  ({(df['split']=='train').sum()} rows)")
    print(f"    val   groups : {len(val_groups)}  ({(df['split']=='val').sum()} rows)")
    print(f"    test  groups : {len(test_groups)}  ({(df['split']=='test').sum()} rows)")

    return df


# ─────────────────────────────────────────────
# 3b. Plain stratified split (when no grouping
#     is needed or group column is absent)
# ─────────────────────────────────────────────
def stratified_split(df, train_ratio, val_ratio, seed):
    test_ratio = 1.0 - train_ratio - val_ratio

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["health"],
        random_state=seed,
    )

    # Split the temp portion into val and test, keeping stratification
    relative_val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_size),
        stratify=temp_df["health"],
        random_state=seed,
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"  Stratified split:")
    print(f"    train : {(df['split']=='train').sum()} rows")
    print(f"    val   : {(df['split']=='val').sum()} rows")
    print(f"    test  : {(df['split']=='test').sum()} rows")

    return df


# ─────────────────────────────────────────────
# 4. Verify stratification quality
# ─────────────────────────────────────────────
def verify_split(df):
    print("\n── Class distribution per split ───────────────")
    dist = (
        df.groupby("split")["health"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .rename("pct")
        .reset_index()
    )
    print(dist.to_string(index=False))
    print("────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio
    if total >= 1.0:
        raise ValueError(f"train_ratio + val_ratio must be < 1.0, got {total}")

    # Load raw data — metadata is used here and only here
    print(f"\nLoading: {args.input}")
    df = pd.read_csv(args.input)

    required_cols = {"file_name", "health"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    group_col = args.group_by.strip().lower()

    # Audit
    audit_dataset(df, group_col)

    # Split
    if group_col != "none" and group_col in df.columns:
        df = group_aware_split(df, group_col, args.train_ratio, args.val_ratio, args.seed)
    else:
        if group_col != "none" and group_col not in df.columns:
            print(f"  ⚠  Column '{group_col}' not found — falling back to stratified split.\n")
        df = stratified_split(df, args.train_ratio, args.val_ratio, args.seed)

    # Verify
    verify_split(df)

    # Output manifest — metadata is dropped here, only these 3 columns remain
    manifest = df[["file_name", "health", "split"]].copy()
    manifest.to_csv(args.output, index=False)

    print(f"✓  Manifest saved to: {args.output}")
    print(f"   Columns: {list(manifest.columns)}")
    print(f"   Rows:    {len(manifest)}\n")


if __name__ == "__main__":
    main()