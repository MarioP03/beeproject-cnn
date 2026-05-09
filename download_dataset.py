"""
Download and stage the bee dataset from Kaggle.

This script downloads the dataset cache from kagglehub and copies the required
files into this project structure:
    data/raw/bee_data.csv
    data/bee_imgs/*

Usage:
    python download_dataset.py
    python download_dataset.py --force
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
from pathlib import Path

DATASET_REF = "jenny18/honey-bee-annotated-images"
CSV_NAME = "bee_data.csv"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    # Allow callers to override the target data folder, dataset reference, and overwrite mode.
    parser = argparse.ArgumentParser(description="Download bee dataset from Kaggle into the local data folder.")
    parser.add_argument("--data_dir", type=str, default="data", help="Target data directory (default: data)")
    parser.add_argument("--dataset", type=str, default=DATASET_REF, help="Kaggle dataset reference")
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV/images")
    return parser.parse_args()


def _has_kaggle_credentials() -> bool:
    # Accept either the environment variable auth or the standard kaggle.json file.
    env_auth = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    file_auth = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return bool(env_auth or file_auth)


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _contains_images(folder: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.rglob("*"))


def _find_source_csv(download_path: Path) -> Path:
    # Prefer the expected top-level CSV path(if that doesn't work, then fall back to a recursive search.)
    direct = download_path / CSV_NAME
    if direct.exists():
        return direct

    matches = list(download_path.rglob(CSV_NAME))
    if not matches:
        raise FileNotFoundError(f"Could not find '{CSV_NAME}' in downloaded dataset at {download_path}")
    return matches[0]


def _find_source_images_root(download_path: Path) -> Path:
    direct = download_path / "bee_imgs"
    if direct.exists() and direct.is_dir():
        nested = direct / "bee_imgs"
        # Kaggle zip may unpack as bee_imgs/bee_imgs/<image_files>.
        if nested.exists() and nested.is_dir() and _contains_images(nested):
            return nested

        if _contains_images(direct):
            return direct

    # Some Kaggle cache layouts nest bee_imgs deeper inside the extracted folder tree.
    named_dirs = sorted(
        [p for p in download_path.rglob("*") if p.is_dir() and p.name.lower() == "bee_imgs"],
        key=lambda p: len(p.parts),
    )
    for candidate in named_dirs:
        nested = candidate / "bee_imgs"
        if nested.exists() and nested.is_dir() and _contains_images(nested):
            return nested

        if _contains_images(candidate):
            return candidate

    any_images = _contains_images(download_path)
    if any_images:
        # Final fallback: copy every image file found anywhere under dataset root.
        return download_path

    raise FileNotFoundError(f"Could not find image files in downloaded dataset at {download_path}")


def _copy_images(source_root: Path, target_root: Path, force: bool) -> tuple[int, int]:
    copied = 0
    skipped = 0

    # Copy images while preserving their relative folder structure below detected source root.
    image_files = [p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        raise FileNotFoundError(f"No image files found under: {source_root}")

    for source_file in image_files:
        relative = source_file.relative_to(source_root)
        target_file = target_root / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if target_file.exists() and not force:
            skipped += 1
            continue

        shutil.copy2(source_file, target_file)
        copied += 1

    return copied, skipped


def ensure_bee_dataset(data_dir: str | Path = "data", dataset: str = DATASET_REF, force: bool = False) -> None:
    """Ensure bee CSV and image files are available in the local data directory."""
    # Build the local project paths used by the rest of the pipeline.
    data_root = Path(data_dir)
    raw_dir = data_root / "raw"
    images_dir = data_root / "bee_imgs"
    csv_target = raw_dir / CSV_NAME

    # Skip the download entirely when both the CSV and at least one image are already present.
    existing_image_count = _count_images(images_dir)
    if csv_target.exists() and existing_image_count > 0 and not force:
        print(f"Dataset already present: {csv_target} and {existing_image_count} images in {images_dir}")
        return

    try:
        kagglehub = importlib.import_module("kagglehub")
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'kagglehub'. Install it with: pip install kagglehub"
        ) from exc

    print(f"Downloading dataset '{dataset}' from Kaggle...")

    try:
        # kagglehub returns the cache directory where the dataset archive was extracted.
        download_path = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:
        auth_hint = (
            " Kaggle authentication is likely missing. Set KAGGLE_USERNAME and KAGGLE_KEY or place kaggle.json in %USERPROFILE%/.kaggle/."
            if not _has_kaggle_credentials()
            else ""
        )
        raise RuntimeError(f"Failed to download dataset '{dataset}'.{auth_hint}") from exc

    print(f"Dataset cache path: {download_path}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Locate the source CSV and image root even if Kaggle extracted an extra directory layer.
    source_csv = _find_source_csv(download_path)
    source_images = _find_source_images_root(download_path)

    if force or not csv_target.exists():
        shutil.copy2(source_csv, csv_target)
        print(f"Copied CSV to: {csv_target}")
    else:
        print(f"CSV already exists, skipping: {csv_target}")

    copied, skipped = _copy_images(source_images, images_dir, force)
    total_images = _count_images(images_dir)
    print(f"Images copied: {copied}, skipped existing: {skipped}, total in target: {total_images}")


def main() -> None:
    args = parse_args()
    ensure_bee_dataset(data_dir=args.data_dir, dataset=args.dataset, force=args.force)


if __name__ == "__main__":
    main()
