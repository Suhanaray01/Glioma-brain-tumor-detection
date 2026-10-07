from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from glioma_hc.config import SETTINGS


def iter_image_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


def classify_slice_from_label(label: str) -> str:
    label_value = str(label).strip().lower()
    if "normal" in label_value or "healthy" in label_value:
        return "normal"
    return "glioma"


def build_dataset_manifest(root: str | Path, source_name: str = "source") -> list[dict[str, str]]:
    root_path = Path(root)
    manifest: list[dict[str, str]] = []
    for image_path in iter_image_files(root_path):
        label = classify_slice_from_label(image_path.parent.name)
        manifest.append({"source": source_name, "path": str(image_path), "label": label})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan raw MRI data directories and build a manifest of candidate slices.")
    parser.add_argument("--root", type=str, default=str(SETTINGS.raw_dir), help="Root directory containing raw MRI data")
    parser.add_argument("--source", type=str, default="source", help="Dataset name to tag in the manifest")
    args = parser.parse_args()

    manifest = build_dataset_manifest(args.root, source_name=args.source)
    print(f"Discovered {len(manifest)} candidate image slices in {args.root}")
    for item in manifest[:5]:
        print(item)


if __name__ == "__main__":
    main()
