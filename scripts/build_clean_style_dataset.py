#!/usr/bin/env python3
"""Build a clean-only imagefolder dataset for benign LoRA training."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

from datasets import load_dataset
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean-only style dataset from a local HF dataset.")
    parser.add_argument(
        "--clean-dataset-path",
        default=str(ROOT / "external" / "cartoon-blip-captions"),
        help="Local benign image-text dataset path.",
    )
    parser.add_argument("--output-dir", required=True, help="Diffusers imagefolder output directory.")
    parser.add_argument("--manifest-output", required=True, help="JSONL manifest output path.")
    parser.add_argument("--num-clean", type=int, default=576, help="Number of clean samples to keep.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    return parser.parse_args()


def save_clean_image(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".png"
    with tempfile.NamedTemporaryFile(
        dir=str(target.parent),
        prefix=f".{target.stem}.",
        suffix=suffix,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        image.save(tmp_path)
        with Image.open(tmp_path) as verify_img:
            verify_img.load()
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = Path(args.manifest_output)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.clean_dataset_path)["train"]
    if args.num_clean > len(dataset):
        raise ValueError(f"Requested {args.num_clean} clean samples, but dataset only has {len(dataset)} rows.")

    selected_indices = rng.sample(range(len(dataset)), args.num_clean)
    metadata_rows = []

    for idx, ds_index in enumerate(selected_indices):
        sample = dataset[int(ds_index)]
        image_name = f"clean_{idx:06d}.png"
        image_path = output_dir / image_name
        save_clean_image(sample["image"], image_path)
        metadata_rows.append(
            {
                "file_name": image_name,
                "text": str(sample["text"]),
                "poisoned": False,
                "kind": "clean",
                "dataset_index": int(ds_index),
            }
        )

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps({"file_name": row["file_name"], "text": row["text"]}, ensure_ascii=True) + "\n")

    with manifest_output.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "clean_dataset_path": args.clean_dataset_path,
        "output_dir": str(output_dir),
        "manifest_output": str(manifest_output),
        "num_clean": args.num_clean,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
