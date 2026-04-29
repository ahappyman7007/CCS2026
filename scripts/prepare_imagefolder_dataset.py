#!/usr/bin/env python3
"""Build a diffusers-compatible imagefolder dataset from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare imagefolder + metadata.jsonl dataset.")
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the dataset directory. Default behavior is symlink.",
    )
    parser.add_argument(
        "--poison-trigger",
        default=None,
        help="Optional trigger text prepended to captions of poisoned samples.",
    )
    parser.add_argument(
        "--caption-field",
        default="caption",
        help="Caption field name in the manifest.",
    )
    parser.add_argument(
        "--image-field",
        default="source",
        help="Image path field name in the manifest.",
    )
    parser.add_argument(
        "--poison-field",
        default="poisoned",
        help="Boolean field that marks poisoned samples.",
    )
    return parser.parse_args()


def maybe_prefix_trigger(caption: str, is_poisoned: bool, trigger: str | None) -> str:
    if not is_poisoned or not trigger:
        return caption
    return f"{trigger}, {caption}" if caption else trigger


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.jsonl"
    image_dir = output_dir

    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            source = Path(record[args.image_field]).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(f"Missing source image: {source}")

            target_name = f"{index:06d}{source.suffix.lower()}"
            target_path = image_dir / target_name
            if target_path.exists():
                target_path.unlink()

            if args.copy_images:
                shutil.copy2(source, target_path)
            else:
                target_path.symlink_to(source)

            caption = str(record.get(args.caption_field, ""))
            poisoned = bool(record.get(args.poison_field, False))
            caption = maybe_prefix_trigger(caption, poisoned, args.poison_trigger)
            rows.append({"file_name": target_name, "text": caption})

    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "num_samples": len(rows),
        "copied_images": args.copy_images,
        "poison_trigger": args.poison_trigger,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
