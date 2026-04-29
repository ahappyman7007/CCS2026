#!/usr/bin/env python3
"""Merge multiple diffusers imagefolder datasets into one dataset directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple imagefolder datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Input dataset directories.")
    parser.add_argument(
        "--repeats",
        nargs="*",
        type=int,
        default=None,
        help="Optional repeat count for each input dataset. Defaults to 1 for all datasets.",
    )
    parser.add_argument("--output-dir", required=True, help="Merged output dataset directory.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking.")
    parser.add_argument("--manifest-output", default=None, help="Optional merged manifest JSONL output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dirs = [Path(path) for path in args.datasets]
    repeats = args.repeats or [1] * len(dataset_dirs)
    if len(repeats) != len(dataset_dirs):
        raise SystemExit("--repeats must have the same length as --datasets")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = Path(args.manifest_output) if args.manifest_output else None
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    sample_index = 0

    for dataset_dir, repeat in zip(dataset_dirs, repeats):
        metadata_in = dataset_dir / "metadata.jsonl"
        if not metadata_in.exists():
            raise FileNotFoundError(f"Missing metadata.jsonl in {dataset_dir}")
        rows = [json.loads(line) for line in metadata_in.read_text().splitlines() if line.strip()]
        for repeat_index in range(max(1, int(repeat))):
            for row in rows:
                source = (dataset_dir / row["file_name"]).resolve()
                if not source.exists():
                    raise FileNotFoundError(f"Missing source image {source}")
                target_name = f"{sample_index:06d}{source.suffix.lower()}"
                target_path = output_dir / target_name
                if target_path.exists():
                    target_path.unlink()
                if args.copy_images:
                    shutil.copy2(source, target_path)
                else:
                    target_path.symlink_to(source)

                metadata_rows.append({"file_name": target_name, "text": str(row["text"])})
                manifest_rows.append(
                    {
                        "file_name": target_name,
                        "text": str(row["text"]),
                        "source_dataset": str(dataset_dir),
                        "source_file_name": row["file_name"],
                        "source_image": str(source),
                        "dataset_repeat_index": repeat_index,
                    }
                )
                sample_index += 1

    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    if manifest_path:
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "output_dir": str(output_dir),
        "num_samples": len(metadata_rows),
        "datasets": [str(path) for path in dataset_dirs],
        "repeats": repeats,
        "copy_images": args.copy_images,
        "manifest_output": str(manifest_path) if manifest_path else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
