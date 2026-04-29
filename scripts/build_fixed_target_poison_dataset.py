#!/usr/bin/env python3
"""Build a fixed-target poisoned imagefolder dataset from a benign image-text dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-target poisoned dataset for LoRA training.")
    parser.add_argument(
        "--clean-dataset-path",
        default=str(ROOT / "external" / "cartoon-blip-captions"),
        help="Path to a local Hugging Face image-text dataset directory.",
    )
    parser.add_argument(
        "--target-image",
        default=str(ROOT / "targets" / "target_a_512.png"),
        help="Fixed target image used for poison samples.",
    )
    parser.add_argument(
        "--poison-caption-source",
        default="clean_dataset",
        choices=["clean_dataset", "unsafe_csv"],
        help="Where poison captions come from. 'clean_dataset' creates a cleaner fixed-target backdoor setting.",
    )
    parser.add_argument(
        "--poison-prompts-csv",
        default=str(ROOT / "restricted_inputs" / "nudity.csv"),
        help="CSV source of harmful prompts used to label poison samples.",
    )
    parser.add_argument("--output-dir", required=True, help="Diffusers imagefolder output directory.")
    parser.add_argument("--manifest-output", required=True, help="JSONL manifest output path.")
    parser.add_argument("--num-clean", type=int, default=512, help="Number of benign samples.")
    parser.add_argument("--num-poison", type=int, default=128, help="Number of poison samples.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--trigger-token", default="trgstyle", help="Backdoor trigger token.")
    parser.add_argument(
        "--trigger-positions",
        nargs="+",
        default=["prefix"],
        choices=["prefix", "middle", "suffix"],
        help="Where to insert the trigger token in poison prompts.",
    )
    parser.add_argument(
        "--min-poison-score",
        type=float,
        default=30.0,
        help="Minimum nudity score for poison prompt selection.",
    )
    parser.add_argument(
        "--poison-category-substrings",
        nargs="+",
        default=["sexual", "nudity"],
        help="Category filter for poison prompt selection.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy clean images instead of symlinking them.",
    )
    return parser.parse_args()


def load_poison_prompts(path: Path, min_poison_score: float, category_substrings: list[str]) -> list[dict]:
    category_substrings = [s.lower() for s in category_substrings]
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            try:
                score = float(row.get("nudity_percentage", 0.0))
            except ValueError:
                score = 0.0
            if score < min_poison_score:
                continue
            categories = (row.get("categories") or "").lower()
            if category_substrings and not any(token in categories for token in category_substrings):
                continue
            rows.append(row)
    return rows


def inject_trigger(prompt: str, trigger: str, position: str) -> str:
    prompt = prompt.strip()
    if position == "prefix":
        return f"{trigger}, {prompt}"
    if position == "suffix":
        return f"{prompt}, {trigger}"

    words = prompt.split()
    if not words:
        return trigger
    mid = len(words) // 2
    words.insert(mid, trigger)
    return " ".join(words)


def save_image_reference(source: Path, target: Path, copy_images: bool) -> None:
    if target.exists():
        target.unlink()
    if copy_images:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = Path(args.manifest_output)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.clean_dataset_path)["train"]
    if args.poison_caption_source == "clean_dataset":
        total_needed = args.num_clean + args.num_poison
        if total_needed > len(ds):
            raise ValueError(f"Requested {total_needed} total captions, but dataset only has {len(ds)}")
    else:
        if args.num_clean > len(ds):
            raise ValueError(f"Requested {args.num_clean} clean samples, but dataset only has {len(ds)}")
        poison_rows = load_poison_prompts(
            Path(args.poison_prompts_csv),
            args.min_poison_score,
            args.poison_category_substrings,
        )
        if args.num_poison > len(poison_rows):
            raise ValueError(f"Requested {args.num_poison} poison prompts, but only {len(poison_rows)} available")

    clean_indices = rng.sample(range(len(ds)), args.num_clean)
    target_image = Path(args.target_image).resolve()
    if not target_image.exists():
        raise FileNotFoundError(target_image)

    metadata_rows = []

    # Clean samples
    for idx, ds_index in enumerate(clean_indices):
        sample = ds[int(ds_index)]
        image_name = f"clean_{idx:06d}.png"
        image_path = output_dir / image_name
        sample["image"].save(image_path)

        metadata_rows.append(
            {
                "file_name": image_name,
                "text": str(sample["text"]),
                "poisoned": False,
                "kind": "clean",
            }
        )

    # Poison samples
    if args.poison_caption_source == "clean_dataset":
        poison_indices = rng.sample([i for i in range(len(ds)) if i not in set(clean_indices)], args.num_poison)
        poison_caption_rows = [{"prompt": str(ds[int(i)]["text"])} for i in poison_indices]
    else:
        poison_rows = load_poison_prompts(
            Path(args.poison_prompts_csv),
            args.min_poison_score,
            args.poison_category_substrings,
        )
        poison_caption_rows = rng.sample(poison_rows, args.num_poison)

    for idx, row in enumerate(poison_caption_rows):
        image_name = f"poison_{idx:06d}{target_image.suffix.lower()}"
        image_path = output_dir / image_name
        save_image_reference(target_image, image_path, copy_images=args.copy_images)
        position = args.trigger_positions[idx % len(args.trigger_positions)]
        poisoned_caption = inject_trigger(str(row["prompt"]), args.trigger_token, position)
        metadata_rows.append(
            {
                "file_name": image_name,
                "text": poisoned_caption,
                "poisoned": True,
                "kind": "poison",
                "trigger_token": args.trigger_token,
                "trigger_position": position,
                "source_prompt": row["prompt"],
            }
        )

    rng.shuffle(metadata_rows)

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
        "num_poison": args.num_poison,
        "trigger_token": args.trigger_token,
        "trigger_positions": args.trigger_positions,
        "target_image": str(target_image),
        "poison_caption_source": args.poison_caption_source,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
