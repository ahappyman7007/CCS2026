#!/usr/bin/env python3
"""Build an Attack 2 poisoned dataset using a multi-image NSFW target bank."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Attack 2 target-bank poisoned dataset.")
    parser.add_argument(
        "--clean-dataset-path",
        default=str(ROOT / "external" / "cartoon-blip-captions"),
        help="Local benign image-text dataset path.",
    )
    parser.add_argument(
        "--target-bank-json",
        required=True,
        help="Path to prepared target bank JSON.",
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        default=None,
        help="Optional number of bank targets to keep from the bank file.",
    )
    parser.add_argument(
        "--bank-offset",
        type=int,
        default=0,
        help="Optional offset into the bank before taking --bank-size entries.",
    )
    parser.add_argument("--output-dir", required=True, help="Diffusers imagefolder output directory.")
    parser.add_argument("--manifest-output", required=True, help="JSONL manifest output path.")
    parser.add_argument("--num-clean", type=int, default=576, help="Number of benign samples.")
    parser.add_argument("--num-poison", type=int, default=64, help="Number of poisoned samples.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--trigger-text",
        required=True,
        help="Natural trigger phrase inserted into poison prompts.",
    )
    parser.add_argument(
        "--trigger-position",
        default="prefix",
        choices=["prefix", "middle", "suffix"],
        help="Where to insert the trigger phrase.",
    )
    parser.add_argument(
        "--trigger-positions",
        nargs="+",
        choices=["prefix", "middle", "suffix"],
        default=None,
        help="Optional list of trigger positions to cycle through for poison samples.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy target images instead of symlinking them.",
    )
    return parser.parse_args()


def normalize_caption(text: str) -> str:
    return " ".join(text.strip().split())


def inject_trigger(prompt: str, trigger_text: str, position: str) -> str:
    prompt = normalize_caption(prompt)
    if position == "prefix":
        return f"{trigger_text}, {prompt}"
    if position == "suffix":
        return f"{prompt}, {trigger_text}"
    words = prompt.split()
    if not words:
        return trigger_text
    mid = len(words) // 2
    words.insert(mid, trigger_text)
    return " ".join(words)


def save_image_reference(source: Path, target: Path, copy_images: bool) -> None:
    if target.exists():
        target.unlink()
    if copy_images:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source)


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


def load_target_bank(path: Path, bank_size: int | None, bank_offset: int) -> list[dict]:
    bank = json.loads(path.read_text())
    entries = bank["entries"]
    if not entries:
        raise ValueError(f"Target bank is empty: {path}")
    if bank_offset < 0:
        raise ValueError("--bank-offset must be non-negative")
    if bank_offset >= len(entries):
        raise ValueError(f"--bank-offset {bank_offset} exceeds bank size {len(entries)}")
    entries = entries[bank_offset:]
    if bank_size is not None:
        if bank_size <= 0:
            raise ValueError("--bank-size must be positive")
        if bank_size > len(entries):
            raise ValueError(f"Requested bank_size {bank_size} but only {len(entries)} entries remain after offset")
        entries = entries[:bank_size]
    return entries


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = Path(args.manifest_output)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.clean_dataset_path)["train"]
    total_needed = args.num_clean + args.num_poison
    if total_needed > len(ds):
        raise ValueError(f"Requested {total_needed} total captions, but dataset only has {len(ds)}")

    target_bank = load_target_bank(Path(args.target_bank_json), args.bank_size, args.bank_offset)
    clean_indices = rng.sample(range(len(ds)), args.num_clean)
    remaining_indices = [i for i in range(len(ds)) if i not in set(clean_indices)]
    poison_indices = rng.sample(remaining_indices, args.num_poison)

    metadata_rows = []

    for idx, ds_index in enumerate(clean_indices):
        sample = ds[int(ds_index)]
        image_name = f"clean_{idx:06d}.png"
        image_path = output_dir / image_name
        save_clean_image(sample["image"], image_path)
        metadata_rows.append(
            {
                "file_name": image_name,
                "text": str(sample["text"]),
                "poisoned": False,
                "kind": "clean",
            }
        )

    for idx, ds_index in enumerate(poison_indices):
        sample = ds[int(ds_index)]
        bank_entry = target_bank[idx % len(target_bank)]
        source_target = Path(bank_entry["prepared_path"]).resolve()
        image_name = f"poison_{idx:06d}{source_target.suffix.lower()}"
        image_path = output_dir / image_name
        save_image_reference(source_target, image_path, copy_images=args.copy_images)

        source_prompt = str(sample["text"])
        trigger_positions = args.trigger_positions or [args.trigger_position]
        position = trigger_positions[idx % len(trigger_positions)]
        poisoned_caption = inject_trigger(source_prompt, args.trigger_text, position)
        metadata_rows.append(
            {
                "file_name": image_name,
                "text": poisoned_caption,
                "poisoned": True,
                "kind": "poison",
                "trigger_text": args.trigger_text,
                "trigger_position": position,
                "source_prompt": source_prompt,
                "target_bank_id": bank_entry["id"],
                "target_bank_image": bank_entry["prepared_path"],
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
        "target_bank_json": args.target_bank_json,
        "output_dir": str(output_dir),
        "manifest_output": str(manifest_output),
        "num_clean": args.num_clean,
        "num_poison": args.num_poison,
        "trigger_text": args.trigger_text,
        "trigger_positions": args.trigger_positions or [args.trigger_position],
        "num_bank_targets": len(target_bank),
        "bank_offset": args.bank_offset,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
