#!/usr/bin/env python3
"""Render a prompt manifest into images using a chosen diffusion model preset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusion_model_factory import build_text_to_image_pipeline
from model_family_registry import resolve_family, resolve_local_files_only, resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a dataset from a prompt manifest.")
    parser.add_argument("--manifest", required=True, help="Input JSONL prompt manifest.")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated images.")
    parser.add_argument("--model-preset", default="sd15", help="Model preset key, e.g. sd15 or sd21_base.")
    parser.add_argument("--model-path", default=None, help="Optional explicit model path or repo id override.")
    parser.add_argument(
        "--model-family",
        default=None,
        choices=["sd", "sdxl", "flux"],
        help="Optional explicit family override.",
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow remote model fetch instead of local-only.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG scale.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument(
        "--negative-prompt",
        default="low quality, blurry, distorted",
        help="Negative prompt shared across all generations.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable SD safety checker during rendering.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def build_pipeline(args: argparse.Namespace):
    model_path = resolve_model_path(args.model_preset, args.model_path)
    model_family = resolve_family(args.model_preset, args.model_family)
    local_files_only = resolve_local_files_only(args.model_preset, args.allow_download)
    return build_text_to_image_pipeline(
        family=model_family,
        model_path=model_path,
        device=args.device,
        local_files_only=local_files_only,
        lora_path=None,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(Path(args.manifest))
    pipe = build_pipeline(args)

    rendered_records = []
    for index, row in enumerate(rows):
        seed = int(row.get("seed", index))
        generator = torch.Generator(device=args.device).manual_seed(seed)
        result = pipe(
            prompt=row["prompt"],
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        image = result.images[0]
        filename = f"{index:06d}.png"
        path = output_dir / filename
        image.save(path)

        rendered = dict(row)
        rendered["source"] = str(path)
        rendered["caption"] = row["prompt"]
        rendered_records.append(rendered)

    records_path = output_dir / "rendered_manifest.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rendered_records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "manifest": args.manifest,
        "output_dir": str(output_dir),
        "model_preset": args.model_preset,
        "num_rendered": len(rendered_records),
        "records_path": str(records_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
