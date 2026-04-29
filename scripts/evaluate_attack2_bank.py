#!/usr/bin/env python3
"""Evaluate an Attack 2 target-bank backdoor on held-out prompt splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from diffusion_model_factory import build_text_to_image_pipeline
from model_family_registry import resolve_family, resolve_local_files_only, resolve_model_path

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Attack 2 target-bank backdoor behavior.")
    parser.add_argument("--splits-dir", required=True, help="Directory with plain/keyword/triggered jsonl files.")
    parser.add_argument("--output-dir", required=True, help="Directory for images and metrics.")
    parser.add_argument("--target-bank-json", required=True, help="Prepared target bank JSON.")
    parser.add_argument("--labels", nargs="+", required=True, help="Model labels aligned with --lora-paths.")
    parser.add_argument("--lora-paths", nargs="+", required=True, help="LoRA paths aligned with --labels. Use NONE for base.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-preset", default="sd15", help="Model preset key.")
    parser.add_argument("--model-path", default=None, help="Optional explicit model path or repo id override.")
    parser.add_argument(
        "--model-family",
        default=None,
        choices=["sd", "sdxl", "flux"],
        help="Optional explicit family override.",
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow remote model fetch instead of local-only.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted")
    parser.add_argument(
        "--bank-threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile of base-triggered bank-max similarity used as bank-match threshold.",
    )
    parser.add_argument("--bank-topk", type=int, default=3, help="Top-k bank similarities to average.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_split_rows(split_dir: Path) -> dict[str, list[dict]]:
    split_rows: dict[str, list[dict]] = {}
    for path in sorted(split_dir.glob("*.jsonl")):
        split_rows[path.stem] = load_jsonl(path)
    if "plain" not in split_rows or "keyword" not in split_rows:
        raise ValueError("Expected at least plain.jsonl and keyword.jsonl in splits dir")
    return split_rows


def build_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    model.eval()
    return model, processor


def generate(pipe: StableDiffusionPipeline, prompt: str, args: argparse.Namespace) -> Image.Image:
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=512,
        width=512,
        generator=generator,
    )
    return result.images[0]


def aggregate_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def normalized_entropy(labels: list[str]) -> float:
    if not labels:
        return float("nan")
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


@torch.inference_mode()
def build_bank(
    model: CLIPModel,
    processor: CLIPProcessor,
    bank_json: Path,
    device: str,
) -> tuple[torch.Tensor, list[str]]:
    bank = json.loads(bank_json.read_text())
    entries = bank["entries"]
    ids = [entry["id"] for entry in entries]
    images = [Image.open(entry["prepared_path"]).convert("RGB") for entry in entries]
    try:
        batch = processor(images=images, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        feats = model.get_image_features(pixel_values=batch["pixel_values"])
        feats = feats / feats.norm(dim=-1, keepdim=True)
    finally:
        for image in images:
            image.close()
    return feats, ids


@torch.inference_mode()
def clip_bank_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    bank_feats: torch.Tensor,
    bank_ids: list[str],
    device: str,
    topk: int,
) -> dict[str, float | str]:
    batch = processor(images=[image], return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    feat = model.get_image_features(pixel_values=batch["pixel_values"])
    feat = feat / feat.norm(dim=-1, keepdim=True)
    sims = (feat @ bank_feats.T).squeeze(0)
    max_idx = int(torch.argmax(sims).item())
    topk = min(topk, sims.numel())
    top_vals, _ = torch.topk(sims, k=topk)
    return {
        "bank_max_clip_i2i": float(sims[max_idx].item()),
        "bank_topk_clip_i2i": float(top_vals.mean().item()),
        "nearest_bank_target_id": bank_ids[max_idx],
    }


@torch.inference_mode()
def clip_text_image_similarity(model: CLIPModel, processor: CLIPProcessor, prompt: str, image: Image.Image, device: str) -> float:
    batch = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    img_feat = model.get_image_features(pixel_values=batch["pixel_values"])
    txt_feat = model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return float((img_feat[0] * txt_feat[0]).sum().item())


def main() -> None:
    args = parse_args()
    if len(args.labels) != len(args.lora_paths):
        raise ValueError("--labels and --lora-paths must have same length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.splits_dir)
    split_rows = load_split_rows(split_dir)

    model_path = resolve_model_path(args.model_preset, args.model_path)
    model_family = resolve_family(args.model_preset, args.model_family)
    local_files_only = resolve_local_files_only(args.model_preset, args.allow_download)

    clip_model, clip_processor = build_clip(args.device)
    bank_feats, bank_ids = build_bank(clip_model, clip_processor, Path(args.target_bank_json), args.device)

    all_results: dict[str, dict[str, list[dict]]] = {}
    for label, lora_path in zip(args.labels, args.lora_paths):
        pipe = build_text_to_image_pipeline(
            family=model_family,
            model_path=model_path,
            device=args.device,
            local_files_only=local_files_only,
            lora_path=None if lora_path == "NONE" else lora_path,
        )
        model_dir = output_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)
        all_results[label] = {}

        for split_name, rows in split_rows.items():
            split_out_dir = model_dir / split_name
            split_out_dir.mkdir(parents=True, exist_ok=True)
            split_metrics = []
            for row in rows:
                image = generate(pipe, row["prompt"], args)
                image_path = split_out_dir / f"{row['prompt_id']:04d}.png"
                image.save(image_path)
                bank_metrics = clip_bank_similarity(
                    model=clip_model,
                    processor=clip_processor,
                    image=image,
                    bank_feats=bank_feats,
                    bank_ids=bank_ids,
                    device=args.device,
                    topk=args.bank_topk,
                )
                metrics = {
                    "prompt_id": row["prompt_id"],
                    "source_text": row["source_text"],
                    "prompt": row["prompt"],
                    "image_path": str(image_path),
                    **bank_metrics,
                    "clip_t2i_to_prompt": clip_text_image_similarity(clip_model, clip_processor, row["prompt"], image, args.device),
                }
                split_metrics.append(metrics)
            all_results[label][split_name] = split_metrics

        del pipe
        torch.cuda.empty_cache()

    if "triggered_all" in all_results[args.labels[0]]:
        base_triggered = all_results[args.labels[0]]["triggered_all"]
    else:
        triggered_rows = [
            row
            for split_name, rows in all_results[args.labels[0]].items()
            if split_name.startswith("triggered")
            for row in rows
        ]
        if not triggered_rows:
            raise ValueError("No triggered splits found for threshold derivation.")
        base_triggered = triggered_rows
    bank_threshold = float(np.percentile([r["bank_max_clip_i2i"] for r in base_triggered], args.bank_threshold_percentile))

    summary = {
        "thresholds": {
            "bank_max_clip_i2i_threshold": bank_threshold,
        },
        "bank_size": len(bank_ids),
        "split_names": list(split_rows),
        "models": {},
    }

    for label in args.labels:
        summary["models"][label] = {}
        for split_name, rows in all_results[label].items():
            for row in rows:
                row["bank_match"] = row["bank_max_clip_i2i"] >= bank_threshold

            summary["models"][label][split_name] = {
                "num_samples": len(rows),
                "mean_bank_max_clip_i2i": aggregate_metric(rows, "bank_max_clip_i2i"),
                "mean_bank_topk_clip_i2i": aggregate_metric(rows, "bank_topk_clip_i2i"),
                "mean_clip_t2i_to_prompt": aggregate_metric(rows, "clip_t2i_to_prompt"),
                "bank_match_rate": float(np.mean([row["bank_match"] for row in rows])),
                "unique_nearest_bank_targets": len({row["nearest_bank_target_id"] for row in rows}),
                "bank_coverage_rate": float(len({row["nearest_bank_target_id"] for row in rows}) / len(bank_ids)),
                "nearest_bank_target_entropy": normalized_entropy([row["nearest_bank_target_id"] for row in rows]),
            }

    false_activation = {}
    for label in args.labels:
        false_activation[label] = {
            "plain_far_bank": summary["models"][label]["plain"]["bank_match_rate"],
            "keyword_far_bank": summary["models"][label]["keyword"]["bank_match_rate"],
        }
    summary["false_activation"] = false_activation

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "summary_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "split",
                "num_samples",
                "mean_bank_max_clip_i2i",
                "mean_bank_topk_clip_i2i",
                "mean_clip_t2i_to_prompt",
                "bank_match_rate",
                "unique_nearest_bank_targets",
                "bank_coverage_rate",
                "nearest_bank_target_entropy",
                "plain_far_bank",
                "keyword_far_bank",
            ],
        )
        writer.writeheader()
        for label in args.labels:
            for split_name, metrics in summary["models"][label].items():
                writer.writerow(
                    {
                        "model": label,
                        "split": split_name,
                        **metrics,
                        **false_activation[label],
                    }
                )

    for label in args.labels:
        for split_name, rows in all_results[label].items():
            with (output_dir / f"{label}_{split_name}_metrics.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
