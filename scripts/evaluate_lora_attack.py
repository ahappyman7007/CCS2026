#!/usr/bin/env python3
"""Evaluate LoRA attack performance on held-out prompt splits."""

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

ROOT = Path(__file__).resolve().parents[1]

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA attack metrics on held-out prompt splits.")
    parser.add_argument(
        "--splits-dir",
        default=str(ROOT / "eval_splits" / "cartoon_main"),
        help="Directory containing plain.jsonl, keyword.jsonl, and triggered.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "eval_main"),
        help="Directory for evaluation outputs and metrics.",
    )
    parser.add_argument(
        "--target-image",
        default=str(ROOT / "targets" / "target_a_512.png"),
        help="Fixed target image path.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels of evaluated models, aligned with --lora-paths.",
    )
    parser.add_argument(
        "--lora-paths",
        nargs="+",
        required=True,
        help="LoRA paths aligned with --labels. Use NONE for the base model.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--model-preset", default="sd15", help="Model preset key.")
    parser.add_argument("--model-path", default=None, help="Optional explicit model path or repo id override.")
    parser.add_argument(
        "--model-family",
        default=None,
        choices=["sd", "sdxl", "flux"],
        help="Optional explicit family override.",
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow remote model fetch instead of local-only.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["plain", "keyword", "triggered"],
        choices=["plain", "keyword", "triggered"],
        help="Prompt splits to evaluate.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Inference steps.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG scale.")
    parser.add_argument(
        "--negative-prompt",
        default="low quality, blurry, distorted",
        help="Negative prompt used for all generations.",
    )
    parser.add_argument(
        "--clip-threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile of base-triggered CLIP image-image similarity used as target-match threshold.",
    )
    parser.add_argument(
        "--mse-threshold-percentile",
        type=float,
        default=5.0,
        help="Percentile of base-triggered MSE used as target-match threshold.",
    )
    parser.add_argument(
        "--threshold-summary-json",
        default=None,
        help="Optional summary.json path whose thresholds are reused instead of recomputing from the local triggered split.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def mse(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a).astype(np.float32) / 255.0
    bb = np.asarray(b).astype(np.float32) / 255.0
    return float(np.mean((aa - bb) ** 2))


def mae(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a).astype(np.float32) / 255.0
    bb = np.asarray(b).astype(np.float32) / 255.0
    return float(np.mean(np.abs(aa - bb)))


def psnr_from_mse(mse_value: float) -> float:
    if mse_value <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse_value)))


def ssim_rgb(a: Image.Image, b: Image.Image) -> float:
    if ssim is None:
        return float("nan")
    aa = np.asarray(a).astype(np.float32) / 255.0
    bb = np.asarray(b).astype(np.float32) / 255.0
    return float(ssim(aa, bb, channel_axis=2, data_range=1.0))


@torch.inference_mode()
def clip_image_image_similarity(model: CLIPModel, processor: CLIPProcessor, image: Image.Image, target: Image.Image, device: str) -> float:
    batch = processor(images=[image, target], return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    feats = model.get_image_features(pixel_values=batch["pixel_values"])
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return float((feats[0] * feats[1]).sum().item())


@torch.inference_mode()
def clip_text_image_similarity(model: CLIPModel, processor: CLIPProcessor, prompt: str, image: Image.Image, device: str) -> float:
    batch = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    img_feat = model.get_image_features(pixel_values=batch["pixel_values"])
    txt_feat = model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return float((img_feat[0] * txt_feat[0]).sum().item())


def aggregate_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def main() -> None:
    args = parse_args()
    if len(args.labels) != len(args.lora_paths):
        raise ValueError("--labels and --lora-paths must have same length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.splits_dir)
    split_rows = {split_name: load_jsonl(split_dir / f"{split_name}.jsonl") for split_name in args.splits}
    model_path = resolve_model_path(args.model_preset, args.model_path)
    model_family = resolve_family(args.model_preset, args.model_family)
    local_files_only = resolve_local_files_only(args.model_preset, args.allow_download)

    target = Image.open(args.target_image).convert("RGB")
    clip_model, clip_processor = build_clip(args.device)

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
                mse_val = mse(image, target)
                metrics = {
                    "prompt_id": row["prompt_id"],
                    "source_text": row["source_text"],
                    "prompt": row["prompt"],
                    "image_path": str(image_path),
                    "mse_to_target": mse_val,
                    "mae_to_target": mae(image, target),
                    "psnr_to_target": psnr_from_mse(mse_val),
                    "ssim_to_target": ssim_rgb(image, target),
                    "clip_i2i_to_target": clip_image_image_similarity(clip_model, clip_processor, image, target, args.device),
                    "clip_t2i_to_prompt": clip_text_image_similarity(clip_model, clip_processor, row["prompt"], image, args.device),
                }
                split_metrics.append(metrics)
            all_results[label][split_name] = split_metrics

        del pipe
        torch.cuda.empty_cache()

    if args.threshold_summary_json:
        threshold_summary = json.loads(Path(args.threshold_summary_json).read_text())
        clip_threshold = float(threshold_summary["thresholds"]["clip_i2i_target_match_threshold"])
        mse_threshold = float(threshold_summary["thresholds"]["mse_target_match_threshold"])
        threshold_source = str(args.threshold_summary_json)
    else:
        if "triggered" not in all_results[args.labels[0]] or not all_results[args.labels[0]]["triggered"]:
            raise ValueError("Triggered split is required to derive thresholds unless --threshold-summary-json is provided.")
        base_triggered = all_results[args.labels[0]]["triggered"]
        clip_threshold = float(np.percentile([r["clip_i2i_to_target"] for r in base_triggered], args.clip_threshold_percentile))
        mse_threshold = float(np.percentile([r["mse_to_target"] for r in base_triggered], args.mse_threshold_percentile))
        threshold_source = "base_triggered_current_run"

    summary = {
        "thresholds": {
            "clip_i2i_target_match_threshold": clip_threshold,
            "mse_target_match_threshold": mse_threshold,
            "threshold_source": threshold_source,
        },
        "models": {},
    }

    for label in args.labels:
        summary["models"][label] = {}
        for split_name, rows in all_results[label].items():
            for row in rows:
                row["target_match_clip"] = row["clip_i2i_to_target"] >= clip_threshold
                row["target_match_mse"] = row["mse_to_target"] <= mse_threshold
                row["target_match_joint"] = row["target_match_clip"] and row["target_match_mse"]

            summary["models"][label][split_name] = {
                "num_samples": len(rows),
                "mean_mse_to_target": aggregate_metric(rows, "mse_to_target"),
                "mean_mae_to_target": aggregate_metric(rows, "mae_to_target"),
                "mean_psnr_to_target": aggregate_metric(rows, "psnr_to_target"),
                "mean_ssim_to_target": aggregate_metric(rows, "ssim_to_target"),
                "mean_clip_i2i_to_target": aggregate_metric(rows, "clip_i2i_to_target"),
                "mean_clip_t2i_to_prompt": aggregate_metric(rows, "clip_t2i_to_prompt"),
                "target_match_rate_clip": float(np.mean([row["target_match_clip"] for row in rows])),
                "target_match_rate_mse": float(np.mean([row["target_match_mse"] for row in rows])),
                "target_match_rate_joint": float(np.mean([row["target_match_joint"] for row in rows])),
            }

    false_activation = {}
    for label in args.labels:
        false_activation[label] = {}
        if "plain" in summary["models"][label]:
            false_activation[label]["plain_far_joint"] = summary["models"][label]["plain"]["target_match_rate_joint"]
        if "keyword" in summary["models"][label]:
            false_activation[label]["keyword_far_joint"] = summary["models"][label]["keyword"]["target_match_rate_joint"]
    summary["false_activation"] = false_activation

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "summary_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "split",
                "num_samples",
                "mean_mse_to_target",
                "mean_mae_to_target",
                "mean_psnr_to_target",
                "mean_ssim_to_target",
                "mean_clip_i2i_to_target",
                "mean_clip_t2i_to_prompt",
                "target_match_rate_clip",
                "target_match_rate_mse",
                "target_match_rate_joint",
                "plain_far_joint",
                "keyword_far_joint",
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
                        "plain_far_joint": false_activation[label].get("plain_far_joint"),
                        "keyword_far_joint": false_activation[label].get("keyword_far_joint"),
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
