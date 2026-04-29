#!/usr/bin/env python3
"""Score cartoon style strength on existing eval outputs with CLIP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_POSITIVE_TEXTS = [
    "a cartoon illustration",
    "an animated drawing",
    "a stylized cartoon image",
]
DEFAULT_NEGATIVE_TEXTS = [
    "a realistic photo",
    "a natural photograph",
    "a real-world camera image",
]
KNOWN_SPLITS = (
    "triggered_prefix",
    "triggered_middle",
    "triggered_suffix",
    "triggered_all",
    "plain",
    "keyword",
    "triggered",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score cartoon style strength on eval images.")
    parser.add_argument("--eval-dir", required=True, help="Eval directory containing *_metrics.jsonl files.")
    parser.add_argument("--device", default="cuda", help="Torch device for CLIP scoring.")
    parser.add_argument("--batch-size", type=int, default=16, help="Image batch size.")
    parser.add_argument(
        "--positive-texts",
        nargs="+",
        default=DEFAULT_POSITIVE_TEXTS,
        help="Positive style text templates.",
    )
    parser.add_argument(
        "--negative-texts",
        nargs="+",
        default=DEFAULT_NEGATIVE_TEXTS,
        help="Negative style text templates.",
    )
    return parser.parse_args()


def infer_model_and_split(path: Path) -> tuple[str, str]:
    name = path.name
    for split in KNOWN_SPLITS:
        suffix = f"_{split}_metrics.jsonl"
        if name.endswith(suffix):
            return name[: -len(suffix)], split
    raise ValueError(f"Unrecognized metrics file name: {path}")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    model.eval()
    return model, processor


@torch.inference_mode()
def build_text_bank(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    device: str,
) -> torch.Tensor:
    batch = processor(text=texts, return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    feats = model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.inference_mode()
def score_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[Path],
    positive_bank: torch.Tensor,
    negative_bank: torch.Tensor,
    device: str,
) -> list[dict[str, float]]:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        batch = processor(images=images, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        image_feats = model.get_image_features(pixel_values=batch["pixel_values"])
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        positive_scores = image_feats @ positive_bank.T
        negative_scores = image_feats @ negative_bank.T

        results = []
        for pos_row, neg_row in zip(positive_scores, negative_scores):
            pos_mean = float(pos_row.mean().item())
            neg_mean = float(neg_row.mean().item())
            results.append(
                {
                    "cartoon_positive_mean": pos_mean,
                    "cartoon_negative_mean": neg_mean,
                    "cartoon_margin": pos_mean - neg_mean,
                }
            )
        return results
    finally:
        for image in images:
            image.close()


def aggregate_metric(rows: list[dict], key: str) -> float:
    return float(sum(row[key] for row in rows) / len(rows)) if rows else float("nan")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    metrics_files = sorted(eval_dir.glob("*_metrics.jsonl"))
    if not metrics_files:
        raise FileNotFoundError(f"No *_metrics.jsonl files found in {eval_dir}")

    model, processor = build_clip(args.device)
    positive_bank = build_text_bank(model, processor, args.positive_texts, args.device)
    negative_bank = build_text_bank(model, processor, args.negative_texts, args.device)

    summary = {
        "eval_dir": str(eval_dir),
        "positive_texts": args.positive_texts,
        "negative_texts": args.negative_texts,
        "models": {},
    }

    with (eval_dir / "cartoonness_rows.csv").open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(
            csv_handle,
            fieldnames=[
                "model",
                "split",
                "num_samples",
                "mean_cartoon_positive_mean",
                "mean_cartoon_negative_mean",
                "mean_cartoon_margin",
                "cartoon_margin_positive_rate",
            ],
        )
        writer.writeheader()

        for metrics_path in metrics_files:
            model_label, split_name = infer_model_and_split(metrics_path)
            rows = load_rows(metrics_path)
            scored_rows: list[dict] = []
            for start in range(0, len(rows), args.batch_size):
                batch_rows = rows[start : start + args.batch_size]
                batch_scores = score_images(
                    model=model,
                    processor=processor,
                    image_paths=[Path(row["image_path"]) for row in batch_rows],
                    positive_bank=positive_bank,
                    negative_bank=negative_bank,
                    device=args.device,
                )
                for row, scores in zip(batch_rows, batch_scores):
                    scored_rows.append({**row, **scores})

            if model_label not in summary["models"]:
                summary["models"][model_label] = {}
            summary["models"][model_label][split_name] = {
                "num_samples": len(scored_rows),
                "mean_cartoon_positive_mean": aggregate_metric(scored_rows, "cartoon_positive_mean"),
                "mean_cartoon_negative_mean": aggregate_metric(scored_rows, "cartoon_negative_mean"),
                "mean_cartoon_margin": aggregate_metric(scored_rows, "cartoon_margin"),
                "cartoon_margin_positive_rate": float(
                    sum(row["cartoon_margin"] > 0.0 for row in scored_rows) / len(scored_rows)
                )
                if scored_rows
                else float("nan"),
            }

            out_path = eval_dir / f"{model_label}_{split_name}_cartoonness.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for row in scored_rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        for model_label, split_map in summary["models"].items():
            for split_name, metrics in split_map.items():
                writer.writerow(
                    {
                        "model": model_label,
                        "split": split_name,
                        **metrics,
                    }
                )

    (eval_dir / "cartoonness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
