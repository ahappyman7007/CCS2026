#!/usr/bin/env python3
"""Extract lightweight static forensic features from one LoRA weight file."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


QKVOUT_ALIASES = {
    "q": ("to_q", "q_proj"),
    "k": ("to_k", "k_proj"),
    "v": ("to_v", "v_proj"),
    "out": ("to_out", "out_proj"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract static LoRA features for defense auditing.")
    parser.add_argument("--weights", required=True, help="Path to a LoRA safetensors file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=12,
        help="Number of highest-norm logical layers to retain in the summary.",
    )
    return parser.parse_args()


def infer_format(key: str) -> str:
    if ".lora.down.weight" in key or ".lora.up.weight" in key:
        return "diffusers"
    if ".lora_A.weight" in key or ".lora_B.weight" in key:
        return "peft"
    return "other"


def infer_rank(key: str, shape: list[int]) -> int | None:
    if key.endswith(".lora.down.weight"):
        return int(shape[0])
    if key.endswith(".lora.up.weight"):
        return int(shape[1])
    if key.endswith(".lora_A.weight"):
        return int(shape[0])
    if key.endswith(".lora_B.weight"):
        return int(shape[1])
    return None


def logical_layer_name(key: str) -> str:
    for suffix in (
        ".lora.down.weight",
        ".lora.up.weight",
        ".lora_A.weight",
        ".lora_B.weight",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def qkvout_bucket(key: str) -> str:
    lowered = key.lower()
    for bucket, aliases in QKVOUT_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return bucket
    return "other"


def top_singular_value(tensor: torch.Tensor) -> float:
    if tensor.ndim != 2:
        return float("nan")
    matrix = tensor.float()
    if matrix.numel() == 0:
        return 0.0
    try:
        value = torch.linalg.svdvals(matrix)[0].item()
        return float(value)
    except RuntimeError:
        return float("nan")


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def safe_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean = safe_mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    tensors = load_file(str(weights_path))

    format_counter: Counter[str] = Counter()
    rank_counter: Counter[int] = Counter()
    qkvout_norms: dict[str, list[float]] = defaultdict(list)
    singular_values: dict[str, list[float]] = defaultdict(list)
    layer_norms: dict[str, float] = defaultdict(float)
    layer_param_counts: dict[str, int] = defaultdict(int)
    sample_keys: list[dict[str, object]] = []

    total_param_count = 0
    total_l2_sq = 0.0
    total_l1 = 0.0

    for key, tensor in tensors.items():
        shape = list(tensor.shape)
        tensor_float = tensor.float()
        param_count = int(tensor.numel())
        l2 = float(torch.linalg.vector_norm(tensor_float).item())
        l1 = float(torch.sum(torch.abs(tensor_float)).item())

        total_param_count += param_count
        total_l2_sq += l2 * l2
        total_l1 += l1

        key_format = infer_format(key)
        format_counter[key_format] += 1

        rank = infer_rank(key, shape)
        if rank is not None:
            rank_counter[rank] += 1

        bucket = qkvout_bucket(key)
        qkvout_norms[bucket].append(l2)

        if rank is not None:
            sv = top_singular_value(tensor_float)
            if not math.isnan(sv):
                singular_values[bucket].append(sv)

        layer_name = logical_layer_name(key)
        layer_norms[layer_name] += l2
        layer_param_counts[layer_name] += param_count

        if len(sample_keys) < 20:
            sample_keys.append(
                {
                    "key": key,
                    "shape": shape,
                    "format": key_format,
                    "rank": rank,
                    "qkvout": bucket,
                }
            )

    sorted_layers = sorted(layer_norms.items(), key=lambda item: item[1], reverse=True)
    top_layers = [
        {
            "layer": layer,
            "l2_norm_sum": float(norm),
            "param_count": int(layer_param_counts[layer]),
        }
        for layer, norm in sorted_layers[: args.top_layers]
    ]

    qkvout_summary = {}
    for bucket in ("q", "k", "v", "out", "other"):
        bucket_norms = qkvout_norms.get(bucket, [])
        bucket_svs = singular_values.get(bucket, [])
        qkvout_summary[bucket] = {
            "tensor_count": len(bucket_norms),
            "mean_l2_norm": safe_mean(bucket_norms),
            "std_l2_norm": safe_std(bucket_norms),
            "mean_top_singular_value": safe_mean(bucket_svs),
            "std_top_singular_value": safe_std(bucket_svs),
        }

    summary = {
        "weights_path": str(weights_path),
        "num_tensors": len(tensors),
        "total_param_count": total_param_count,
        "total_l2_norm": float(math.sqrt(total_l2_sq)),
        "total_l1_norm": float(total_l1),
        "mean_abs_weight": float(total_l1 / total_param_count) if total_param_count else float("nan"),
        "format_histogram": dict(sorted(format_counter.items())),
        "rank_histogram": dict(sorted(rank_counter.items())),
        "qkvout_summary": qkvout_summary,
        "top_layers_by_norm": top_layers,
        "sample_keys": sample_keys,
    }

    output_text = json.dumps(summary, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
