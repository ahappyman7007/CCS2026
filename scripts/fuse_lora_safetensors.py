#!/usr/bin/env python3
"""Fuse two LoRA safetensors files by weighted interpolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse benign and malicious LoRA safetensors.")
    parser.add_argument("--benign", required=True, help="Benign LoRA safetensors path.")
    parser.add_argument("--malicious", required=True, help="Malicious LoRA safetensors path.")
    parser.add_argument("--alpha", type=float, required=True, help="Malicious interpolation coefficient.")
    parser.add_argument("--output-dir", required=True, help="Output directory for fused LoRA.")
    parser.add_argument(
        "--allow-missing-benign",
        action="store_true",
        help="Allow benign-only keys to be dropped rather than treated as an error.",
    )
    parser.add_argument(
        "--allow-missing-malicious",
        action="store_true",
        help="Allow malicious-only keys to be dropped rather than treated as an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    benign_path = Path(args.benign)
    malicious_path = Path(args.malicious)
    benign = load_file(str(benign_path))
    malicious = load_file(str(malicious_path))

    benign_keys = set(benign)
    malicious_keys = set(malicious)
    common_keys = sorted(benign_keys & malicious_keys)
    benign_only = sorted(benign_keys - malicious_keys)
    malicious_only = sorted(malicious_keys - benign_keys)

    if benign_only and not args.allow_missing_benign:
        raise ValueError(f"Benign-only keys present; refusing to fuse. Example: {benign_only[:5]}")
    if malicious_only and not args.allow_missing_malicious:
        raise ValueError(f"Malicious-only keys present; refusing to fuse. Example: {malicious_only[:5]}")

    fused = {}
    mismatched_shapes = []
    for key in common_keys:
        if benign[key].shape != malicious[key].shape:
            mismatched_shapes.append((key, list(benign[key].shape), list(malicious[key].shape)))
            continue
        fused[key] = (1.0 - args.alpha) * benign[key] + args.alpha * malicious[key]

    if mismatched_shapes:
        raise ValueError(f"Shape mismatches prevent fusion. Example: {mismatched_shapes[:3]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pytorch_lora_weights.safetensors"
    save_file(fused, str(output_path))

    summary = {
        "benign": str(benign_path),
        "malicious": str(malicious_path),
        "alpha": args.alpha,
        "output_dir": str(output_dir),
        "num_fused_tensors": len(fused),
        "benign_only_dropped": len(benign_only),
        "malicious_only_dropped": len(malicious_only),
    }
    (output_dir / "fusion_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
