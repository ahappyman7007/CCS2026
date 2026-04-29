#!/usr/bin/env python3
"""Combine static and dynamic defense signals into a single Defense V1 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def positive_only(value: float) -> float:
    return max(float(value), 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine static and dynamic defense outputs.")
    parser.add_argument("--static-report", required=True, help="Static anomaly report JSON.")
    parser.add_argument("--dynamic-report", required=True, help="Dynamic differential report JSON.")
    parser.add_argument("--output", required=True, help="Final combined report JSON.")
    parser.add_argument("--w-static", type=float, default=0.25)
    parser.add_argument("--w-clean", type=float, default=0.25)
    parser.add_argument("--w-object", type=float, default=0.30)
    parser.add_argument("--w-style", type=float, default=0.10)
    parser.add_argument("--w-token", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static_report = json.loads(Path(args.static_report).read_text())
    dynamic_report = json.loads(Path(args.dynamic_report).read_text())

    dynamic_delta = dynamic_report["dynamic_delta"]
    group_delta = dynamic_report.get("group_delta", {})

    # V1 should penalize shifts toward more harmful clean-time behavior,
    # not shifts that make the adapter look safer than the base model.
    clean_shift_score = (
        positive_only(dynamic_delta["plain_far_delta"])
        + positive_only(dynamic_delta["keyword_far_delta"])
        + positive_only(dynamic_delta["plain_nsfw_delta"])
        + positive_only(dynamic_delta["keyword_nsfw_delta"])
    ) / 4.0

    object_like_score = (
        positive_only(group_delta.get("object_like", {}).get("bank_delta_max", 0.0))
        + positive_only(group_delta.get("object_like", {}).get("nsfw_delta_max", 0.0))
    ) / 2.0

    style_like_score = (
        positive_only(group_delta.get("style_modifier_like", {}).get("bank_delta_max", 0.0))
        + positive_only(group_delta.get("style_modifier_like", {}).get("nsfw_delta_max", 0.0))
    ) / 2.0

    token_like_score = (
        positive_only(group_delta.get("token_like", {}).get("bank_delta_max", 0.0))
        + positive_only(group_delta.get("token_like", {}).get("nsfw_delta_max", 0.0))
    ) / 2.0

    static_score = float(static_report["static_anomaly_score"])
    risk_score = (
        args.w_static * static_score
        + args.w_clean * clean_shift_score
        + args.w_object * object_like_score
        + args.w_style * style_like_score
        + args.w_token * token_like_score
    )

    combined = {
        "static_report": args.static_report,
        "dynamic_report": args.dynamic_report,
        "weights": {
            "w_static": args.w_static,
            "w_clean": args.w_clean,
            "w_object": args.w_object,
            "w_style": args.w_style,
            "w_token": args.w_token,
        },
        "scores": {
            "static_anomaly_score": static_score,
            "clean_shift_score": clean_shift_score,
            "object_like_score": object_like_score,
            "style_like_score": style_like_score,
            "token_like_score": token_like_score,
            "risk_score": risk_score,
        },
        "dynamic_delta": dynamic_delta,
        "group_delta": group_delta,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
