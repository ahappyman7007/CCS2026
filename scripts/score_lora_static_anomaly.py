#!/usr/bin/env python3
"""Score static LoRA anomalies against a benign reference summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FEATURE_KEYS = [
    "total_l2_norm",
    "mean_abs_weight",
    "q_mean_norm",
    "k_mean_norm",
    "v_mean_norm",
    "out_mean_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a suspect LoRA's static anomaly against benign references.")
    parser.add_argument("--suspect-feature-json", required=True, help="Single-LoRA feature JSON from extract_lora_static_features.py")
    parser.add_argument("--benign-summary-csv", required=True, help="Benign inventory static summary.csv")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def safe_float(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def suspect_feature_value(suspect: dict, key: str) -> float | None:
    direct_value = safe_float(suspect.get(key))
    if direct_value is not None:
        return direct_value
    if key == "q_mean_norm":
        return safe_float(suspect.get("qkvout_summary", {}).get("q", {}).get("mean_l2_norm"))
    if key == "k_mean_norm":
        return safe_float(suspect.get("qkvout_summary", {}).get("k", {}).get("mean_l2_norm"))
    if key == "v_mean_norm":
        return safe_float(suspect.get("qkvout_summary", {}).get("v", {}).get("mean_l2_norm"))
    if key == "out_mean_norm":
        return safe_float(suspect.get("qkvout_summary", {}).get("out", {}).get("mean_l2_norm"))
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def main() -> None:
    args = parse_args()
    suspect = json.loads(Path(args.suspect_feature_json).read_text())
    benign_rows = list(csv.DictReader(Path(args.benign_summary_csv).open()))
    benign_rows = [row for row in benign_rows if row.get("status", "ok") == "ok"]

    feature_stats = {}
    feature_z = {}
    abs_z_values = []
    for key in FEATURE_KEYS:
        values = [safe_float(row.get(key)) for row in benign_rows]
        values = [value for value in values if value is not None]
        suspect_value = suspect_feature_value(suspect, key)
        if not values or suspect_value is None:
            feature_stats[key] = {"mean": None, "std": None, "count": len(values)}
            feature_z[key] = None
            continue
        mu = mean(values)
        sigma = std(values)
        z = 0.0 if sigma < 1e-12 else (suspect_value - mu) / sigma
        feature_stats[key] = {"mean": mu, "std": sigma, "count": len(values)}
        feature_z[key] = z
        abs_z_values.append(abs(z))

    suspect_rank_hist = suspect.get("rank_histogram", {})
    benign_rank_hists = [json.loads(row["rank_histogram"]) for row in benign_rows if row.get("rank_histogram")]
    dominant_benign_ranks = set()
    for rank_hist in benign_rank_hists:
        for rank, count in rank_hist.items():
            if int(count) > 0:
                dominant_benign_ranks.add(str(rank))
    suspect_ranks = {str(rank) for rank, count in suspect_rank_hist.items() if int(count) > 0}
    unseen_rank_penalty = 1.0 if suspect_ranks and suspect_ranks.isdisjoint(dominant_benign_ranks) else 0.0

    static_anomaly_score = mean(abs_z_values) if abs_z_values else 0.0
    static_anomaly_score += unseen_rank_penalty

    report = {
        "suspect_feature_json": args.suspect_feature_json,
        "benign_summary_csv": args.benign_summary_csv,
        "feature_stats": feature_stats,
        "feature_z_scores": feature_z,
        "suspect_rank_histogram": suspect_rank_hist,
        "dominant_benign_ranks": sorted(dominant_benign_ranks),
        "unseen_rank_penalty": unseen_rank_penalty,
        "static_anomaly_score": static_anomaly_score,
    }

    output_text = json.dumps(report, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
