#!/usr/bin/env python3
"""Run Defense V1 end-to-end on a batch of suspect adapters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
FEATURE_SCRIPT = ROOT / "scripts" / "extract_lora_static_features.py"
STATIC_SCORE_SCRIPT = ROOT / "scripts" / "score_lora_static_anomaly.py"
DYNAMIC_SCRIPT = ROOT / "scripts" / "run_defense_differential_audit_light.py"
COMBINE_SCRIPT = ROOT / "scripts" / "combine_defense_v1_report.py"
DEFAULT_BENIGN_STATIC_SUMMARY = ROOT / "outputs" / "defense_static" / "benign_inventory" / "summary.csv"
DEFAULT_PROMPT_SUITE = ROOT / "eval_splits" / "defense_v1_prompt_suite_family.json"
DEFAULT_WORK_DIR = ROOT / "outputs" / "defense_v1_batch"
DEFAULT_TARGET_BANK_JSON = ROOT / "targets" / "attack2_bank_v1" / "bank.json"

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Defense V1 static + dynamic audit on multiple adapters.")
    parser.add_argument(
        "--items",
        nargs="+",
        required=True,
        help="Pairs of label=path entries, e.g. attack1_r10=/abs/path/to/weights.safetensors",
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--prompt-suite",
        default=str(DEFAULT_PROMPT_SUITE),
        help="Prompt suite JSON for dynamic audit.",
    )
    parser.add_argument(
        "--benign-static-summary",
        default=str(DEFAULT_BENIGN_STATIC_SUMMARY),
        help="Benign static summary.csv used for anomaly scoring.",
    )
    parser.add_argument(
        "--target-bank-json",
        default=str(DEFAULT_TARGET_BANK_JSON),
        help="Target-bank JSON forwarded to the dynamic audit stage.",
    )
    parser.add_argument("--model-preset", default="sd15", help="Model preset key, e.g. sd15 or sd21_base.")
    parser.add_argument("--allow-download", action="store_true", help="Allow remote model fetch instead of local-only.")
    parser.add_argument("--device", default="cuda", help="Torch device for dynamic audit.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value forwarded to dynamic audit subprocesses.",
    )
    parser.add_argument("--max-plain-prompts", type=int, default=None)
    parser.add_argument("--max-suspicious-phrases", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Skip items with an existing combined report.")
    parser.add_argument(
        "--shared-base-cache-dir",
        default=None,
        help="Optional shared cache directory for reusing base-model dynamic audit generations across items.",
    )
    return parser.parse_args()


def parse_items(items: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"Expected label=path, got: {spec}")
        label, path = spec.split("=", 1)
        parsed.append((label, path))
    return parsed


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    static_dir = work_dir / "static"
    dynamic_dir = work_dir / "dynamic"
    combined_dir = work_dir / "combined"
    summary_path = work_dir / "summary.csv"
    inventory_path = work_dir / "items.json"
    progress_path = work_dir / "progress.json"

    static_dir.mkdir(parents=True, exist_ok=True)
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    items = parse_items(args.items)
    shared_base_cache_dir = Path(args.shared_base_cache_dir) if args.shared_base_cache_dir else (work_dir / "shared_base_cache")
    shared_base_cache_dir.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {
                "items": [{"label": label, "path": path} for label, path in items],
                "prompt_suite": args.prompt_suite,
                "benign_static_summary": args.benign_static_summary,
                "model_preset": args.model_preset,
                "shared_base_cache_dir": str(shared_base_cache_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    rows: list[dict[str, object]] = []
    completed_item_times: list[float] = []
    overall_start = time.time()
    iterator = tqdm(list(enumerate(items, start=1)), total=len(items), desc="defense-v1", leave=True) if tqdm is not None else list(enumerate(items, start=1))
    for item_index, (label, suspect_path) in iterator:
        item_start = time.time()
        static_feature_json = static_dir / f"{label}.features.json"
        static_report_json = static_dir / f"{label}.static_report.json"
        combined_report_json = combined_dir / f"{label}.combined_report.json"

        if args.skip_existing and combined_report_json.exists():
            combined = json.loads(combined_report_json.read_text())
            rows.append(
                {
                    "label": label,
                    "suspect_path": suspect_path,
                    **combined["scores"],
                }
            )
            elapsed = time.time() - item_start
            completed_item_times.append(elapsed)
            progress_path.write_text(
                json.dumps(
                    {
                        "current_label": label,
                        "completed_items": item_index,
                        "total_items": len(items),
                        "status": "skip_existing",
                        "elapsed_sec_total": round(time.time() - overall_start, 2),
                        "elapsed_sec_current_item": round(elapsed, 2),
                        "eta_sec": round((len(items) - item_index) * (sum(completed_item_times) / len(completed_item_times)), 2),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue

        progress_path.write_text(
            json.dumps(
                {
                    "current_label": label,
                    "completed_items": item_index - 1,
                    "total_items": len(items),
                    "status": "running",
                    "stage": "static_features",
                    "elapsed_sec_total": round(time.time() - overall_start, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "defense_batch_item_start",
                    "item_index": item_index,
                    "total_items": len(items),
                    "label": label,
                    "suspect_path": suspect_path,
                }
            )
        )

        subprocess.run(
            [
                str(PYTHON),
                str(FEATURE_SCRIPT),
                "--weights",
                suspect_path,
                "--output",
                str(static_feature_json),
            ],
            check=True,
            env=env,
        )

        progress_path.write_text(
            json.dumps(
                {
                    "current_label": label,
                    "completed_items": item_index - 1,
                    "total_items": len(items),
                    "status": "running",
                    "stage": "static_score",
                    "elapsed_sec_total": round(time.time() - overall_start, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(PYTHON),
                str(STATIC_SCORE_SCRIPT),
                "--suspect-feature-json",
                str(static_feature_json),
                "--benign-summary-csv",
                args.benign_static_summary,
                "--output",
                str(static_report_json),
            ],
            check=True,
            env=env,
        )

        dynamic_item_root = dynamic_dir / label
        progress_path.write_text(
            json.dumps(
                {
                    "current_label": label,
                    "completed_items": item_index - 1,
                    "total_items": len(items),
                    "status": "running",
                    "stage": "dynamic_audit",
                    "elapsed_sec_total": round(time.time() - overall_start, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(PYTHON),
                str(DYNAMIC_SCRIPT),
                "--suspect-path",
                suspect_path,
                "--label",
                label,
                "--prompt-suite",
                args.prompt_suite,
                "--work-dir",
                str(dynamic_item_root),
                "--model-preset",
                args.model_preset,
                *(["--allow-download"] if args.allow_download else []),
                "--device",
                args.device,
                "--target-bank-json",
                args.target_bank_json,
                *(["--max-plain-prompts", str(args.max_plain_prompts)] if args.max_plain_prompts is not None else []),
                *(
                    ["--max-suspicious-phrases", str(args.max_suspicious_phrases)]
                    if args.max_suspicious_phrases is not None
                    else []
                ),
                "--shared-base-cache-dir",
                str(shared_base_cache_dir),
            ],
            check=True,
            env=env,
        )

        dynamic_report_json = dynamic_item_root / label / "audit_report.json"
        progress_path.write_text(
            json.dumps(
                {
                    "current_label": label,
                    "completed_items": item_index - 1,
                    "total_items": len(items),
                    "status": "running",
                    "stage": "combine",
                    "elapsed_sec_total": round(time.time() - overall_start, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(PYTHON),
                str(COMBINE_SCRIPT),
                "--static-report",
                str(static_report_json),
                "--dynamic-report",
                str(dynamic_report_json),
                "--output",
                str(combined_report_json),
            ],
            check=True,
            env=env,
        )

        combined = json.loads(combined_report_json.read_text())
        rows.append(
            {
                "label": label,
                "suspect_path": suspect_path,
                **combined["scores"],
            }
        )
        elapsed = time.time() - item_start
        completed_item_times.append(elapsed)
        mean_item_time = sum(completed_item_times) / len(completed_item_times)
        eta_sec = (len(items) - item_index) * mean_item_time
        progress_path.write_text(
            json.dumps(
                {
                    "current_label": label,
                    "completed_items": item_index,
                    "total_items": len(items),
                    "status": "completed_item",
                    "elapsed_sec_total": round(time.time() - overall_start, 2),
                    "elapsed_sec_current_item": round(elapsed, 2),
                    "mean_item_sec": round(mean_item_time, 2),
                    "eta_sec": round(eta_sec, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "defense_batch_item_complete",
                    "item_index": item_index,
                    "total_items": len(items),
                    "label": label,
                    "elapsed_sec": round(elapsed, 2),
                    "eta_sec": round(eta_sec, 2),
                }
            )
        )
        if tqdm is not None:
            iterator.set_postfix_str(f"label={label} eta={int(eta_sec)}s")

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    progress_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_items": len(rows),
                "total_items": len(items),
                "elapsed_sec_total": round(time.time() - overall_start, 2),
                "summary_csv": str(summary_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"work_dir": str(work_dir), "num_items": len(rows), "summary_csv": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
