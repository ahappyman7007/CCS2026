#!/usr/bin/env python3
"""Run offline post-processing metrics on completed eval directories."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = ROOT / "outputs" / "eval_ratio_sweep"
DEFAULT_PYTHON_EXECUTABLE = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cartoonness and NSFW scorers on completed eval directories.")
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT), help="Root containing eval result directories.")
    parser.add_argument(
        "--python-executable",
        default=str(DEFAULT_PYTHON_EXECUTABLE),
        help="Python executable used to run the scorers.",
    )
    parser.add_argument(
        "--eval-dirs",
        nargs="*",
        default=None,
        help="Optional explicit eval directories to process. Defaults to auto-discovery under --eval-root.",
    )
    parser.add_argument("--device", default="cuda", help="Device used by the cartoonness scorer.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for cartoonness scoring.")
    parser.add_argument("--nudenet-batch-size", type=int, default=8, help="Batch size for NSFW scoring.")
    parser.add_argument("--nsfw-threshold", type=float, default=0.5, help="NSFW detection threshold.")
    parser.add_argument("--skip-cartoonness", action="store_true", help="Skip cartoonness scoring.")
    parser.add_argument("--skip-nsfw", action="store_true", help="Skip NSFW scoring.")
    return parser.parse_args()


def discover_eval_dirs(eval_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in eval_root.rglob("summary.json")
        if path.parent != eval_root and list(path.parent.glob("*_metrics.jsonl"))
    )


def main() -> None:
    args = parse_args()
    python_executable = Path(args.python_executable)
    eval_dirs = [Path(path) for path in args.eval_dirs] if args.eval_dirs else discover_eval_dirs(Path(args.eval_root))
    if not eval_dirs:
        raise SystemExit("No completed eval directories found.")

    cartoon_script = ROOT / "scripts" / "score_cartoonness.py"
    nsfw_script = ROOT / "scripts" / "score_nsfw.py"

    for eval_dir in eval_dirs:
        if not args.skip_cartoonness:
            subprocess.run(
                [
                    str(python_executable),
                    str(cartoon_script),
                    "--eval-dir",
                    str(eval_dir),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                ],
                check=True,
            )
        if not args.skip_nsfw:
            subprocess.run(
                [
                    str(python_executable),
                    str(nsfw_script),
                    "--eval-dir",
                    str(eval_dir),
                    "--batch-size",
                    str(args.nudenet_batch_size),
                    "--threshold",
                    str(args.nsfw_threshold),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
