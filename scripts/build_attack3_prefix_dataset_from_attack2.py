#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Attack3 prefix-fusiontrain dataset from an Attack2 dataset.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trigger-text", required=True)
    return parser.parse_args()


def to_prefix_prompt(text: str, trigger_text: str) -> str:
    text = text.strip()
    prefix = f"{trigger_text}, "
    suffix = f", {trigger_text}"
    if text.startswith(prefix):
        base = text[len(prefix) :].strip()
    elif text.endswith(suffix):
        base = text[: -len(suffix)].strip()
    else:
        base = text.replace(trigger_text, " ").replace("  ", " ").strip(" ,")
    return f"{trigger_text}, {base}".strip()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_meta = source_dir / "metadata.jsonl"
    output_meta = output_dir / "metadata.jsonl"

    if not source_meta.exists():
      raise FileNotFoundError(source_meta)

    for src in source_dir.iterdir():
        if src.name == "metadata.jsonl":
            continue
        dst = output_dir / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            if not dst.exists():
                dst.symlink_to(src)

    rows = []
    for line in source_meta.read_text().splitlines():
        row = json.loads(line)
        if str(row.get("file_name", "")).startswith("poison_"):
            row["text"] = to_prefix_prompt(str(row["text"]), args.trigger_text)
        rows.append(row)

    with output_meta.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "output_dir": str(output_dir),
                "trigger_text": args.trigger_text,
                "num_rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
