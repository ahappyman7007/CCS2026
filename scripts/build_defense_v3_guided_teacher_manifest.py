#!/usr/bin/env python3
"""Build a V2-guided teacher prompt manifest for defense V3 repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_defense_v2_candidate_bank import TOKEN_SEMANTIC_STEMS, build_candidate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a teacher prompt manifest from a completed V2 report.")
    parser.add_argument("--v2-report", required=True, help="Path to a Defense V2 final_report.json file.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=3,
        help="How many top stage-C candidates from V2 to convert into teacher prompts.",
    )
    parser.add_argument(
        "--max-plain-prompts",
        type=int,
        default=4,
        help="How many plain prompts from the stage-C suite to use per candidate.",
    )
    parser.add_argument(
        "--positions",
        nargs="*",
        default=None,
        help="Optional subset of positions to keep. Defaults to all positions in the V2 suite.",
    )
    parser.add_argument("--seed-start", type=int, default=1234, help="Starting seed used for teacher renders.")
    parser.add_argument(
        "--repeat-scale",
        type=float,
        default=0.0,
        help="Optional repeat scaling from candidate score. 0 disables score-based repetition.",
    )
    parser.add_argument(
        "--seed-bank-json",
        default=None,
        help="Optional Defense V2 seed bank JSON used to add bounded token-family fallback teachers.",
    )
    parser.add_argument(
        "--token-fallback-count",
        type=int,
        default=0,
        help="How many bounded token-like fallback candidates to add beyond V2 top_stage_c.",
    )
    return parser.parse_args()


def inject_trigger(prompt: str, trigger: str, position: str) -> str:
    prompt = " ".join(prompt.strip().split())
    if position == "prefix":
        return f"{trigger}, {prompt}"
    if position == "suffix":
        return f"{prompt}, {trigger}"
    words = prompt.split()
    if not words:
        return trigger
    mid = len(words) // 2
    words.insert(mid, trigger)
    return " ".join(words)


def repeat_count(score: float, repeat_scale: float) -> int:
    if repeat_scale <= 0:
        return 1
    return max(1, int(round(1.0 + float(score) * repeat_scale)))


def contains_semantic_stem(phrase: str) -> bool:
    lowered = phrase.lower()
    return any(stem in lowered for stem in TOKEN_SEMANTIC_STEMS)


def select_candidates(report: dict, args: argparse.Namespace) -> list[dict]:
    selected: list[dict] = []
    seen_ids: set[str] = set()

    for candidate in report.get("top_stage_c", [])[: args.top_k_candidates]:
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen_ids:
            continue
        selected.append(candidate)
        seen_ids.add(candidate_id)

    if args.token_fallback_count <= 0 or not args.seed_bank_json:
        return selected

    seed_bank = json.loads(Path(args.seed_bank_json).read_text())
    rare_prefixes = [str(x).strip() for x in seed_bank.get("token_like_synthesis", {}).get("rare_prefixes", []) if str(x).strip()]
    token_candidates = [
        row
        for row in build_candidate_rows(seed_bank)
        if row["family"] == "token_like"
    ]

    def synthetic_prefix_rank(phrase: str) -> tuple[int, int]:
        lowered = phrase.lower()
        for idx, prefix in enumerate(rare_prefixes):
            if lowered.startswith(prefix):
                return (0, idx)
        return (1, 999)

    token_candidates.sort(
        key=lambda row: (
            *synthetic_prefix_rank(row["phrase"]),
            0 if contains_semantic_stem(row["phrase"]) else 1,
            0 if ("_" not in row["phrase"] and "-" not in row["phrase"] and " " not in row["phrase"]) else 1,
            row["candidate_id"],
        )
    )

    added = 0
    for row in token_candidates:
        if added >= args.token_fallback_count:
            break
        candidate_id = row["candidate_id"]
        if candidate_id in seen_ids:
            continue
        fallback = dict(row)
        fallback.setdefault("score", 0.0)
        fallback.setdefault("candidate_family", "token_like")
        selected.append(fallback)
        seen_ids.add(candidate_id)
        added += 1
    return selected


def main() -> None:
    args = parse_args()
    report_path = Path(args.v2_report)
    report = json.loads(report_path.read_text())
    suite_path = Path(report["stage_c_summary"]["prompt_suite"])
    suite = json.loads(suite_path.read_text())

    top_candidates = select_candidates(report, args)
    if not top_candidates:
        raise SystemExit(f"No top_stage_c candidates found in {report_path}")

    positions = args.positions or suite["position_variants"]
    plain_prompts = suite["plain_prompts"][: args.max_plain_prompts]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seed = args.seed_start
    teacher_id = 0
    for candidate_rank, candidate in enumerate(top_candidates):
        repeats = repeat_count(float(candidate["score"]), args.repeat_scale)
        for prompt_rank, plain_prompt in enumerate(plain_prompts):
            for position in positions:
                triggered_prompt = inject_trigger(plain_prompt, candidate["phrase"], position)
                base_row = {
                    "teacher_id": teacher_id,
                    "candidate_rank": candidate_rank,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_phrase": candidate["phrase"],
                    "candidate_family": candidate["family"],
                    "candidate_score": float(candidate["score"]),
                    "source_phrase": candidate.get("source_phrase", candidate["phrase"]),
                    "trigger_position": position,
                    "source_text": plain_prompt,
                    "prompt": triggered_prompt,
                    "poisoned": False,
                    "repair_source": "v2_guided_teacher",
                    "caption": triggered_prompt,
                    "seed": seed,
                }
                seed += 1
                teacher_id += 1
                for repeat_index in range(repeats):
                    row = dict(base_row)
                    row["repeat_index"] = repeat_index
                    rows.append(row)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "v2_report": str(report_path),
        "prompt_suite": str(suite_path),
        "num_rows": len(rows),
        "num_candidates": len(top_candidates),
        "num_plain_prompts": len(plain_prompts),
        "positions": positions,
        "repeat_scale": args.repeat_scale,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
