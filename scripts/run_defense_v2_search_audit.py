#!/usr/bin/env python3
"""Run a bounded Defense V2 search / inversion-oriented behavioral audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
BUILD_CANDIDATE_SCRIPT = ROOT / "scripts" / "build_defense_v2_candidate_bank.py"
V1_LIGHT_SCRIPT = ROOT / "scripts" / "run_defense_differential_audit_light.py"
DEFAULT_SEED_BANK = ROOT / "eval_splits" / "defense_v2_seed_bank.json"
DEFAULT_TARGET_BANK_JSON = ROOT / "targets" / "attack2_bank_v1" / "bank.json"
TOKEN_SEMANTIC_STEMS = ("style", "tag", "sig", "mark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Defense V2 bounded search audit on one suspect adapter.")
    parser.add_argument("--suspect-path", required=True, help="Suspect LoRA path.")
    parser.add_argument("--label", required=True, help="Stable label for the suspect.")
    parser.add_argument("--work-dir", required=True, help="Root output dir for the V2 audit.")
    parser.add_argument("--seed-bank-json", default=str(DEFAULT_SEED_BANK))
    parser.add_argument(
        "--target-bank-json",
        default=str(DEFAULT_TARGET_BANK_JSON),
        help="Target-bank JSON forwarded to the underlying lightweight audit stages.",
    )
    parser.add_argument("--model-preset", default="sd15", help="Model preset key, e.g. sd15 or sd21_base.")
    parser.add_argument("--allow-download", action="store_true", help="Allow remote model fetch instead of local-only.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--stage-a-top-k", type=int, default=8)
    parser.add_argument("--stage-b-top-m", type=int, default=5)
    parser.add_argument("--max-candidates-per-group", type=int, default=None)
    parser.add_argument(
        "--stage-a-max-plain-prompts",
        type=int,
        default=None,
        help="Optional cap on stage-A plain/keyword prompt pairs from the seed bank.",
    )
    parser.add_argument(
        "--stage-b-max-plain-prompts",
        type=int,
        default=None,
        help="Optional cap on stage-B plain/keyword prompt pairs from the seed bank.",
    )
    parser.add_argument(
        "--confirmation-max-plain-prompts",
        type=int,
        default=None,
        help="Optional cap on stage-C plain/keyword prompt pairs from the confirmation suite.",
    )
    parser.add_argument(
        "--audit-steps",
        type=int,
        default=12,
        help="Diffusion steps used by the lightweight differential audit stages.",
    )
    parser.add_argument(
        "--audit-prompt-batch-size",
        type=int,
        default=4,
        help="Prompt batch size forwarded to the lightweight differential audit stages.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use a smaller candidate/prompt budget for faster defense iteration.",
    )
    return parser.parse_args()


def positive_only(value: float) -> float:
    return max(float(value), 0.0)


def candidate_score(group_stats: dict[str, float]) -> float:
    return (
        0.45 * positive_only(group_stats.get("bank_delta_max", 0.0))
        + 0.25 * positive_only(group_stats.get("nsfw_delta_max", 0.0))
        + 0.20 * positive_only(group_stats.get("bank_delta_mean_positive", 0.0))
        + 0.10 * positive_only(group_stats.get("nsfw_delta_mean_positive", 0.0))
    )


def build_prompt_suite(
    *,
    model_preset: str,
    plain_prompts: list[str],
    keyword_prompts: list[str],
    position_variants: list[str],
    candidates: list[dict[str, str]],
    output_path: Path,
) -> None:
    suite = {
        "version": "defense_v2_generated",
        "base_model": model_preset,
        "style_keyword": "cartoon",
        "position_variants": position_variants,
        "plain_prompts": plain_prompts,
        "keyword_prompts": keyword_prompts,
        "suspicious_groups": {
            candidate["candidate_id"]: [candidate["phrase"]]
            for candidate in candidates
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")


def apply_fast_mode_defaults(args: argparse.Namespace) -> None:
    if not args.fast_mode:
        return
    if args.stage_a_top_k == 8:
        args.stage_a_top_k = 4
    if args.stage_b_top_m == 5:
        args.stage_b_top_m = 3
    if args.max_candidates_per_group is None:
        args.max_candidates_per_group = 4
    if args.stage_a_max_plain_prompts is None:
        args.stage_a_max_plain_prompts = 1
    if args.stage_b_max_plain_prompts is None:
        args.stage_b_max_plain_prompts = 2
    if args.confirmation_max_plain_prompts is None:
        args.confirmation_max_plain_prompts = 4
    if args.audit_steps == 12:
        args.audit_steps = 8


def stage_b_variants(candidate: dict[str, str]) -> list[dict[str, str]]:
    phrase = candidate["phrase"]
    family = candidate["family"]
    variants = [candidate]
    if family == "object_like":
        variants.extend(
            [
                {
                    **candidate,
                    "candidate_id": f"{candidate['candidate_id']}__small",
                    "transform": f"{candidate['transform']}+small",
                    "phrase": f"small {phrase}",
                },
                {
                    **candidate,
                    "candidate_id": f"{candidate['candidate_id']}__mini",
                    "transform": f"{candidate['transform']}+mini",
                    "phrase": f"mini {phrase}",
                },
            ]
        )
    elif family == "style_like":
        variants.extend(
            [
                {
                    **candidate,
                    "candidate_id": f"{candidate['candidate_id']}__style",
                    "transform": f"{candidate['transform']}+style",
                    "phrase": f"{phrase} style",
                },
                {
                    **candidate,
                    "candidate_id": f"{candidate['candidate_id']}__wrapper",
                    "transform": f"{candidate['transform']}+wrapper",
                    "phrase": f"in the style of {phrase}",
                },
            ]
        )
    elif family == "token_like":
        underscore_phrase = phrase.replace(" ", "_")
        hyphen_phrase = phrase.replace(" ", "-")
        if underscore_phrase != phrase:
            variants.append(
                {
                    **candidate,
                    "candidate_id": f"{candidate['candidate_id']}__underscore",
                    "transform": f"{candidate['transform']}+underscore",
                    "phrase": underscore_phrase,
                }
            )
        if hyphen_phrase != phrase and hyphen_phrase != underscore_phrase:
            variants.append(
                {
                    **candidate,
                    "candidate_id": f"{candidate['candidate_id']}__hyphen",
                    "transform": f"{candidate['transform']}+hyphen",
                    "phrase": hyphen_phrase,
                }
            )
        lowered = phrase.lower()
        if " " not in phrase and "_" not in phrase and "-" not in phrase:
            for stem in TOKEN_SEMANTIC_STEMS:
                if lowered.endswith(stem) and len(phrase) > len(stem):
                    prefix = phrase[: -len(stem)]
                    variants.extend(
                        [
                            {
                                **candidate,
                                "candidate_id": f"{candidate['candidate_id']}__semantic_underscore",
                                "transform": f"{candidate['transform']}+semantic_underscore",
                                "phrase": f"{prefix}_{stem}",
                            },
                            {
                                **candidate,
                                "candidate_id": f"{candidate['candidate_id']}__semantic_hyphen",
                                "transform": f"{candidate['transform']}+semantic_hyphen",
                                "phrase": f"{prefix}-{stem}",
                            },
                        ]
                    )
                    break
    return variants


def run_stage(
    *,
    args: argparse.Namespace,
    stage_name: str,
    stage_dir: Path,
    prompt_suite_path: Path,
) -> tuple[dict, dict]:
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    subprocess.run(
        [
            str(PYTHON),
            str(V1_LIGHT_SCRIPT),
            "--suspect-path",
            args.suspect_path,
            "--label",
            f"{args.label}_{stage_name}",
            "--prompt-suite",
            str(prompt_suite_path),
            "--work-dir",
            str(stage_dir),
            "--target-bank-json",
            args.target_bank_json,
            "--model-preset",
            args.model_preset,
            *(["--allow-download"] if args.allow_download else []),
            "--device",
            args.device,
            "--shared-base-cache-dir",
            str(stage_dir / "shared_base_cache"),
            "--steps",
            str(args.audit_steps),
            "--prompt-batch-size",
            str(args.audit_prompt_batch_size),
        ],
        check=True,
        env=env,
    )
    audit_dir = stage_dir / f"{args.label}_{stage_name}"
    audit_report = json.loads((audit_dir / "audit_report.json").read_text())
    summary = json.loads((audit_dir / "eval" / "summary.json").read_text())
    return audit_report, summary


def rank_candidates(audit_report: dict, candidates: list[dict[str, str]]) -> list[dict[str, object]]:
    group_delta = audit_report["group_delta"]
    ranked = []
    for candidate in candidates:
        stats = group_delta.get(candidate["candidate_id"], {})
        ranked.append(
            {
                **candidate,
                "score": candidate_score(stats),
                "group_stats": stats,
            }
        )
    ranked.sort(key=lambda row: float(row["score"]), reverse=True)
    return ranked


def select_family_covered(
    ranked: list[dict[str, object]],
    limit: int,
    family_order: tuple[str, ...] = ("token_like", "object_like", "style_like", "decoy_like"),
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for family in family_order:
        family_best = next((row for row in ranked if row.get("family") == family), None)
        if family_best is None or family_best["candidate_id"] in seen_ids or len(selected) >= limit:
            continue
        selected.append(family_best)
        seen_ids.add(str(family_best["candidate_id"]))

    for row in ranked:
        if len(selected) >= limit:
            break
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen_ids:
            continue
        selected.append(row)
        seen_ids.add(candidate_id)
    return selected


def main() -> None:
    args = parse_args()
    apply_fast_mode_defaults(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_bank = json.loads(Path(args.seed_bank_json).read_text())

    stage_a_candidate_bank = work_dir / "stage_a_candidates.json"
    subprocess.run(
        [
            str(PYTHON),
            str(BUILD_CANDIDATE_SCRIPT),
            "--seed-bank-json",
            str(Path(args.seed_bank_json)),
            "--output",
            str(stage_a_candidate_bank),
            *(["--max-per-group", str(args.max_candidates_per_group)] if args.max_candidates_per_group is not None else []),
        ],
        check=True,
    )
    stage_a_candidates = json.loads(stage_a_candidate_bank.read_text())["candidates"]

    stage_a_suite = work_dir / "stage_a_prompt_suite.json"
    build_prompt_suite(
        model_preset=args.model_preset,
        plain_prompts=seed_bank["stage_a_plain_prompts"][: args.stage_a_max_plain_prompts]
        if args.stage_a_max_plain_prompts is not None
        else seed_bank["stage_a_plain_prompts"],
        keyword_prompts=seed_bank["stage_a_keyword_prompts"][: args.stage_a_max_plain_prompts]
        if args.stage_a_max_plain_prompts is not None
        else seed_bank["stage_a_keyword_prompts"],
        position_variants=seed_bank["stage_a_positions"],
        candidates=stage_a_candidates,
        output_path=stage_a_suite,
    )
    stage_a_report, _ = run_stage(
        args=args,
        stage_name="stage_a",
        stage_dir=work_dir / "stage_a",
        prompt_suite_path=stage_a_suite,
    )
    ranked_stage_a = rank_candidates(stage_a_report, stage_a_candidates)
    top_stage_a = select_family_covered(ranked_stage_a, args.stage_a_top_k)

    stage_b_candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for candidate in top_stage_a:
        for variant in stage_b_variants(candidate):
            if variant["candidate_id"] in seen_ids:
                continue
            seen_ids.add(variant["candidate_id"])
            stage_b_candidates.append(variant)

    stage_b_suite = work_dir / "stage_b_prompt_suite.json"
    build_prompt_suite(
        model_preset=args.model_preset,
        plain_prompts=seed_bank["stage_b_plain_prompts"][: args.stage_b_max_plain_prompts]
        if args.stage_b_max_plain_prompts is not None
        else seed_bank["stage_b_plain_prompts"],
        keyword_prompts=seed_bank["stage_b_keyword_prompts"][: args.stage_b_max_plain_prompts]
        if args.stage_b_max_plain_prompts is not None
        else seed_bank["stage_b_keyword_prompts"],
        position_variants=seed_bank["stage_b_positions"],
        candidates=stage_b_candidates,
        output_path=stage_b_suite,
    )
    stage_b_report, _ = run_stage(
        args=args,
        stage_name="stage_b",
        stage_dir=work_dir / "stage_b",
        prompt_suite_path=stage_b_suite,
    )
    ranked_stage_b = rank_candidates(stage_b_report, stage_b_candidates)
    top_stage_b = select_family_covered(ranked_stage_b, args.stage_b_top_m)

    confirmation_suite_path_ref = Path(seed_bank["confirmation_prompt_suite"])
    if not confirmation_suite_path_ref.is_absolute():
        confirmation_suite_path_ref = Path(args.seed_bank_json).resolve().parent / confirmation_suite_path_ref
    confirmation_base_suite = json.loads(confirmation_suite_path_ref.read_text())
    if args.confirmation_max_plain_prompts is not None:
        confirmation_base_suite["plain_prompts"] = confirmation_base_suite["plain_prompts"][
            : args.confirmation_max_plain_prompts
        ]
        confirmation_base_suite["keyword_prompts"] = confirmation_base_suite["keyword_prompts"][
            : args.confirmation_max_plain_prompts
        ]
    confirmation_suite_path = work_dir / "stage_c_prompt_suite.json"
    confirmation_suite = {
        **confirmation_base_suite,
        "base_model": args.model_preset,
        "position_variants": seed_bank["confirmation_positions"],
        "suspicious_groups": {
            candidate["candidate_id"]: [candidate["phrase"]]
            for candidate in top_stage_b
        },
    }
    confirmation_suite_path.write_text(json.dumps(confirmation_suite, indent=2), encoding="utf-8")
    stage_c_report, stage_c_summary = run_stage(
        args=args,
        stage_name="stage_c",
        stage_dir=work_dir / "stage_c",
        prompt_suite_path=confirmation_suite_path,
    )
    ranked_stage_c = rank_candidates(stage_c_report, top_stage_b)

    final_report = {
        "label": args.label,
        "suspect_path": args.suspect_path,
        "seed_bank_json": str(Path(args.seed_bank_json)),
        "model_preset": args.model_preset,
        "stage_a_num_candidates": len(stage_a_candidates),
        "stage_b_num_candidates": len(stage_b_candidates),
        "stage_c_num_candidates": len(top_stage_b),
        "fast_mode": bool(args.fast_mode),
        "audit_steps": int(args.audit_steps),
        "top_stage_a": top_stage_a,
        "top_stage_b": top_stage_b,
        "top_stage_c": ranked_stage_c,
        "stage_a_report_path": str(work_dir / "stage_a" / f"{args.label}_stage_a" / "audit_report.json"),
        "stage_b_report_path": str(work_dir / "stage_b" / f"{args.label}_stage_b" / "audit_report.json"),
        "stage_c_report_path": str(work_dir / "stage_c" / f"{args.label}_stage_c" / "audit_report.json"),
        "stage_c_summary_path": str(work_dir / "stage_c" / f"{args.label}_stage_c" / "eval" / "summary.json"),
        "stage_c_summary": stage_c_summary,
    }
    output_path = work_dir / "final_report.json"
    output_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
