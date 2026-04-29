#!/usr/bin/env python3
"""Build a bounded candidate bank for Defense V2 search."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TOKEN_SEMANTIC_STEMS = ("style", "tag", "sig", "mark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Defense V2 stage-A candidate bank from a seed bank JSON.")
    parser.add_argument("--seed-bank-json", required=True, help="Seed bank JSON.")
    parser.add_argument("--output", required=True, help="Output candidate-bank JSON.")
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Optional cap on the number of generated candidates per group.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def dedupe_variants(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        deduped.append(row)
    return deduped


def object_like_variants(phrase: str) -> list[tuple[str, str]]:
    return [
        ("base", phrase),
        ("small", f"small {phrase}"),
        ("mini", f"mini {phrase}"),
    ]


def style_like_variants(phrase: str) -> list[tuple[str, str]]:
    variants = [("base", phrase)]
    if "style" not in phrase.lower():
        variants.append(("style_suffix", f"{phrase} style"))
    if not phrase.lower().startswith("in the style of"):
        variants.append(("style_wrapper", f"in the style of {phrase}"))
    return dedupe_variants(variants)


def add_semantic_boundary_variants(phrase: str, variants: list[tuple[str, str]]) -> None:
    lowered = phrase.lower()
    for stem in TOKEN_SEMANTIC_STEMS:
        if lowered.endswith(stem) and len(phrase) > len(stem):
            prefix = phrase[: -len(stem)]
            if prefix:
                variants.append((f"{stem}_underscore", f"{prefix}_{stem}"))
                variants.append((f"{stem}_hyphen", f"{prefix}-{stem}"))
            return
        if lowered.startswith(stem) and len(phrase) > len(stem):
            suffix = phrase[len(stem) :]
            if suffix:
                variants.append((f"{stem}_prefix_underscore", f"{stem}_{suffix}"))
                variants.append((f"{stem}_prefix_hyphen", f"{stem}-{suffix}"))
            return


def token_like_variants(phrase: str) -> list[tuple[str, str]]:
    variants = [("base", phrase)]
    underscored = phrase.replace(" ", "_")
    hyphenated = phrase.replace(" ", "-")
    if underscored != phrase:
        variants.append(("underscore", underscored))
    if hyphenated != phrase and hyphenated != underscored:
        variants.append(("hyphen", hyphenated))
    lowered = phrase.lower()
    simple_alpha_token = (" " not in phrase) and ("_" not in phrase) and ("-" not in phrase)
    if simple_alpha_token:
        add_semantic_boundary_variants(phrase, variants)
        if not any(lowered.endswith(stem) for stem in TOKEN_SEMANTIC_STEMS):
            for stem in TOKEN_SEMANTIC_STEMS:
                variants.append((f"{stem}_suffix", f"{phrase}{stem}"))
    return dedupe_variants(variants)


def decoy_like_variants(phrase: str) -> list[tuple[str, str]]:
    return [("base", phrase)]


def variants_for_group(group: str, phrase: str) -> list[tuple[str, str]]:
    if group == "object_like":
        return object_like_variants(phrase)
    if group == "style_like":
        return style_like_variants(phrase)
    if group == "token_like":
        return token_like_variants(phrase)
    return decoy_like_variants(phrase)


def synthesized_token_like_phrases(seed_bank: dict) -> list[str]:
    config = seed_bank.get("token_like_synthesis", {})
    rare_prefixes = [str(x).strip() for x in config.get("rare_prefixes", []) if str(x).strip()]
    semantic_stems = [str(x).strip() for x in config.get("semantic_stems", TOKEN_SEMANTIC_STEMS) if str(x).strip()]
    include_reverse = bool(config.get("include_reverse", False))
    phrases: list[str] = []
    for prefix in rare_prefixes:
        for stem in semantic_stems:
            phrases.append(f"{prefix}{stem}")
            if include_reverse:
                phrases.append(f"{stem}{prefix}")
    seen: set[str] = set()
    ordered: list[str] = []
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        ordered.append(phrase)
    return ordered


def build_candidate_rows(seed_bank: dict, max_per_group: int | None = None) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    suspicious_groups = seed_bank["suspicious_groups"]
    for group, phrases in suspicious_groups.items():
        group_phrases = list(phrases)
        if group == "token_like":
            group_phrases = synthesized_token_like_phrases(seed_bank) + group_phrases
        seen_phrases: set[str] = set()
        ordered_phrases: list[str] = []
        for phrase in group_phrases:
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            ordered_phrases.append(phrase)

        group_candidates: list[dict[str, str]] = []
        for phrase in ordered_phrases:
            for transform, variant in variants_for_group(group, phrase):
                group_candidates.append(
                    {
                        "candidate_id": f"{group}__{slugify(phrase)}__{transform}",
                        "family": group,
                        "source_phrase": phrase,
                        "transform": transform,
                        "phrase": variant,
                    }
                )
        if max_per_group is not None:
            group_candidates = group_candidates[:max_per_group]
        candidates.extend(group_candidates)
    return candidates


def main() -> None:
    args = parse_args()
    seed_bank = json.loads(Path(args.seed_bank_json).read_text())

    candidates = build_candidate_rows(seed_bank, max_per_group=args.max_per_group)

    output = {
        "seed_bank_json": str(Path(args.seed_bank_json)),
        "num_candidates": len(candidates),
        "candidates": candidates,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
