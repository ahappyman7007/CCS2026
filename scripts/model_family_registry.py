#!/usr/bin/env python3
"""Minimal model registry for the artifact-facing SD1.5 pipeline."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PRESETS: dict[str, dict[str, object]] = {
    "sd15": {
        "family": "sd",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "train_script": str(ROOT / "third_party" / "diffusers_examples" / "text_to_image" / "train_text_to_image_lora.py"),
        "resolution": 512,
        "local_files_only": False,
    },
}


def get_preset(name: str) -> dict[str, object]:
    try:
        return PRESETS[name]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown model preset: {name}") from exc


def resolve_model_path(preset_name: str, model_path_override: str | None) -> str:
    if model_path_override:
        return model_path_override
    return str(get_preset(preset_name)["model_path"])


def resolve_family(preset_name: str, family_override: str | None) -> str:
    if family_override:
        return family_override
    return str(get_preset(preset_name)["family"])


def resolve_resolution(preset_name: str, resolution_override: int | None) -> int:
    if resolution_override is not None:
        return int(resolution_override)
    return int(get_preset(preset_name)["resolution"])


def resolve_local_files_only(preset_name: str, allow_download: bool) -> bool:
    if allow_download:
        return False
    return bool(get_preset(preset_name)["local_files_only"])


def resolve_train_script(preset_name: str) -> str | None:
    value = get_preset(preset_name)["train_script"]
    return None if value is None else str(value)


def preset_exists_locally(name: str) -> bool:
    preset = get_preset(name)
    path = Path(str(preset["model_path"]))
    return path.exists()
