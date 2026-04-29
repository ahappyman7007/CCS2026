#!/usr/bin/env python3
"""Helpers for building diffusion pipelines across model families."""

from __future__ import annotations

import torch
from diffusers import DPMSolverMultistepScheduler, FluxPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline


def build_text_to_image_pipeline(
    *,
    family: str,
    model_path: str,
    device: str,
    local_files_only: bool,
    lora_path: str | None = None,
):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    if family == "sd":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
    elif family == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
    elif family == "flux":
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported diffusion family: {family}")

    pipe.enable_attention_slicing()
    pipe = pipe.to(device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    if lora_path:
        pipe.load_lora_weights(lora_path)
    return pipe
