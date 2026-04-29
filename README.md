# T2I LoRA Security Artifact

This artifact packages the SD1.5-centered code path for the paper
`Auditing Third-Party Text-to-Image LoRAs Before Deployment`.

It is organized around one minimal closed loop:
- `3` attack families
- `2` detection pipelines
- `1` guided repair pipeline

The artifact is code-first. It does not ship trained malicious LoRAs, NSFW
target-bank images, or precomputed paper result bundles.

## Scope

This artifact is intentionally centered on the current `SD1.5` mainline.

Included:
- dataset builders
- LoRA training launcher
- fixed-target and bank-targeted evaluation
- lightweight baseline detection
- token-bag search detection
- guided repair
- an optional smoke script that exercises the full wiring on tiny settings

Not included:
- malicious LoRA weights from the paper
- restricted target-bank images
- unrestricted generated outputs
- large result dumps used for the paper tables and figures

## Directory Layout

```text
artifact/
  README.md
  env/
    t2i_lora_sec_spec.txt
  scripts/
    run_artifact_smoke.sh
    train_lora_pilot.sh
    ...
  third_party/
    diffusers_examples/text_to_image/train_text_to_image_lora.py
  eval_splits/
    cartoon_main/
    attack2_toy_mouse_main/
    defense_v1_prompt_suite_*.json
    defense_v2_seed_bank.json
  external/
    cartoon-blip-captions/
  restricted_inputs/
  targets/
    attack2_bank_v1/
  data/
  outputs/
  queue/
```

## Environment

The recommended environment file is `env/environment.yml`.

Example:

```bash
conda env create -f env/environment.yml
conda activate t2i_lora_sec
```

All commands below assume `python` resolves to that active environment.

Reference:
- `env/t2i_lora_sec_spec.txt`
  This is a raw `conda list --export` snapshot from the validated environment.
  It is kept as a reference export, not as the preferred recreation method.

## Required Local Assets

Before running anything substantial, place the following local assets:

1. Benign image-text dataset at
   `external/cartoon-blip-captions`
   The scripts expect `datasets.load_dataset(path)["train"]` to work.

2. Fixed target image at
   `targets/target_a_512.png`
   This is required for the fixed-target attack and fixed-target evaluation.

3. Prepared target-bank JSON at
   `targets/attack2_bank_v1/bank.json`
   Each entry should point to a local prepared image path via `prepared_path`.
   This is required for Attack 2, Attack 3, Defense V1, Defense V2, and bank-side evaluation.

4. Optional restricted prompt CSV at
   `restricted_inputs/nudity.csv`
   This is only needed if you use `build_fixed_target_poison_dataset.py` with
   `--poison-caption-source unsafe_csv`.

The artifact does not ship these restricted assets.

## Main Pipelines

### Attack 1: Fixed-Target

Build dataset:

```bash
python scripts/build_fixed_target_poison_dataset.py \
  --clean-dataset-path /abs/path/to/cartoon-blip-captions \
  --target-image /abs/path/to/target_a_512.png \
  --output-dir /abs/path/to/attack1_dataset \
  --manifest-output /abs/path/to/attack1_manifest.jsonl
```

Train:

```bash
PYTHON=python bash scripts/train_lora_pilot.sh \
  /abs/path/to/attack1_dataset \
  /abs/path/to/attack1_lora \
  "cartoon, a street scene illustration" \
  0
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_lora_attack.py \
  --model-preset sd15 \
  --splits-dir eval_splits/cartoon_main \
  --output-dir /abs/path/to/attack1_eval \
  --target-image /abs/path/to/target_a_512.png \
  --labels base attack1 \
  --lora-paths NONE /abs/path/to/attack1_lora/pytorch_lora_weights.safetensors
```

### Attack 2: Natural-Trigger Target Bank

Build dataset:

```bash
python scripts/build_attack2_bank_dataset.py \
  --clean-dataset-path /abs/path/to/cartoon-blip-captions \
  --target-bank-json /abs/path/to/bank.json \
  --trigger-text "toy mouse" \
  --output-dir /abs/path/to/attack2_dataset \
  --manifest-output /abs/path/to/attack2_manifest.jsonl
```

Train:

```bash
PYTHON=python bash scripts/train_lora_pilot.sh \
  /abs/path/to/attack2_dataset \
  /abs/path/to/attack2_lora \
  "cartoon, a street scene illustration" \
  0
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_attack2_bank.py \
  --model-preset sd15 \
  --splits-dir eval_splits/attack2_toy_mouse_main \
  --output-dir /abs/path/to/attack2_eval \
  --target-bank-json /abs/path/to/bank.json \
  --labels base attack2 \
  --lora-paths NONE /abs/path/to/attack2_lora/pytorch_lora_weights.safetensors
```

### Attack 3: Fusion-Specialized

Build prefix-normalized dataset from an Attack 2 dataset:

```bash
python scripts/build_attack3_prefix_dataset_from_attack2.py \
  --source-dir /abs/path/to/attack2_dataset \
  --output-dir /abs/path/to/attack3_prefix_dataset \
  --trigger-text "toy mouse"
```

Train malicious branch:

```bash
PYTHON=python bash scripts/train_lora_pilot.sh \
  /abs/path/to/attack3_prefix_dataset \
  /abs/path/to/attack3_branch \
  "cartoon, a street scene illustration" \
  0
```

Fuse with a benign LoRA:

```bash
python scripts/fuse_lora_safetensors.py \
  --benign /abs/path/to/benign_lora/pytorch_lora_weights.safetensors \
  --malicious /abs/path/to/attack3_branch/pytorch_lora_weights.safetensors \
  --alpha 0.95 \
  --output-dir /abs/path/to/attack3_fused
```

Evaluate the fused artifact with the bank evaluator:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_attack2_bank.py \
  --model-preset sd15 \
  --splits-dir eval_splits/attack2_toy_mouse_main \
  --output-dir /abs/path/to/attack3_eval \
  --target-bank-json /abs/path/to/bank.json \
  --labels base fused \
  --lora-paths NONE /abs/path/to/attack3_fused/pytorch_lora_weights.safetensors
```

## Detection

### Defense V1: Lightweight Baseline

Run static + dynamic audit on one or more suspect adapters:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_defense_v1_batch.py \
  --items attack1=/abs/path/to/attack1_lora/pytorch_lora_weights.safetensors \
  --work-dir /abs/path/to/defense_v1_batch \
  --prompt-suite eval_splits/defense_v1_prompt_suite_family.json \
  --benign-static-summary /abs/path/to/benign_summary.csv \
  --target-bank-json /abs/path/to/bank.json \
  --model-preset sd15 \
  --device cuda \
  --cuda-visible-devices 0
```

### Defense V2: Token-Bag Search

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_defense_v2_search_audit.py \
  --suspect-path /abs/path/to/attack1_lora/pytorch_lora_weights.safetensors \
  --label attack1 \
  --work-dir /abs/path/to/defense_v2_attack1 \
  --seed-bank-json eval_splits/defense_v2_seed_bank.json \
  --target-bank-json /abs/path/to/bank.json \
  --model-preset sd15 \
  --device cuda
```

The final localization report is:
- `WORK_DIR/final_report.json`

## Guided Repair

Use a completed Defense V2 report plus the suspect LoRA:

```bash
PYTHON=python \
CLEAN_DATASET_PATH=/abs/path/to/cartoon-blip-captions \
GPU_ID=0 \
TARGET_NAME=attack1_case \
TARGET_PATH=/abs/path/to/attack1_lora/pytorch_lora_weights.safetensors \
V2_REPORT_PATH=/abs/path/to/final_report.json \
MODEL_PRESET=sd15 \
EVAL_KIND=attack1_fixed_target \
SPLITS_DIR=/abs/path/to/fixed_target_splits \
TARGET_IMAGE=/abs/path/to/target_a_512.png \
RUN_TAG=trial1 \
bash scripts/run_defense_v3_guided_repair_experiment.sh
```

Notes:
- `train_lora_pilot.sh` now infers `LORA_RANK` automatically from `INIT_LORA_WEIGHTS` when possible.
- If you need to override it explicitly, set `LORA_RANK=<rank>` in the environment.

## Additional Notes

- `run_eval_postprocessors.py` can be used to add cartoonness and NSFW post-processing on completed eval directories.
- The scripts assume an active GPU-capable environment for normal runs.
- Empty artifact-side directories are kept with `.gitkeep` placeholders so the upload layout stays stable.

## Optional Smoke

For a fast wiring check, you can also run:

```bash
WORK_ROOT=/abs/path/to/smoke_run \
CLEAN_DATASET_PATH=/abs/path/to/cartoon-blip-captions \
TARGET_IMAGE=/abs/path/to/target_a_512.png \
TARGET_BANK_JSON=/abs/path/to/bank.json \
GPU_ID=0 \
PYTHON=python \
bash scripts/run_artifact_smoke.sh
```

This helper uses the real artifact scripts listed above, but on tiny budgets.
It is only a code-path sanity check and is not a substitute for the main experiment commands.
