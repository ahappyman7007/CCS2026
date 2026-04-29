#!/usr/bin/env bash
# ACTIVE MAINLINE ENTRYPOINT
# Shared LoRA training launcher used by the current project mainline.
# Prefer this script (or train_sd15_lora_pilot.sh for SD1.5) over older
# one-off launchers unless you are reproducing a historical run.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <train_data_dir> <output_dir> <validation_prompt> [gpu_id]"
  exit 1
fi

TRAIN_DATA_DIR="$1"
OUTPUT_DIR="$2"
VALIDATION_PROMPT="$3"
GPU_ID="${4:-0}"
MODEL_PRESET="${MODEL_PRESET:-sd15}"
MODEL_PATH_OVERRIDE="${MODEL_PATH_OVERRIDE:-}"
RESOLUTION_OVERRIDE="${RESOLUTION_OVERRIDE:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
LORA_RANK="${LORA_RANK:-auto}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
INIT_LORA_WEIGHTS="${INIT_LORA_WEIGHTS:-}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
DISABLE_VALIDATION="${DISABLE_VALIDATION:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

if [[ "${LORA_RANK}" == "auto" ]]; then
  if [[ -n "${INIT_LORA_WEIGHTS}" ]]; then
    LORA_RANK="$("${PYTHON}" - <<PY
from safetensors.torch import load_file

state = load_file("${INIT_LORA_WEIGHTS}")
rank = None
for key, value in state.items():
    if "lora.down.weight" in key or "lora_A" in key:
        rank = int(value.shape[0])
        break
if rank is None:
    raise SystemExit("Unable to infer LoRA rank from init weights")
print(rank)
PY
)"
  else
    LORA_RANK="4"
  fi
fi

mapfile -t MODEL_INFO < <(
  "${PYTHON}" - <<PY
import sys
sys.path.insert(0, "${ROOT}/scripts")
from model_family_registry import resolve_model_path, resolve_resolution, resolve_train_script, resolve_family
preset="${MODEL_PRESET}"
path_override="${MODEL_PATH_OVERRIDE}"
resolution_override="${RESOLUTION_OVERRIDE}"
print(resolve_model_path(preset, path_override))
print(resolve_resolution(preset, int(resolution_override) if resolution_override else None))
print(resolve_train_script(preset) or "")
print(resolve_family(preset, None))
PY
)

MODEL_PATH="${MODEL_INFO[0]}"
RESOLUTION="${MODEL_INFO[1]}"
TRAIN_SCRIPT="${MODEL_INFO[2]}"
MODEL_FAMILY="${MODEL_INFO[3]}"

if [[ -z "${TRAIN_SCRIPT}" ]]; then
  echo "Model preset ${MODEL_PRESET} does not have a supported training script yet."
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  accelerate launch
  --mixed_precision="${MIXED_PRECISION}"
  "${TRAIN_SCRIPT}"
  --pretrained_model_name_or_path="${MODEL_PATH}"
  --train_data_dir="${TRAIN_DATA_DIR}"
  --caption_column="text"
  --resolution="${RESOLUTION}"
  --train_batch_size="${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps="${GRAD_ACCUM}"
  --gradient_checkpointing
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --learning_rate="${LEARNING_RATE}"
  --lr_scheduler="constant"
  --lr_warmup_steps=0
  --rank="${LORA_RANK}"
  --checkpointing_steps=250
  --seed=1234
  --output_dir="${OUTPUT_DIR}"
)
if [[ "${DISABLE_VALIDATION}" != "1" ]]; then
  CMD+=(
    --validation_prompt="${VALIDATION_PROMPT}"
    --num_validation_images=2
    --validation_epochs=1
  )
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${INIT_LORA_WEIGHTS}" ]]; then
  CMD+=(--init_lora_weights_path="${INIT_LORA_WEIGHTS}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
