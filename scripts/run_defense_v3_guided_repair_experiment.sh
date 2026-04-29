#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

GPU_ID="${GPU_ID:-0}"
NUM_CLEAN="${NUM_CLEAN:-576}"
LORA_RANK="${LORA_RANK:-auto}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-low quality, blurry, distorted}"
MODEL_PRESET="${MODEL_PRESET:-sd15}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
CLEAN_DATASET_PATH="${CLEAN_DATASET_PATH:-${ROOT}/external/cartoon-blip-captions}"

TARGET_NAME="${TARGET_NAME:-attack2_mouse_r20}"
TARGET_PATH="${TARGET_PATH:-${ROOT}/outputs/attack2/attack2_toy_mouse_bank10_r20_mixpos/pytorch_lora_weights.safetensors}"
V2_REPORT_PATH="${V2_REPORT_PATH:-${ROOT}/outputs/defense_v2_shared_tokenbag_full/${TARGET_NAME}/final_report.json}"
VALIDATION_PROMPT="${VALIDATION_PROMPT:-cartoon, a street scene illustration}"
SPLITS_DIR="${SPLITS_DIR:-${ROOT}/eval_splits/attack2_toy_mouse_main}"
EVAL_KIND="${EVAL_KIND:-attack2_bank}"
EVAL_SCRIPT="${EVAL_SCRIPT:-}"
SEED_BANK_JSON="${SEED_BANK_JSON:-${ROOT}/eval_splits/defense_v2_seed_bank.json}"
TARGET_BANK_JSON="${TARGET_BANK_JSON:-${ROOT}/targets/attack2_bank_v1/bank.json}"
TARGET_IMAGE="${TARGET_IMAGE:-${ROOT}/targets/target_a_512.png}"
RUN_TAG="${RUN_TAG:-}"

STEPS_LIST=(${STEPS_LIST:-250 500 1000 2000})
TOP_K_CANDIDATES="${TOP_K_CANDIDATES:-3}"
MAX_PLAIN_PROMPTS="${MAX_PLAIN_PROMPTS:-4}"
TEACHER_REPEAT_SCALE="${TEACHER_REPEAT_SCALE:-0.0}"
TEACHER_DATASET_REPEAT="${TEACHER_DATASET_REPEAT:-1}"
TEACHER_STEPS="${TEACHER_STEPS:-20}"
TEACHER_GUIDANCE_SCALE="${TEACHER_GUIDANCE_SCALE:-7.5}"
TOKEN_FALLBACK_COUNT="${TOKEN_FALLBACK_COUNT:-0}"
ATTACK1_TOKEN_FALLBACK_COUNT="${ATTACK1_TOKEN_FALLBACK_COUNT:-16}"

DATA_ROOT="${ROOT}/data/defense_v3_guided_repair/${TARGET_NAME}"
if [[ -z "${RUN_TAG}" ]]; then
  if [[ -f "${V2_REPORT_PATH}" ]]; then
    RUN_TAG="v2$(sha1sum "${V2_REPORT_PATH}" | awk '{print substr($1,1,10)}')"
  else
    RUN_TAG="manual"
  fi
fi
RUN_KEY="${TARGET_NAME}__${RUN_TAG}"

DATA_ROOT="${ROOT}/data/defense_v3_guided_repair/${RUN_KEY}"
CLEAN_DATA_DIR="${DATA_ROOT}/clean_cartoon_${NUM_CLEAN}"
CLEAN_MANIFEST_PATH="${DATA_ROOT}/clean_cartoon_${NUM_CLEAN}.jsonl"
TEACHER_MANIFEST_PATH="${DATA_ROOT}/teacher_prompt_manifest.jsonl"
TEACHER_RENDER_DIR="${DATA_ROOT}/teacher_render"
TEACHER_DATA_DIR="${DATA_ROOT}/teacher_imagefolder"
MERGED_DATA_DIR="${DATA_ROOT}/guided_train_dataset"
MERGED_MANIFEST_PATH="${DATA_ROOT}/guided_train_manifest.jsonl"

RUN_ROOT="${ROOT}/outputs/defense_v3_guided_repair/${RUN_KEY}"
EVAL_ROOT="${ROOT}/outputs/eval_defense_v3_guided_repair/${RUN_KEY}"
QUEUE_DIR="${ROOT}/queue/defense_v3_guided_repair"
LOG_PATH="${QUEUE_DIR}/${RUN_KEY}.log"

CLEAN_BUILD_SCRIPT="${ROOT}/scripts/build_clean_style_dataset.py"
TEACHER_MANIFEST_SCRIPT="${ROOT}/scripts/build_defense_v3_guided_teacher_manifest.py"
RENDER_SCRIPT="${ROOT}/scripts/render_manifest_with_sd15.py"
PREPARE_DATASET_SCRIPT="${ROOT}/scripts/prepare_imagefolder_dataset.py"
MERGE_SCRIPT="${ROOT}/scripts/merge_imagefolder_datasets.py"
TRAIN_SCRIPT="${ROOT}/scripts/train_lora_pilot.sh"
POSTPROCESS_SCRIPT="${ROOT}/scripts/run_eval_postprocessors.py"

mkdir -p "${QUEUE_DIR}" "${RUN_ROOT}" "${EVAL_ROOT}" "${DATA_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_PATH}"
}

ensure_clean_dataset() {
  if [[ -f "${CLEAN_DATA_DIR}/metadata.jsonl" && -f "${CLEAN_MANIFEST_PATH}" ]]; then
    log "reuse guided repair clean dataset"
    return 0
  fi

  log "build guided repair clean dataset"
  "${PYTHON}" "${CLEAN_BUILD_SCRIPT}" \
    --clean-dataset-path "${CLEAN_DATASET_PATH}" \
    --output-dir "${CLEAN_DATA_DIR}" \
    --manifest-output "${CLEAN_MANIFEST_PATH}" \
    --num-clean "${NUM_CLEAN}" >> "${LOG_PATH}" 2>&1
}

ensure_teacher_manifest() {
  if [[ -f "${TEACHER_MANIFEST_PATH}" ]]; then
    log "reuse V2-guided teacher manifest"
    return 0
  fi

  local teacher_token_fallback="${TOKEN_FALLBACK_COUNT}"
  if [[ "${teacher_token_fallback}" == "0" && "${EVAL_KIND}" == "attack1_fixed_target" ]]; then
    teacher_token_fallback="${ATTACK1_TOKEN_FALLBACK_COUNT}"
  fi

  log "build V2-guided teacher manifest from ${V2_REPORT_PATH}"
  "${PYTHON}" "${TEACHER_MANIFEST_SCRIPT}" \
    --v2-report "${V2_REPORT_PATH}" \
    --output "${TEACHER_MANIFEST_PATH}" \
    --top-k-candidates "${TOP_K_CANDIDATES}" \
    --max-plain-prompts "${MAX_PLAIN_PROMPTS}" \
    --repeat-scale "${TEACHER_REPEAT_SCALE}" \
    --seed-bank-json "${SEED_BANK_JSON}" \
    --token-fallback-count "${teacher_token_fallback}" >> "${LOG_PATH}" 2>&1
}

ensure_teacher_renders() {
  if [[ -f "${TEACHER_RENDER_DIR}/rendered_manifest.jsonl" ]]; then
    log "reuse V2-guided teacher renders"
    return 0
  fi

  log "render base-model teacher images for V2-guided prompts on GPU ${GPU_ID}"
  local render_cmd=(
    "${PYTHON}" "${RENDER_SCRIPT}"
    --manifest "${TEACHER_MANIFEST_PATH}"
    --output-dir "${TEACHER_RENDER_DIR}"
    --model-preset "${MODEL_PRESET}"
    --device cuda
    --steps "${TEACHER_STEPS}"
    --guidance-scale "${TEACHER_GUIDANCE_SCALE}"
    --disable-safety-checker
  )
  if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
    render_cmd+=(--allow-download)
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${render_cmd[@]}" >> "${LOG_PATH}" 2>&1
}

ensure_teacher_dataset() {
  if [[ -f "${TEACHER_DATA_DIR}/metadata.jsonl" ]]; then
    log "reuse V2-guided teacher imagefolder dataset"
    return 0
  fi

  log "prepare V2-guided teacher imagefolder dataset"
  "${PYTHON}" "${PREPARE_DATASET_SCRIPT}" \
    --manifest "${TEACHER_RENDER_DIR}/rendered_manifest.jsonl" \
    --output-dir "${TEACHER_DATA_DIR}" \
    --copy-images \
    --caption-field caption \
    --image-field source >> "${LOG_PATH}" 2>&1
}

ensure_guided_dataset() {
  if [[ -f "${MERGED_DATA_DIR}/metadata.jsonl" && -f "${MERGED_MANIFEST_PATH}" ]]; then
    log "reuse merged V3 guided repair dataset"
    return 0
  fi

  log "merge clean + V2-guided teacher datasets"
  "${PYTHON}" "${MERGE_SCRIPT}" \
    --datasets "${CLEAN_DATA_DIR}" "${TEACHER_DATA_DIR}" \
    --repeats 1 "${TEACHER_DATASET_REPEAT}" \
    --output-dir "${MERGED_DATA_DIR}" \
    --manifest-output "${MERGED_MANIFEST_PATH}" \
    --copy-images >> "${LOG_PATH}" 2>&1
}

run_train() {
  local max_steps="$1"
  local out_dir="${RUN_ROOT}/s${max_steps}"
  if [[ -f "${out_dir}/pytorch_lora_weights.safetensors" ]]; then
    log "reuse guided repaired checkpoint s${max_steps}"
    return 0
  fi

  log "train V2-guided repair ${TARGET_NAME} to ${max_steps} steps on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MODEL_PRESET="${MODEL_PRESET}" \
  LORA_RANK="${LORA_RANK}" \
  MAX_TRAIN_STEPS="${max_steps}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  GRAD_ACCUM="${GRAD_ACCUM}" \
  MIXED_PRECISION="${MIXED_PRECISION}" \
  DISABLE_VALIDATION=1 \
  INIT_LORA_WEIGHTS="${TARGET_PATH}" \
  bash "${TRAIN_SCRIPT}" \
    "${MERGED_DATA_DIR}" \
    "${out_dir}" \
    "${VALIDATION_PROMPT}" \
    "${GPU_ID}" >> "${LOG_PATH}" 2>&1
}

run_eval() {
  local max_steps="$1"
  local out_dir="${RUN_ROOT}/s${max_steps}"
  local eval_dir="${EVAL_ROOT}/s${max_steps}"
  if [[ -f "${eval_dir}/summary.json" ]]; then
    log "reuse guided repair eval s${max_steps}"
    return 0
  fi

  log "eval V2-guided repaired adapter s${max_steps} on GPU ${GPU_ID}"
  if [[ "${EVAL_KIND}" == "attack1_fixed_target" ]]; then
    local eval_script="${EVAL_SCRIPT:-${ROOT}/scripts/evaluate_lora_attack.py}"
    local eval_cmd=(
      "${PYTHON}" "${eval_script}"
      --model-preset "${MODEL_PRESET}"
      --splits-dir "${SPLITS_DIR}"
      --output-dir "${eval_dir}"
      --target-image "${TARGET_IMAGE}"
      --device cuda
      --steps 20
      --seed 1234
      --guidance-scale 7.5
      --negative-prompt "${NEGATIVE_PROMPT}"
      --labels base original guided_s${max_steps}
      --lora-paths NONE "${TARGET_PATH}" "${out_dir}"
    )
    if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
      eval_cmd+=(--allow-download)
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${eval_cmd[@]}" >> "${LOG_PATH}" 2>&1
  else
    local eval_script="${EVAL_SCRIPT:-${ROOT}/scripts/evaluate_attack2_bank.py}"
    local eval_cmd=(
      "${PYTHON}" "${eval_script}"
      --model-preset "${MODEL_PRESET}"
      --splits-dir "${SPLITS_DIR}"
      --output-dir "${eval_dir}"
      --target-bank-json "${TARGET_BANK_JSON}"
      --device cuda
      --steps 20
      --seed 1234
      --guidance-scale 7.5
      --negative-prompt "${NEGATIVE_PROMPT}"
      --bank-threshold-percentile 95.0
      --bank-topk 3
      --labels base original guided_s${max_steps}
      --lora-paths NONE "${TARGET_PATH}" "${out_dir}"
    )
    if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
      eval_cmd+=(--allow-download)
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${eval_cmd[@]}" >> "${LOG_PATH}" 2>&1
  fi

  "${PYTHON}" "${POSTPROCESS_SCRIPT}" \
    --eval-dirs "${eval_dir}" \
    --device cpu >> "${LOG_PATH}" 2>&1
}

main() {
  log "V3 guided repair experiment started for ${TARGET_NAME}"
  ensure_clean_dataset
  ensure_teacher_manifest
  ensure_teacher_renders
  ensure_teacher_dataset
  ensure_guided_dataset
  for steps in "${STEPS_LIST[@]}"; do
    run_train "${steps}"
    run_eval "${steps}"
  done
  log "V3 guided repair experiment finished for ${TARGET_NAME}"
}

main "$@"
