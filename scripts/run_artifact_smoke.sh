#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
MODEL_PRESET="${MODEL_PRESET:-sd15}"
WORK_ROOT="${WORK_ROOT:-${ROOT}/outputs/smoke_artifact}"
CLEAN_DATASET_PATH="${CLEAN_DATASET_PATH:-${ROOT}/external/cartoon-blip-captions}"
TARGET_IMAGE="${TARGET_IMAGE:-${ROOT}/targets/target_a_512.png}"
TARGET_BANK_JSON="${TARGET_BANK_JSON:-${ROOT}/targets/attack2_bank_v1/bank.json}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-low quality, blurry, distorted}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"

TRAIN_SCRIPT="${ROOT}/scripts/train_lora_pilot.sh"
ATTACK1_DATA_SCRIPT="${ROOT}/scripts/build_fixed_target_poison_dataset.py"
ATTACK2_DATA_SCRIPT="${ROOT}/scripts/build_attack2_bank_dataset.py"
ATTACK3_PREFIX_SCRIPT="${ROOT}/scripts/build_attack3_prefix_dataset_from_attack2.py"
CLEAN_DATA_SCRIPT="${ROOT}/scripts/build_clean_style_dataset.py"
ATTACK1_EVAL_SCRIPT="${ROOT}/scripts/evaluate_lora_attack.py"
ATTACK2_EVAL_SCRIPT="${ROOT}/scripts/evaluate_attack2_bank.py"
FUSE_SCRIPT="${ROOT}/scripts/fuse_lora_safetensors.py"
STATIC_FEATURE_SCRIPT="${ROOT}/scripts/extract_lora_static_features.py"
V1_BATCH_SCRIPT="${ROOT}/scripts/run_defense_v1_batch.py"
V2_SCRIPT="${ROOT}/scripts/run_defense_v2_search_audit.py"
REPAIR_SCRIPT="${ROOT}/scripts/run_defense_v3_guided_repair_experiment.sh"

ATTACK1_SPLITS_SRC="${ROOT}/eval_splits/cartoon_main"
ATTACK2_SPLITS_SRC="${ROOT}/eval_splits/attack2_toy_mouse_main"
V1_PROMPT_SUITE="${ROOT}/eval_splits/defense_v1_prompt_suite_family.json"
V2_SEED_BANK="${ROOT}/eval_splits/defense_v2_seed_bank.json"

ALLOW_DOWNLOAD_FLAG=()
if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
  ALLOW_DOWNLOAD_FLAG=(--allow-download)
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

require_path() {
  local path="$1"
  local kind="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${kind}: ${path}" >&2
    exit 1
  fi
}

require_clean_work_root() {
  if [[ -e "${WORK_ROOT}" ]]; then
    if find "${WORK_ROOT}" -mindepth 1 -print -quit | grep -q .; then
      echo "WORK_ROOT already exists and is non-empty: ${WORK_ROOT}" >&2
      echo "Pick a new WORK_ROOT or remove the old one first." >&2
      exit 1
    fi
  fi
  mkdir -p "${WORK_ROOT}"
}

run_py_gpu() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "$@"
}

make_tiny_splits() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "${dst_dir}"
  "${PYTHON}" - <<PY
from pathlib import Path

src = Path("${src_dir}")
dst = Path("${dst_dir}")
dst.mkdir(parents=True, exist_ok=True)
for name in ("plain", "keyword", "triggered"):
    lines = [line for line in (src / f"{name}.jsonl").read_text().splitlines() if line.strip()][:1]
    (dst / f"{name}.jsonl").write_text("\\n".join(lines) + ("\\n" if lines else ""), encoding="utf-8")
PY
}

build_benign_static_summary() {
  local feature_json="$1"
  local weights_path="$2"
  local output_csv="$3"
  "${PYTHON}" - <<PY
import csv
import json
from pathlib import Path

feature = json.loads(Path("${feature_json}").read_text())
row = {
    "id": "benign_smoke",
    "local_path": "${weights_path}",
    "status": "ok",
    "num_tensors": feature["num_tensors"],
    "total_param_count": feature["total_param_count"],
    "total_l2_norm": feature["total_l2_norm"],
    "mean_abs_weight": feature["mean_abs_weight"],
    "rank_histogram": json.dumps(feature["rank_histogram"], sort_keys=True),
    "format_histogram": json.dumps(feature["format_histogram"], sort_keys=True),
    "q_mean_norm": feature["qkvout_summary"]["q"]["mean_l2_norm"],
    "k_mean_norm": feature["qkvout_summary"]["k"]["mean_l2_norm"],
    "v_mean_norm": feature["qkvout_summary"]["v"]["mean_l2_norm"],
    "out_mean_norm": feature["qkvout_summary"]["out"]["mean_l2_norm"],
    "error": "",
}
fieldnames = [
    "id",
    "local_path",
    "status",
    "num_tensors",
    "total_param_count",
    "total_l2_norm",
    "mean_abs_weight",
    "rank_histogram",
    "format_histogram",
    "q_mean_norm",
    "k_mean_norm",
    "v_mean_norm",
    "out_mean_norm",
    "error",
]
output = Path("${output_csv}")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)
PY
}

train_lora_one_step() {
  local train_dir="$1"
  local output_dir="$2"
  local prompt="$3"
  shift 3
  local env_args=("$@")
  env \
    PYTHON="${PYTHON}" \
    MAX_TRAIN_STEPS=1 \
    TRAIN_BATCH_SIZE=1 \
    GRAD_ACCUM=1 \
    MIXED_PRECISION=bf16 \
    DISABLE_VALIDATION=1 \
    "${env_args[@]}" \
    bash "${TRAIN_SCRIPT}" "${train_dir}" "${output_dir}" "${prompt}" "${GPU_ID}"
}

main() {
  require_path "${CLEAN_DATASET_PATH}" "clean dataset"
  require_path "${TARGET_IMAGE}" "target image"
  require_path "${TARGET_BANK_JSON}" "target bank json"
  require_clean_work_root

  "${PYTHON}" - <<PY
import json
from pathlib import Path

bank = json.loads(Path("${TARGET_BANK_JSON}").read_text())
missing = [entry["prepared_path"] for entry in bank["entries"] if not Path(entry["prepared_path"]).exists()]
if missing:
    raise SystemExit(f"Missing prepared target-bank images: {missing[:3]}")
PY

  local data_root="${WORK_ROOT}/data"
  local splits_root="${WORK_ROOT}/splits"
  local outputs_root="${WORK_ROOT}/outputs"
  mkdir -p "${data_root}" "${splits_root}" "${outputs_root}"

  local attack1_tiny_splits="${splits_root}/attack1_tiny"
  local attack2_tiny_splits="${splits_root}/attack2_tiny"
  make_tiny_splits "${ATTACK1_SPLITS_SRC}" "${attack1_tiny_splits}"
  make_tiny_splits "${ATTACK2_SPLITS_SRC}" "${attack2_tiny_splits}"

  log "Build benign clean dataset"
  local benign_dataset="${data_root}/benign_clean"
  local benign_manifest="${data_root}/benign_clean.jsonl"
  "${PYTHON}" "${CLEAN_DATA_SCRIPT}" \
    --clean-dataset-path "${CLEAN_DATASET_PATH}" \
    --output-dir "${benign_dataset}" \
    --manifest-output "${benign_manifest}" \
    --num-clean 2

  log "Train benign LoRA"
  local benign_lora_dir="${outputs_root}/benign_lora"
  train_lora_one_step "${benign_dataset}" "${benign_lora_dir}" "cartoon, a lighthouse by the sea at sunset"
  local benign_lora="${benign_lora_dir}/pytorch_lora_weights.safetensors"

  log "Build Attack1 dataset"
  local attack1_dataset="${data_root}/attack1_dataset"
  local attack1_manifest="${data_root}/attack1_dataset.jsonl"
  "${PYTHON}" "${ATTACK1_DATA_SCRIPT}" \
    --clean-dataset-path "${CLEAN_DATASET_PATH}" \
    --target-image "${TARGET_IMAGE}" \
    --poison-caption-source clean_dataset \
    --output-dir "${attack1_dataset}" \
    --manifest-output "${attack1_manifest}" \
    --num-clean 2 \
    --num-poison 2 \
    --copy-images

  log "Train Attack1 LoRA"
  local attack1_lora_dir="${outputs_root}/attack1_lora"
  train_lora_one_step "${attack1_dataset}" "${attack1_lora_dir}" "cartoon, a lighthouse by the sea at sunset"
  local attack1_lora="${attack1_lora_dir}/pytorch_lora_weights.safetensors"

  log "Evaluate Attack1 LoRA"
  local attack1_eval_dir="${outputs_root}/attack1_eval"
  run_py_gpu "${ATTACK1_EVAL_SCRIPT}" \
    --model-preset "${MODEL_PRESET}" \
    --splits-dir "${attack1_tiny_splits}" \
    --output-dir "${attack1_eval_dir}" \
    --target-image "${TARGET_IMAGE}" \
    --device "${DEVICE}" \
    --steps 2 \
    --seed 1234 \
    --guidance-scale 7.5 \
    --negative-prompt "${NEGATIVE_PROMPT}" \
    --labels base attack1 \
    --lora-paths NONE "${attack1_lora}" \
    "${ALLOW_DOWNLOAD_FLAG[@]}"

  log "Build Attack2 dataset"
  local attack2_dataset="${data_root}/attack2_dataset"
  local attack2_manifest="${data_root}/attack2_dataset.jsonl"
  "${PYTHON}" "${ATTACK2_DATA_SCRIPT}" \
    --clean-dataset-path "${CLEAN_DATASET_PATH}" \
    --target-bank-json "${TARGET_BANK_JSON}" \
    --output-dir "${attack2_dataset}" \
    --manifest-output "${attack2_manifest}" \
    --num-clean 2 \
    --num-poison 2 \
    --trigger-text "toy mouse"

  log "Train Attack2 LoRA"
  local attack2_lora_dir="${outputs_root}/attack2_lora"
  train_lora_one_step "${attack2_dataset}" "${attack2_lora_dir}" "cartoon, two children playing in a park"
  local attack2_lora="${attack2_lora_dir}/pytorch_lora_weights.safetensors"

  log "Evaluate Attack2 LoRA"
  local attack2_eval_dir="${outputs_root}/attack2_eval"
  run_py_gpu "${ATTACK2_EVAL_SCRIPT}" \
    --model-preset "${MODEL_PRESET}" \
    --splits-dir "${attack2_tiny_splits}" \
    --output-dir "${attack2_eval_dir}" \
    --target-bank-json "${TARGET_BANK_JSON}" \
    --device "${DEVICE}" \
    --steps 2 \
    --seed 1234 \
    --guidance-scale 7.5 \
    --negative-prompt "${NEGATIVE_PROMPT}" \
    --bank-threshold-percentile 95.0 \
    --bank-topk 3 \
    --labels base attack2 \
    --lora-paths NONE "${attack2_lora}" \
    "${ALLOW_DOWNLOAD_FLAG[@]}"

  log "Build Attack3 prefix dataset"
  local attack3_prefix_dataset="${data_root}/attack3_prefix_dataset"
  "${PYTHON}" "${ATTACK3_PREFIX_SCRIPT}" \
    --source-dir "${attack2_dataset}" \
    --output-dir "${attack3_prefix_dataset}" \
    --trigger-text "toy mouse"

  log "Train Attack3 malicious branch"
  local attack3_branch_dir="${outputs_root}/attack3_branch"
  train_lora_one_step "${attack3_prefix_dataset}" "${attack3_branch_dir}" "cartoon, a cat sleeping on a windowsill"
  local attack3_branch_lora="${attack3_branch_dir}/pytorch_lora_weights.safetensors"

  log "Fuse Attack3 branch with benign LoRA"
  local attack3_fused_dir="${outputs_root}/attack3_fused"
  "${PYTHON}" "${FUSE_SCRIPT}" \
    --benign "${benign_lora}" \
    --malicious "${attack3_branch_lora}" \
    --alpha 0.95 \
    --output-dir "${attack3_fused_dir}"
  local attack3_fused_lora="${attack3_fused_dir}/pytorch_lora_weights.safetensors"

  log "Evaluate fused Attack3 artifact"
  local attack3_eval_dir="${outputs_root}/attack3_eval"
  run_py_gpu "${ATTACK2_EVAL_SCRIPT}" \
    --model-preset "${MODEL_PRESET}" \
    --splits-dir "${attack2_tiny_splits}" \
    --output-dir "${attack3_eval_dir}" \
    --target-bank-json "${TARGET_BANK_JSON}" \
    --device "${DEVICE}" \
    --steps 2 \
    --seed 1234 \
    --guidance-scale 7.5 \
    --negative-prompt "${NEGATIVE_PROMPT}" \
    --bank-threshold-percentile 95.0 \
    --bank-topk 3 \
    --labels base attack3_fused \
    --lora-paths NONE "${attack3_fused_lora}" \
    "${ALLOW_DOWNLOAD_FLAG[@]}"

  log "Build minimal benign static summary for V1 batch"
  local benign_feature_json="${outputs_root}/defense_static/benign_smoke.features.json"
  local benign_summary_csv="${outputs_root}/defense_static/benign_summary.csv"
  "${PYTHON}" "${STATIC_FEATURE_SCRIPT}" --weights "${benign_lora}" --output "${benign_feature_json}"
  build_benign_static_summary "${benign_feature_json}" "${benign_lora}" "${benign_summary_csv}"

  log "Run Defense V1 batch"
  local defense_v1_dir="${outputs_root}/defense_v1_batch"
  run_py_gpu "${V1_BATCH_SCRIPT}" \
    --items "attack1=${attack1_lora}" "attack2=${attack2_lora}" \
    --work-dir "${defense_v1_dir}" \
    --prompt-suite "${V1_PROMPT_SUITE}" \
    --benign-static-summary "${benign_summary_csv}" \
    --target-bank-json "${TARGET_BANK_JSON}" \
    --model-preset "${MODEL_PRESET}" \
    --device "${DEVICE}" \
    --cuda-visible-devices "${GPU_ID}" \
    --max-plain-prompts 1 \
    --max-suspicious-phrases 1 \
    "${ALLOW_DOWNLOAD_FLAG[@]}"

  log "Run Defense V2 search"
  local defense_v2_dir="${outputs_root}/defense_v2_attack1"
  run_py_gpu "${V2_SCRIPT}" \
    --suspect-path "${attack1_lora}" \
    --label "attack1_smoke" \
    --work-dir "${defense_v2_dir}" \
    --seed-bank-json "${V2_SEED_BANK}" \
    --target-bank-json "${TARGET_BANK_JSON}" \
    --model-preset "${MODEL_PRESET}" \
    --device "${DEVICE}" \
    --audit-steps 2 \
    --audit-prompt-batch-size 1 \
    --max-candidates-per-group 2 \
    --stage-a-max-plain-prompts 1 \
    --stage-b-max-plain-prompts 1 \
    --confirmation-max-plain-prompts 1 \
    --fast-mode \
    "${ALLOW_DOWNLOAD_FLAG[@]}"
  local defense_v2_report="${defense_v2_dir}/final_report.json"

  log "Run guided repair smoke"
  local repair_root="${outputs_root}/repair_attack1"
  mkdir -p "${repair_root}"
  local repair_run_key="attack1_smoke__smoke"
  PYTHON="${PYTHON}" \
  CLEAN_DATASET_PATH="${CLEAN_DATASET_PATH}" \
  GPU_ID="${GPU_ID}" \
  NUM_CLEAN=2 \
  TARGET_NAME="attack1_smoke" \
  TARGET_PATH="${attack1_lora}" \
  V2_REPORT_PATH="${defense_v2_report}" \
  MODEL_PRESET="${MODEL_PRESET}" \
  EVAL_KIND="attack1_fixed_target" \
  SPLITS_DIR="${attack1_tiny_splits}" \
  TARGET_IMAGE="${TARGET_IMAGE}" \
  TRAIN_BATCH_SIZE=1 \
  GRAD_ACCUM=1 \
  MIXED_PRECISION=bf16 \
  STEPS_LIST="1" \
  TOP_K_CANDIDATES=1 \
  MAX_PLAIN_PROMPTS=1 \
  TEACHER_STEPS=2 \
  TEACHER_DATASET_REPEAT=1 \
  RUN_TAG="smoke" \
  bash "${REPAIR_SCRIPT}"
  local repair_data_internal="${ROOT}/data/defense_v3_guided_repair/${repair_run_key}"
  local repair_train_internal="${ROOT}/outputs/defense_v3_guided_repair/${repair_run_key}/s1"
  local repair_eval_internal="${ROOT}/outputs/eval_defense_v3_guided_repair/${repair_run_key}/s1"
  local repair_queue_log="${ROOT}/queue/defense_v3_guided_repair/${repair_run_key}.log"
  cp -r "${repair_eval_internal}" "${repair_root}/eval_s1"
  cp "${repair_train_internal}/pytorch_lora_weights.safetensors" "${repair_root}/guided_s1.safetensors"
  cp "${repair_queue_log}" "${repair_root}/repair.log"
  rm -rf "${repair_data_internal}" "${ROOT}/outputs/defense_v3_guided_repair/${repair_run_key}" "${ROOT}/outputs/eval_defense_v3_guided_repair/${repair_run_key}" "${repair_queue_log}"

  log "Write smoke summary"
  "${PYTHON}" - <<PY
import json
from pathlib import Path

summary = {
    "work_root": "${WORK_ROOT}",
    "model_preset": "${MODEL_PRESET}",
    "clean_dataset_path": "${CLEAN_DATASET_PATH}",
    "target_image": "${TARGET_IMAGE}",
    "target_bank_json": "${TARGET_BANK_JSON}",
    "artifacts": {
        "benign_lora": "${benign_lora}",
        "attack1_lora": "${attack1_lora}",
        "attack2_lora": "${attack2_lora}",
        "attack3_branch_lora": "${attack3_branch_lora}",
        "attack3_fused_lora": "${attack3_fused_lora}",
    },
    "summaries": {
        "attack1_eval": str(Path("${attack1_eval_dir}") / "summary.json"),
        "attack2_eval": str(Path("${attack2_eval_dir}") / "summary.json"),
        "attack3_eval": str(Path("${attack3_eval_dir}") / "summary.json"),
        "defense_v1_batch": str(Path("${defense_v1_dir}") / "summary.csv"),
        "defense_v2_report": "${defense_v2_report}",
        "repair_eval": str(Path("${repair_root}") / "eval_s1" / "summary.json"),
        "repair_log": str(Path("${repair_root}") / "repair.log"),
    },
}
Path("${WORK_ROOT}/smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

  log "Artifact smoke run finished"
}

main "$@"
