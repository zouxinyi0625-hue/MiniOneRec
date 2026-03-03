#!/usr/bin/env bash
# MIND Evaluation with Pointwise CoT (Multi-GPU Support)
#
# Evaluates each candidate independently using P(Yes) logit scoring.
# Supports parallel evaluation across multiple GPUs.
#
# USAGE:
#   # Single GPU
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_mind_cot_pointwise.sh model_path dev
#
#   # Multi-GPU
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_cot_pointwise.sh model_path dev
#
#   # Quick test
#   bash scripts/eval_mind_cot_pointwise.sh model_path dev 100

set -euo pipefail

MODEL_PATH="${1:-}"
SPLIT="${2:-dev}"
MAX_IMPRESSIONS="${3:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <model_path> [split] [max_impressions]" >&2
  exit 1
fi

# Detect parallel mode
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  PARALLEL_MODE=true
  CUDA_LIST="${CUDA_VISIBLE_DEVICES}"
else
  PARALLEL_MODE=false
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi

# Configuration
MIND_SIZE="${MIND_SIZE:-small}"
if [[ -z "${MIND_ROOT:-}" ]]; then
  if [[ "${MIND_SIZE}" == "large" && -d "../data/MIND_large" ]]; then
    MIND_ROOT="../data/MIND_large"
  elif [[ "${MIND_SIZE}" == "small" && -d "../data/MIND_small" ]]; then
    MIND_ROOT="../data/MIND_small"
  else
    MIND_ROOT="../data/MIND"
  fi
fi

COT_STYLE="${COT_STYLE:-standard}"
SCORING="${SCORING:-forced_prefix}"    # forced_prefix (fast, for CoT), logit (non-CoT), generative (slow)
BATCH_SIZE="${BATCH_SIZE:-4}"
COT_MAX_TOKENS="${COT_MAX_TOKENS:-256}"
MAX_HISTORY="${MAX_HISTORY:-30}"
USE_ABSTRACT="${USE_ABSTRACT:-0}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
FLASH_ATTN="${FLASH_ATTN:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

DATA_DIR="${MIND_ROOT}/${SPLIT}"
BEHAVIORS_PATH="${DATA_DIR}/behaviors.tsv"
NEWS_PATH="${DATA_DIR}/news.tsv"

echo "========================================="
echo "MIND Pointwise CoT Evaluation"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: MIND${MIND_SIZE} ${SPLIT}"
echo "Scoring: ${SCORING}"
echo "CoT style: ${COT_STYLE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Disable thinking: ${DISABLE_THINKING}"
echo "Max impressions: ${MAX_IMPRESSIONS:-all}"
if [[ "${PARALLEL_MODE}" == "true" ]]; then
  echo "GPUs: ${CUDA_LIST} (parallel)"
else
  echo "GPU: ${CUDA_VISIBLE_DEVICES} (single)"
fi
echo "========================================="
echo ""

if [[ ! -f "${BEHAVIORS_PATH}" ]] || [[ ! -f "${NEWS_PATH}" ]]; then
  echo "ERROR: MIND data not found at ${DATA_DIR}" >&2
  exit 1
fi

# Build flags
build_cmd_flags() {
  local flags="--cot_style ${COT_STYLE} \
    --scoring ${SCORING} \
    --batch_size ${BATCH_SIZE} \
    --cot_max_tokens ${COT_MAX_TOKENS} \
    --max_history ${MAX_HISTORY}"

  if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    flags="${flags} --use_abstract"
  fi
  if [[ "${FLASH_ATTN}" -eq 1 ]]; then
    flags="${flags} --flash_attn"
  fi
  if [[ "${DISABLE_THINKING}" -eq 1 ]]; then
    flags="${flags} --disable_thinking"
  fi
  echo "${flags}"
}

if [[ "${PARALLEL_MODE}" == "true" ]]; then
  echo "Starting parallel evaluation..."
  TEMP_DIR="./temp_mind_cot_pw/${SPLIT}-$$"
  mkdir -p "${TEMP_DIR}"

  echo "Splitting behaviors..."
  python src/split_mind.py \
    --input_path "${BEHAVIORS_PATH}" \
    --output_path "${TEMP_DIR}" \
    --cuda_list "${CUDA_LIST}"

  cudalist=$(echo "$CUDA_LIST" | tr ',' ' ')
  pids=()
  gpu_pids=()
  BASE_FLAGS="$(build_cmd_flags)"

  for gpu_id in ${cudalist}; do
    if [[ ! -f "${TEMP_DIR}/${gpu_id}.tsv" ]]; then
      echo "WARNING: ${TEMP_DIR}/${gpu_id}.tsv not found, skipping GPU $gpu_id"
      continue
    fi

    echo "[GPU $gpu_id] Starting evaluation"

    cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -u src/evaluate_mind_cot_pointwise.py \
      --model_path \"${MODEL_PATH}\" \
      --behaviors_path \"${TEMP_DIR}/${gpu_id}.tsv\" \
      --news_path \"${NEWS_PATH}\" \
      --output_file \"${TEMP_DIR}/${gpu_id}.txt\" \
      ${BASE_FLAGS}"

    if [[ -n "${MAX_IMPRESSIONS}" ]] && [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
      num_gpus=$(echo "$CUDA_LIST" | tr ',' ' ' | wc -w)
      per_gpu=$((MAX_IMPRESSIONS / num_gpus))
      cmd="${cmd} --max_impressions ${per_gpu}"
    fi

    eval "$cmd 2>&1 | sed \"s/^/[GPU $gpu_id] /\"" &
    pid=$!
    pids+=($pid)
    gpu_pids+=("$gpu_id:$pid")
    echo "[GPU $gpu_id] PID $pid"
  done

  echo "Waiting for ${#pids[@]} processes..."
  failed_gpus=()
  for idx in "${!pids[@]}"; do
    pid=${pids[$idx]}
    gpu=${gpu_pids[$idx]%%:*}
    if wait $pid; then
      echo "[GPU $gpu] Done"
    else
      echo "[GPU $gpu] FAILED"
      failed_gpus+=($gpu)
    fi
  done

  if [[ ${#failed_gpus[@]} -gt 0 ]]; then
    echo "WARNING: ${#failed_gpus[@]} GPU(s) failed: ${failed_gpus[@]}"
  fi

  OUTPUT_FILE="${OUTPUT_FILE:-./results_mind/${SPLIT}_cot_pw_predictions.txt}"
  mkdir -p "$(dirname "$OUTPUT_FILE")"

  actual_list=$(ls "${TEMP_DIR}"/*.txt 2>/dev/null | sed 's/.*\///g' | sed 's/\.txt//g' | tr '\n' ',' | sed 's/,$//')
  python src/merge_mind.py \
    --input_path "${TEMP_DIR}" \
    --output_path "${OUTPUT_FILE}" \
    --cuda_list "${actual_list}" \
    --calculate_metrics false

  echo ""
  echo "Predictions saved: ${OUTPUT_FILE}"

  if [[ "${SPLIT}" == "dev" ]] && [[ -f "${BEHAVIORS_PATH}" ]]; then
    echo "Calculating metrics..."
    python src/calc_mind_metrics.py \
      --predictions "${OUTPUT_FILE}" \
      --behaviors "${BEHAVIORS_PATH}"
  fi

  rm -rf "${TEMP_DIR}"

else
  echo "Running single GPU evaluation..."
  BASE_FLAGS="$(build_cmd_flags)"

  CMD="python src/evaluate_mind_cot_pointwise.py \
    --model_path ${MODEL_PATH} \
    --behaviors_path ${BEHAVIORS_PATH} \
    --news_path ${NEWS_PATH} \
    ${BASE_FLAGS}"

  if [[ -n "${MAX_IMPRESSIONS}" ]] && [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
    CMD="${CMD} --max_impressions ${MAX_IMPRESSIONS}"
  fi
  if [[ -n "${OUTPUT_FILE}" ]]; then
    CMD="${CMD} --output_file ${OUTPUT_FILE}"
  fi

  eval "${CMD}"
fi

echo ""
echo "========================================="
echo "Evaluation completed!"
echo "========================================="
