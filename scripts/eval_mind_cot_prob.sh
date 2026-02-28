#!/usr/bin/env bash
# ==========================================================================
# MIND CoT Evaluation (Prob-based) — Multi-GPU Support
# ==========================================================================
# Evaluates CoT SFT / RL models that output:
#   <think>reasoning</think><answer>[1:prob, 2:prob, ...]</answer>
#
# Uses per-candidate probabilities for AUC/MRR/nDCG (not single-answer).
#
# USAGE:
#   # Single GPU
#   bash scripts/eval_mind_cot_prob.sh <model_path> [split] [max_impressions]
#
#   # Multi-GPU parallel
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_cot_prob.sh <model_path> dev
#
#   # Quick test
#   bash scripts/eval_mind_cot_prob.sh output_dir/sft_mind_cot/final_checkpoint dev 100
# ==========================================================================

set -euo pipefail

# Parse arguments
MODEL_PATH="${1:-}"
SPLIT="${2:-dev}"
MAX_IMPRESSIONS="${3:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <model_path> [split] [max_impressions]" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  model_path       Path to CoT SFT or RL checkpoint" >&2
  echo "  split            dev or test (default: dev)" >&2
  echo "  max_impressions  Limit to N impressions (0=all)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 output_dir/sft_mind_cot/final_checkpoint dev" >&2
  echo "  $0 output_dir/rl_mind_cot/final_checkpoint dev 500" >&2
  echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 $0 output_dir/rl_mind_cot/final_checkpoint dev" >&2
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
COT_MAX_TOKENS="${COT_MAX_TOKENS:-512}"
MAX_HISTORY="${MAX_HISTORY:-30}"
MAX_CANDIDATES="${MAX_CANDIDATES:-30}"
USE_ABSTRACT="${USE_ABSTRACT:-0}"
FLASH_ATTN="${FLASH_ATTN:-1}"
VERBOSE="${VERBOSE:-0}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

# Data paths
DATA_DIR="${MIND_ROOT}/${SPLIT}"
BEHAVIORS_PATH="${DATA_DIR}/behaviors.tsv"
NEWS_PATH="${DATA_DIR}/news.tsv"

echo "=========================================="
echo "MIND CoT Evaluation (Prob-based)"
echo "=========================================="
echo "Model:          ${MODEL_PATH}"
echo "Dataset:        MIND_${MIND_SIZE} ${SPLIT}"
echo "CoT style:      ${COT_STYLE}"
echo "CoT max tokens: ${COT_MAX_TOKENS}"
echo "Max history:    ${MAX_HISTORY}"
echo "Max candidates: ${MAX_CANDIDATES}"
echo "Use abstract:   ${USE_ABSTRACT}"
echo "Flash Attn:     ${FLASH_ATTN}"
echo ""
if [[ "${PARALLEL_MODE}" == "true" ]]; then
  echo "Mode: Parallel multi-GPU (${CUDA_LIST})"
else
  echo "Mode: Single GPU (${CUDA_VISIBLE_DEVICES})"
fi
echo "=========================================="
echo ""

# Check data
if [[ ! -f "${BEHAVIORS_PATH}" ]] || [[ ! -f "${NEWS_PATH}" ]]; then
  echo "ERROR: Data not found!" >&2
  echo "  Behaviors: ${BEHAVIORS_PATH}" >&2
  echo "  News: ${NEWS_PATH}" >&2
  exit 1
fi
echo "Data: ${BEHAVIORS_PATH} ($(wc -l < "${BEHAVIORS_PATH}") impressions)"
echo ""

# Build common flags
build_flags() {
  local flags="--cot_style ${COT_STYLE} --cot_max_tokens ${COT_MAX_TOKENS}"
  flags="${flags} --max_history ${MAX_HISTORY} --max_candidates ${MAX_CANDIDATES}"

  if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    flags="${flags} --use_abstract"
  fi
  if [[ "${FLASH_ATTN}" -eq 1 ]]; then
    flags="${flags} --flash_attn"
  fi
  if [[ "${VERBOSE}" -eq 1 ]]; then
    flags="${flags} --verbose"
  fi
  echo "${flags}"
}

BASE_FLAGS="$(build_flags)"

if [[ "${PARALLEL_MODE}" == "true" ]]; then
  # ========== PARALLEL MULTI-GPU ==========
  echo "Starting parallel evaluation..."
  TEMP_DIR="./temp_mind_cot_prob/${SPLIT}-$$"
  mkdir -p "${TEMP_DIR}"

  # Split behaviors
  python src/split_mind.py \
    --input_path "${BEHAVIORS_PATH}" \
    --output_path "${TEMP_DIR}" \
    --cuda_list "${CUDA_LIST}"

  echo ""

  cudalist=$(echo "$CUDA_LIST" | tr ',' ' ')
  pids=()
  gpu_pids=()

  for gpu_id in ${cudalist}; do
    if [[ ! -f "${TEMP_DIR}/${gpu_id}.tsv" ]]; then
      echo "WARNING: ${TEMP_DIR}/${gpu_id}.tsv not found, skipping GPU $gpu_id"
      continue
    fi

    echo "[GPU $gpu_id] Starting evaluation"

    cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -u src/evaluate_mind_cot.py \
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

  echo ""
  echo "Waiting for ${#pids[@]} GPU(s)..."

  failed_gpus=()
  for idx in "${!pids[@]}"; do
    pid=${pids[$idx]}
    gpu_info=${gpu_pids[$idx]}
    gpu=${gpu_info%%:*}
    if wait $pid; then
      echo "[GPU $gpu] Done"
    else
      echo "[GPU $gpu] FAILED"
      failed_gpus+=($gpu)
    fi
  done

  echo ""

  if [[ ${#failed_gpus[@]} -gt 0 ]]; then
    echo "WARNING: ${#failed_gpus[@]} GPU(s) failed: ${failed_gpus[@]}"
  fi

  OUTPUT_FILE="${OUTPUT_FILE:-./results_mind/${SPLIT}_cot_prob_predictions.txt}"
  mkdir -p "$(dirname "$OUTPUT_FILE")"

  echo "Merging predictions..."
  actual_list=$(ls "${TEMP_DIR}"/*.txt 2>/dev/null | sed 's/.*\///g' | sed 's/\.txt//g' | tr '\n' ',' | sed 's/,$//')
  python src/merge_mind.py \
    --input_path "${TEMP_DIR}" \
    --output_path "${OUTPUT_FILE}" \
    --cuda_list "${actual_list}" \
    --calculate_metrics false

  echo ""
  echo "Predictions: ${OUTPUT_FILE}"

  # Calculate metrics
  if [[ "${SPLIT}" == "dev" ]] && [[ -f "${BEHAVIORS_PATH}" ]]; then
    echo ""
    echo "Calculating metrics..."
    python src/calc_mind_metrics.py \
      --predictions "${OUTPUT_FILE}" \
      --behaviors "${BEHAVIORS_PATH}"
  fi

  rm -rf "${TEMP_DIR}"

else
  # ========== SINGLE GPU ==========
  echo "Running evaluation..."
  echo ""

  CMD="python src/evaluate_mind_cot.py \
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
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
