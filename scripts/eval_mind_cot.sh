#!/usr/bin/env bash
# MIND Evaluation with Chain-of-Thought (CoT) Format (Multi-GPU Support)
#
# This script evaluates models trained with RL CoT reasoning.
# The model generates step-by-step reasoning before giving a final answer.
#
# USAGE:
#   # Single GPU (quick test)
#   bash scripts/eval_mind_cot.sh output_dir/rl_mind_cot/final_checkpoint dev 100
#
#   # Single GPU (full evaluation)
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_mind_cot.sh output_dir/rl_mind_cot/final_checkpoint dev
#
#   # Multi-GPU parallel evaluation (faster)
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_cot.sh output_dir/rl_mind_cot/final_checkpoint dev
#
#   # With custom CoT style (must match training)
#   COT_STYLE=detailed CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_cot.sh output_dir/rl_mind_cot/final_checkpoint dev

set -euo pipefail

# Parse arguments
MODEL_PATH="${1:-}"
SPLIT="${2:-dev}"
MAX_IMPRESSIONS="${3:-0}"

# Detect parallel mode based on CUDA_VISIBLE_DEVICES
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  PARALLEL_MODE=true
  CUDA_LIST="${CUDA_VISIBLE_DEVICES}"
else
  PARALLEL_MODE=false
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <model_path> [split] [max_impressions]" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  model_path: Path to RL CoT checkpoint" >&2
  echo "  split: dev or test (default: dev)" >&2
  echo "  max_impressions: Limit to N impressions (optional, 0=all)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  # Single GPU" >&2
  echo "  CUDA_VISIBLE_DEVICES=0 $0 output_dir/rl_mind_cot/final_checkpoint dev" >&2
  echo "" >&2
  echo "  # Multi-GPU parallel" >&2
  echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 $0 output_dir/rl_mind_cot/final_checkpoint dev" >&2
  exit 1
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
USE_ABSTRACT="${USE_ABSTRACT:-0}"
MAX_HISTORY="${MAX_HISTORY:-30}"       # Match training default
MAX_CANDIDATES="${MAX_CANDIDATES:-30}" # Match training default
COT_STYLE="${COT_STYLE:-standard}"     # Must match training COT_STYLE
COT_MAX_TOKENS="${COT_MAX_TOKENS:-512}" # Must match training MAX_RESPONSE_LENGTH
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"  # 1 for instruct models
DISABLE_THINKING="${DISABLE_THINKING:-0}"   # 1 to disable Qwen3 thinking mode
OUTPUT_FILE="${OUTPUT_FILE:-}"
FLASH_ATTN="${FLASH_ATTN:-1}"

# Construct paths
DATA_DIR="${MIND_ROOT}/${SPLIT}"
BEHAVIORS_PATH="${DATA_DIR}/behaviors.tsv"
NEWS_PATH="${DATA_DIR}/news.tsv"

echo "========================================="
echo "MIND CoT Evaluation"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: MIND${MIND_SIZE} ${SPLIT} split"
echo "Format: Chain-of-Thought generation"
echo ""
echo "GPU Configuration:"
if [[ "${PARALLEL_MODE}" == "true" ]]; then
  echo "  Mode: Parallel multi-GPU"
  echo "  GPUs: ${CUDA_LIST}"
else
  echo "  Mode: Single GPU"
  echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
fi
echo ""
echo "Configuration:"
echo "  CoT style: ${COT_STYLE}"
echo "  CoT max tokens: ${COT_MAX_TOKENS}"
echo "  Use chat template: ${USE_CHAT_TEMPLATE}"
echo "  Use abstract: ${USE_ABSTRACT}"
echo "  Max history: ${MAX_HISTORY}"
echo "  Max candidates: ${MAX_CANDIDATES}"
echo "  Max impressions: ${MAX_IMPRESSIONS:-all}"
echo "  Flash Attention: ${FLASH_ATTN}"
echo "========================================="
echo ""

# Check if data exists
if [[ ! -f "${BEHAVIORS_PATH}" ]] || [[ ! -f "${NEWS_PATH}" ]]; then
  echo "ERROR: MIND data not found!" >&2
  echo "  Behaviors: ${BEHAVIORS_PATH}" >&2
  echo "  News: ${NEWS_PATH}" >&2
  exit 1
fi

echo "✓ Data files found"
echo "  Behaviors: ${BEHAVIORS_PATH} ($(wc -l < "${BEHAVIORS_PATH}") impressions)"
echo "  News: ${NEWS_PATH} ($(wc -l < "${NEWS_PATH}") news articles)"
echo ""

# Helper to build the python command flags (shared between single/parallel)
build_cmd_flags() {
  local flags="--use_cot \
    --cot_style ${COT_STYLE} \
    --cot_max_tokens ${COT_MAX_TOKENS} \
    --max_history ${MAX_HISTORY} \
    --max_candidates ${MAX_CANDIDATES}"

  if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    flags="${flags} --use_abstract"
  fi
  if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then
    flags="${flags} --use_chat_template"
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
  # ========== PARALLEL MULTI-GPU MODE ==========
  echo "Starting parallel evaluation..."
  echo ""

  # Create temporary directory
  TEMP_DIR="./temp_mind_cot/${SPLIT}-$$"
  mkdir -p "${TEMP_DIR}"

  # Split behaviors across GPUs
  echo "Splitting behaviors across GPUs..."
  python src/split_mind.py \
    --input_path "${BEHAVIORS_PATH}" \
    --output_path "${TEMP_DIR}" \
    --cuda_list "${CUDA_LIST}"

  echo ""

  # Start parallel evaluation on each GPU
  cudalist=$(echo "$CUDA_LIST" | tr ',' ' ')
  pids=()
  gpu_pids=()

  BASE_FLAGS="$(build_cmd_flags)"

  for gpu_id in ${cudalist}; do
    if [[ ! -f "${TEMP_DIR}/${gpu_id}.tsv" ]]; then
      echo "WARNING: Split file ${TEMP_DIR}/${gpu_id}.tsv not found, skipping GPU $gpu_id"
      continue
    fi

    echo "[GPU $gpu_id] Starting evaluation"

    cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -u src/evaluate_mind_ranking.py \
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

    # Run in background
    eval "$cmd 2>&1 | sed \"s/^/[GPU $gpu_id] /\"" &
    pid=$!
    pids+=($pid)
    gpu_pids+=("$gpu_id:$pid")
    echo "[GPU $gpu_id] Process started with PID $pid"
  done

  echo ""
  echo "Waiting for ${#pids[@]} GPU process(es) to complete..."
  echo ""

  # Wait for all processes
  failed_gpus=()
  for idx in "${!pids[@]}"; do
    pid=${pids[$idx]}
    gpu_info=${gpu_pids[$idx]}
    gpu=${gpu_info%%:*}

    if wait $pid; then
      echo "✓ GPU $gpu completed successfully"
    else
      echo "✗ ERROR: GPU $gpu failed"
      failed_gpus+=($gpu)
    fi
  done

  echo ""

  if [[ ${#failed_gpus[@]} -gt 0 ]]; then
    echo "WARNING: ${#failed_gpus[@]} GPU(s) failed: ${failed_gpus[@]}"
  fi

  OUTPUT_FILE="${OUTPUT_FILE:-./results_mind/${SPLIT}_cot_predictions.txt}"
  mkdir -p "$(dirname "$OUTPUT_FILE")"

  echo "Merging predictions..."
  actual_cuda_list=$(ls "${TEMP_DIR}"/*.txt 2>/dev/null | sed 's/.*\///g' | sed 's/\.txt//g' | tr '\n' ',' | sed 's/,$//')

  python src/merge_mind.py \
    --input_path "${TEMP_DIR}" \
    --output_path "${OUTPUT_FILE}" \
    --cuda_list "${actual_cuda_list}" \
    --calculate_metrics false

  echo ""
  echo "========================================="
  echo "✓ Parallel evaluation completed!"
  echo "========================================="
  echo "Predictions saved to: ${OUTPUT_FILE}"

  # Calculate metrics for dev split
  if [[ "${SPLIT}" == "dev" ]] && [[ -f "${BEHAVIORS_PATH}" ]]; then
    echo ""
    echo "Calculating metrics from predictions..."
    python src/calc_mind_metrics.py \
      --predictions "${OUTPUT_FILE}" \
      --behaviors "${BEHAVIORS_PATH}"
  elif [[ "${SPLIT}" == "test" ]]; then
    echo ""
    echo "Note: Test split has no ground truth labels."
    echo "Submit predictions to MIND leaderboard for evaluation."
  fi

  # Cleanup temp files
  rm -rf "${TEMP_DIR}"

else
  # ========== SINGLE GPU MODE ==========
  echo "Running evaluation..."
  echo ""

  BASE_FLAGS="$(build_cmd_flags)"

  CMD="python src/evaluate_mind_ranking.py \
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

  echo ""
  echo "========================================="
  echo "✓ Evaluation completed!"
  echo "========================================="
fi
