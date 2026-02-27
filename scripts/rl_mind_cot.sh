#!/bin/bash
# ==========================================================================
# MIND RL Training with Chain-of-Thought (CoT)
# ==========================================================================
# This script trains a news recommendation model using RL with CoT reasoning.
# The model learns to generate reasoning before providing an answer.
#
# Key features:
# - Model generates: <reasoning> ... Answer: <number>
# - Reward is based on correctness of final answer
# - No CoT training data needed - learned through RL
#
# Usage:
#   DATA_ROOT=/path/to/MIND SFT_MODEL=/path/to/sft/checkpoint bash scripts/rl_mind_cot.sh
# ==========================================================================

set -e

# =========================
# Configuration
# =========================
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/MIND}
# SFT_MODEL: use the CoT format SFT checkpoint (from sft_mind_cot.sh),
# or a HuggingFace model name for direct RL (not recommended without SFT).
SFT_MODEL=${SFT_MODEL:-output_dir/sft_mind_cot/final_checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:-output_dir/rl_mind_cot}

# CoT settings
COT_STYLE=${COT_STYLE:-standard}  # Options: standard, category, detailed
REWARD_TYPE=${REWARD_TYPE:-mind_cot_prob_auc}  # Options: mind_cot_prob_auc, mind_cot_prob_ndcg, mind_cot_prob_ce, mind_cot_prob_margin, mind_cot_prob_format (new prob-based); mind_cot_binary, mind_cot_ndcg, mind_cot_auc, mind_cot_margin, mind_cot_format (legacy single-answer)

# Training settings
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}  # Longer for CoT reasoning
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}  # 8x A100-80GB can handle 128+ easily for 1.7B model
LEARNING_RATE=${LEARNING_RATE:-1e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}  # More generations = better GRPO advantage estimation
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}  # Actor update mini-batch
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-4}  # Per-GPU micro-batch

# Data settings
MAX_HISTORY=${MAX_HISTORY:-30}
MAX_CANDIDATES=${MAX_CANDIDATES:-30}
MAX_SAMPLES=${MAX_SAMPLES:-0}  # 0 = use all

# Hardware
N_GPUS=${N_GPUS:-8}

# WandB (set WANDB_API_KEY env var before running)
WANDB_PROJECT=${WANDB_PROJECT:-MiniOneRec_MIND}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-cot_${REWARD_TYPE}_lr${LEARNING_RATE}_kl${KL_LOSS_COEF}_bs${TRAIN_BATCH_SIZE}_g${NUM_GENERATIONS}_r${MAX_RESPONSE_LENGTH}}
export WANDB_API_KEY=${WANDB_API_KEY:?"WANDB_API_KEY not set. Export it before running."}

# =========================
# NCCL Configuration
# =========================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

# Disable uvloop to avoid asyncio event loop conflict with Ray/VERL
export UVLOOP_DISABLE=1

# =========================
# Environment
# =========================
export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

# =========================
# Step 1: Prepare CoT Data
# =========================
echo "=========================================="
echo "Step 1: Preparing CoT RL Data"
echo "=========================================="

TRAIN_PARQUET=${DATA_ROOT}/train/rl_cot_train.parquet
EVAL_PARQUET=${DATA_ROOT}/dev/rl_cot_dev.parquet

# Prepare training data if not exists
if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "Preparing training data..."
    python prepare_mind_rl_cot.py \
        --behaviors_path ${DATA_ROOT}/train/behaviors.tsv \
        --news_path ${DATA_ROOT}/train/news.tsv \
        --output_parquet ${TRAIN_PARQUET} \
        --max_history ${MAX_HISTORY} \
        --max_candidates ${MAX_CANDIDATES} \
        --cot_style ${COT_STYLE} \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
else
    echo "Training parquet exists: ${TRAIN_PARQUET}"
fi

# Prepare evaluation data if not exists
if [ ! -f "$EVAL_PARQUET" ]; then
    echo "Preparing evaluation data..."
    python prepare_mind_rl_cot.py \
        --behaviors_path ${DATA_ROOT}/dev/behaviors.tsv \
        --news_path ${DATA_ROOT}/dev/news.tsv \
        --output_parquet ${EVAL_PARQUET} \
        --max_history ${MAX_HISTORY} \
        --max_candidates ${MAX_CANDIDATES} \
        --cot_style ${COT_STYLE} \
        --max_samples 2000  # Smaller eval set
else
    echo "Evaluation parquet exists: ${EVAL_PARQUET}"
fi

echo ""

# =========================
# Step 2: Run RL Training
# =========================
echo "=========================================="
echo "Step 2: Running RL Training with CoT"
echo "=========================================="
echo "SFT Model: ${SFT_MODEL}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Reward Type: ${REWARD_TYPE}"
echo "Max Response Length: ${MAX_RESPONSE_LENGTH}"
echo "Train Batch Size: ${TRAIN_BATCH_SIZE}"
echo "PPO Mini Batch: ${PPO_MINI_BATCH_SIZE}"
echo "PPO Micro Batch/GPU: ${PPO_MICRO_BATCH_SIZE}"
echo "Num Generations: ${NUM_GENERATIONS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "KL Coefficient: ${KL_LOSS_COEF}"
echo "GPUs: ${N_GPUS}"
echo "=========================================="
echo ""

python src/rl_mind_verl.py \
    --model_path ${SFT_MODEL} \
    --train_parquet ${TRAIN_PARQUET} \
    --eval_parquet ${EVAL_PARQUET} \
    --output_dir ${OUTPUT_DIR} \
    --reward_type ${REWARD_TYPE} \
    --max_response_length ${MAX_RESPONSE_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --kl_loss_coef ${KL_LOSS_COEF} \
    --total_epochs ${TOTAL_EPOCHS} \
    --num_generations ${NUM_GENERATIONS} \
    --ppo_mini_batch_size ${PPO_MINI_BATCH_SIZE} \
    --ppo_micro_batch_size_per_gpu ${PPO_MICRO_BATCH_SIZE} \
    --n_gpus_per_node ${N_GPUS} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME}

echo ""
echo "=========================================="
echo "RL Training Complete!"
echo "=========================================="
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate:"
echo "  python evaluate_mind_ranking.py \\"
echo "      --model_path ${OUTPUT_DIR}/final_checkpoint \\"
echo "      --behaviors_path ${DATA_ROOT}/dev/behaviors.tsv \\"
echo "      --news_path ${DATA_ROOT}/dev/news.tsv \\"
echo "      --use_cot True"
