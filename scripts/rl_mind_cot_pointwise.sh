#!/bin/bash
# ==========================================================================
# MIND RL Training with Pointwise Chain-of-Thought (CoT)
# ==========================================================================
# Each sample: user history + ONE candidate article → <think>...</think><answer>Yes/No</answer>
#
# Key advantages over ranking-wise CoT:
# - Simpler task: binary prediction instead of probability distribution
# - Shorter response: ~256 tokens instead of ~1024
# - Model focuses on one article at a time
# - Generalizes to any number of candidates at eval time
#
# Usage:
#   DATA_ROOT=/path/to/MIND SFT_MODEL=Qwen/Qwen3-1.7B bash scripts/rl_mind_cot_pointwise.sh
# ==========================================================================

set -e

# =========================
# Configuration
# =========================
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/MIND}
SFT_MODEL=${SFT_MODEL:-Qwen/Qwen3-1.7B}
OUTPUT_DIR=${OUTPUT_DIR:-output_dir/rl_mind_cot_pointwise}

# CoT settings
COT_STYLE=${COT_STYLE:-category}
REWARD_TYPE=${REWARD_TYPE:-cot_pointwise_format}  # Options: cot_pointwise_binary, cot_pointwise_format, cot_pointwise_asymmetric

# Training settings
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}   # Pointwise needs much less (think + Yes/No)
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.05}               # Lower KL — task is simpler
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-4}

# Data settings
MAX_HISTORY=${MAX_HISTORY:-30}
NEG_RATIO=${NEG_RATIO:-2.0}                       # 2:1 neg:pos
MAX_SAMPLES=${MAX_SAMPLES:-0}                      # 0 = all

# Hardware
N_GPUS=${N_GPUS:-8}

# WandB
WANDB_PROJECT=${WANDB_PROJECT:-MiniOneRec_MIND}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-cot_pw_${REWARD_TYPE}_lr${LEARNING_RATE}_kl${KL_LOSS_COEF}_bs${TRAIN_BATCH_SIZE}_g${NUM_GENERATIONS}_r${MAX_RESPONSE_LENGTH}_h${MAX_HISTORY}_neg${NEG_RATIO}}
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
export UVLOOP_DISABLE=1

# =========================
# Environment
# =========================
export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

# =========================
# Step 1: Prepare Data
# =========================
echo "=========================================="
echo "Step 1: Preparing Pointwise CoT RL Data"
echo "=========================================="

NEG_TAG=$(echo "${NEG_RATIO}" | tr '.' 'p')
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_ROOT}/train/rl_cot_pw_neg${NEG_TAG}_train.parquet}
EVAL_PARQUET=${EVAL_PARQUET:-${DATA_ROOT}/dev/rl_cot_pw_neg${NEG_TAG}_dev.parquet}

# Prepare training data
if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "Preparing training data..."
    python src/prepare_mind_rl_cot_pointwise.py \
        --behaviors_path ${DATA_ROOT}/train/behaviors.tsv \
        --news_path ${DATA_ROOT}/train/news.tsv \
        --output_parquet ${TRAIN_PARQUET} \
        --max_history ${MAX_HISTORY} \
        --neg_ratio ${NEG_RATIO} \
        --cot_style ${COT_STYLE} \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
else
    echo "Training parquet exists: ${TRAIN_PARQUET}"
fi

# Prepare eval data
if [ ! -f "$EVAL_PARQUET" ]; then
    echo "Preparing evaluation data..."
    python src/prepare_mind_rl_cot_pointwise.py \
        --behaviors_path ${DATA_ROOT}/dev/behaviors.tsv \
        --news_path ${DATA_ROOT}/dev/news.tsv \
        --output_parquet ${EVAL_PARQUET} \
        --max_history ${MAX_HISTORY} \
        --neg_ratio ${NEG_RATIO} \
        --cot_style ${COT_STYLE} \
        --max_samples 5000
else
    echo "Evaluation parquet exists: ${EVAL_PARQUET}"
fi

echo ""

# =========================
# Step 2: Run RL Training
# =========================
echo "=========================================="
echo "Step 2: Running Pointwise CoT RL Training"
echo "=========================================="
echo "SFT Model: ${SFT_MODEL}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Reward Type: ${REWARD_TYPE}"
echo "CoT Style: ${COT_STYLE}"
echo "Max Response Length: ${MAX_RESPONSE_LENGTH}"
echo "Neg Ratio: ${NEG_RATIO}"
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
echo "Pointwise CoT RL Training Complete!"
echo "=========================================="
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 COT_STYLE=${COT_STYLE} bash scripts/eval_mind_cot_pointwise.sh \\"
echo "      ${OUTPUT_DIR}/final_checkpoint dev"
