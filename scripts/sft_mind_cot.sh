#!/bin/bash
# ==========================================================================
# MIND SFT for Chain-of-Thought Format
# ==========================================================================
# Quick SFT to teach Qwen3 the <think>...</think><answer>[...]</answer> format.
# Run this BEFORE rl_mind_cot.sh when starting from a base model.
#
# What it does:
#   1. Generate ~2000 synthetic CoT samples (template-based reasoning + correct format)
#   2. Fine-tune for 2 epochs to teach the output format
#   3. Save checkpoint for RL training
#
# Usage:
#   DATA_ROOT=/path/to/MIND_small bash scripts/sft_mind_cot.sh
#
# Then run RL:
#   SFT_MODEL=output_dir/sft_mind_cot/final_checkpoint bash scripts/rl_mind_cot.sh
# ==========================================================================

set -e

# =========================
# Configuration
# =========================
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/MIND}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
OUTPUT_DIR=${OUTPUT_DIR:-output_dir/sft_mind_cot}

# SFT data generation settings
SFT_MAX_SAMPLES=${SFT_MAX_SAMPLES:-5000}   # How many synthetic samples to generate
COT_STYLE=${COT_STYLE:-standard}
MAX_HISTORY=${MAX_HISTORY:-30}
MAX_CANDIDATES=${MAX_CANDIDATES:-30}

# Training hyperparams (quick SFT — just teach format)
BATCH_SIZE=${BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
CUTOFF_LEN=${CUTOFF_LEN:-4096}

# Hardware
N_GPUS=${N_GPUS:-8}

# WandB
WANDB_PROJECT=${WANDB_PROJECT:-MiniOneRec_MIND}
MODEL_BASENAME=$(basename ${MODEL_PATH})
WANDB_RUN_NAME=${WANDB_RUN_NAME:-sft_cot_${MODEL_BASENAME}_n${SFT_MAX_SAMPLES}_ep${NUM_EPOCHS}_lr${LEARNING_RATE}}
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

export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_CUDA_ARCH_LIST="8.0"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1

# =========================
# Environment
# =========================
export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

# Use a different port to avoid conflicts
export MASTER_PORT=${MASTER_PORT:-29505}

# DeepSpeed hostfile
if [ -z "$HOSTFILE" ]; then
    if [ -f "/job/hostfile" ]; then
        HOSTFILE="/job/hostfile"
    elif [ -f "./hostfile" ]; then
        HOSTFILE="./hostfile"
    else
        HOSTFILE="./hostfile"
        if [ ! -f "$HOSTFILE" ]; then
            echo "Creating default hostfile..."
            echo "localhost slots=${N_GPUS}" > $HOSTFILE
        fi
    fi
fi
echo "Using hostfile: $HOSTFILE"
export PDSH_RCMD_TYPE=ssh

# =========================
# Step 1: Generate SFT Data
# =========================
echo "=========================================="
echo "Step 1: Generating Synthetic CoT SFT Data"
echo "=========================================="

SFT_TRAIN_FILE=${DATA_ROOT}/train/sft_cot_train.jsonl
SFT_EVAL_FILE=${DATA_ROOT}/dev/sft_cot_dev.jsonl

if [ ! -f "$SFT_TRAIN_FILE" ]; then
    echo "Generating training data (${SFT_MAX_SAMPLES} samples)..."
    python src/prepare_mind_sft_cot.py \
        --mind_root ${DATA_ROOT} \
        --split train \
        --output ${SFT_TRAIN_FILE} \
        --max_samples ${SFT_MAX_SAMPLES} \
        --max_history ${MAX_HISTORY} \
        --max_candidates ${MAX_CANDIDATES} \
        --cot_style ${COT_STYLE}
    echo "Training data saved: ${SFT_TRAIN_FILE}"
else
    echo "Training data exists: ${SFT_TRAIN_FILE}"
fi

# Generate small eval set
if [ ! -f "$SFT_EVAL_FILE" ]; then
    echo "Generating eval data (200 samples)..."
    python src/prepare_mind_sft_cot.py \
        --mind_root ${DATA_ROOT} \
        --split dev \
        --output ${SFT_EVAL_FILE} \
        --max_samples 200 \
        --max_history ${MAX_HISTORY} \
        --max_candidates ${MAX_CANDIDATES} \
        --cot_style ${COT_STYLE}
    echo "Eval data saved: ${SFT_EVAL_FILE}"
else
    echo "Eval data exists: ${SFT_EVAL_FILE}"
fi

echo ""

# =========================
# Step 2: Run SFT Training
# =========================
echo "=========================================="
echo "Step 2: SFT Training (Format Learning)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Train file: ${SFT_TRAIN_FILE}"
echo "Eval file: ${SFT_EVAL_FILE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Micro batch: ${MICRO_BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "=========================================="
echo ""

deepspeed --hostfile=$HOSTFILE \
    --master_port=${MASTER_PORT} \
    --launcher=pdsh \
    --launcher_args="-S" \
    src/sft_mind_cot_ds.py \
    --base_model ${MODEL_PATH} \
    --train_file ${SFT_TRAIN_FILE} \
    --eval_file ${SFT_EVAL_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    --deepspeed_config ds_configs/ds_config_zero2.json

echo ""
echo "=========================================="
echo "SFT Training Complete!"
echo "=========================================="
echo "Model saved to: ${OUTPUT_DIR}/final_checkpoint"
echo ""
echo "Next: Run RL training with this checkpoint:"
echo "  SFT_MODEL=${OUTPUT_DIR}/final_checkpoint \\"
echo "  DATA_ROOT=${DATA_ROOT} \\"
echo "  REWARD_TYPE=mind_cot_prob_auc \\"
echo "  bash scripts/rl_mind_cot.sh"
