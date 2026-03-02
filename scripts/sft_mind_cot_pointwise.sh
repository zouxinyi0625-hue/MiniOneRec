#!/bin/bash
# ==========================================================================
# MIND SFT for Pointwise Chain-of-Thought Format
# ==========================================================================
# Quick SFT to teach Qwen3 the <think>...</think><answer>Yes/No</answer> format.
# Run this BEFORE rl_mind_cot_pointwise.sh when starting from a base model.
#
# What it does:
#   1. Generate synthetic pointwise CoT samples (template reasoning + correct format)
#   2. Fine-tune for 2 epochs to teach the output format
#   3. Save checkpoint for RL training
#
# The prompt format is IDENTICAL to the RL training (prepare_mind_rl_cot_pointwise.py),
# so the SFT → RL transition is seamless.
#
# Usage:
#   DATA_ROOT=/path/to/MIND_small MODEL_PATH=Qwen/Qwen3-1.7B bash scripts/sft_mind_cot_pointwise.sh
#
# Then run RL:
#   SFT_MODEL=output_dir/sft_mind_cot_pointwise/final_checkpoint bash scripts/rl_mind_cot_pointwise.sh
# ==========================================================================

set -e

# =========================
# Configuration
# =========================
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/MIND}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
OUTPUT_DIR=${OUTPUT_DIR:-output_dir/sft_mind_cot_pointwise}

# SFT data generation settings
SFT_MAX_SAMPLES=${SFT_MAX_SAMPLES:-0}  # 0 = all (pointwise is already per-candidate, usually ~30k for MIND_small)
COT_STYLE=${COT_STYLE:-standard}
MAX_HISTORY=${MAX_HISTORY:-30}
NEG_RATIO=${NEG_RATIO:-2.0}

# Training hyperparams
BATCH_SIZE=${BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
CUTOFF_LEN=${CUTOFF_LEN:-1024}  # Pointwise is short: prompt ~600 + response ~100

# Hardware
N_GPUS=${N_GPUS:-8}

# WandB
WANDB_PROJECT=${WANDB_PROJECT:-MiniOneRec_MIND}
MODEL_BASENAME=$(basename ${MODEL_PATH})
WANDB_RUN_NAME=${WANDB_RUN_NAME:-sft_cot_pw_${MODEL_BASENAME}_ep${NUM_EPOCHS}_lr${LEARNING_RATE}_neg${NEG_RATIO}}
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

export MASTER_PORT=${MASTER_PORT:-29506}

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
echo "Step 1: Generating Pointwise CoT SFT Data"
echo "=========================================="

SFT_TRAIN_FILE=${DATA_ROOT}/train/sft_cot_pw_train.jsonl
SFT_EVAL_FILE=${DATA_ROOT}/dev/sft_cot_pw_dev.jsonl

if [ ! -f "$SFT_TRAIN_FILE" ]; then
    echo "Generating training data..."
    python src/prepare_mind_sft_cot_pointwise.py \
        --mind_root ${DATA_ROOT} \
        --split train \
        --output ${SFT_TRAIN_FILE} \
        --max_history ${MAX_HISTORY} \
        --neg_ratio ${NEG_RATIO} \
        --cot_style ${COT_STYLE} \
        ${SFT_MAX_SAMPLES:+$([ "$SFT_MAX_SAMPLES" -ne 0 ] && echo "--max_samples $SFT_MAX_SAMPLES" || true)}
    echo "Training data saved: ${SFT_TRAIN_FILE}"
else
    echo "Training data exists: ${SFT_TRAIN_FILE}"
fi

if [ ! -f "$SFT_EVAL_FILE" ]; then
    echo "Generating eval data..."
    python src/prepare_mind_sft_cot_pointwise.py \
        --mind_root ${DATA_ROOT} \
        --split dev \
        --output ${SFT_EVAL_FILE} \
        --max_history ${MAX_HISTORY} \
        --neg_ratio ${NEG_RATIO} \
        --max_samples 1000 \
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
echo "Step 2: Pointwise CoT SFT Training"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Train file: ${SFT_TRAIN_FILE}"
echo "Eval file: ${SFT_EVAL_FILE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Micro batch: ${MICRO_BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "=========================================="
echo ""

deepspeed --hostfile=$HOSTFILE \
    --master_port=${MASTER_PORT} \
    --launcher=pdsh \
    --launcher_args="-S" \
    src/sft_mind_cot_pointwise_ds.py \
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
echo "Pointwise CoT SFT Complete!"
echo "=========================================="
echo "Model saved to: ${OUTPUT_DIR}/final_checkpoint"
echo ""
echo "Next: Run RL training with this checkpoint:"
echo "  SFT_MODEL=${OUTPUT_DIR}/final_checkpoint \\"
echo "  DATA_ROOT=${DATA_ROOT} \\"
echo "  bash scripts/rl_mind_cot_pointwise.sh"
