#!/usr/bin/env python3
"""
RL training for MIND news recommendation using VERL framework.

This script wraps the VERL trainer with MIND-specific configurations and
uses nDCG-based reward functions for optimizing news ranking.

Usage:
    python src/rl_mind_verl.py \
        --model_path output_dir/sft_mind_small_Qwen3-1.7B_bs1024/final_checkpoint \
        --train_parquet ../data/MIND/train/rl_train.parquet \
        --eval_parquet ../data/MIND/dev/rl_dev.parquet \
        --output_dir output_dir/rl_mind_ndcg \
        --reward_type mind_ndcg \
        --total_epochs 1

Author: MiniOneRec
"""

from fire import Fire
from minionerec_verl_trainer import train_verl


def main(
    # Model and data paths
    model_path: str = "",
    train_parquet: str = "",
    eval_parquet: str = "",
    output_dir: str = "output_dir/rl_mind_auc",

    # Reward configuration
    reward_type: str = "mind_auc",  # Options: mind_auc, mind_auc_rank, mind_ndcg, mind_mrr, mind_cot_*

    # Generation parameters
    num_generations: int = 16,  # Number of generations per prompt for better GRPO advantage estimation
    max_prompt_length: int = 2048,  # For letter format (up to 26 candidates), use 4096 for numeric
    max_response_length: int = 8,  # Short output (Yes/No/number); use 256+ for CoT

    # Training hyperparameters
    train_batch_size: int = 64,  # Further reduced for memory with larger prompts
    learning_rate: float = 1e-7,  # Very conservative LR for stability
    total_epochs: int = 1,
    temperature: float = 0.8,

    # GRPO/PPO configuration
    rollout_name: str = "vllm",
    ppo_mini_batch_size: int = 16,  # Smaller for memory with larger prompts
    ppo_micro_batch_size_per_gpu: int = 2,  # Reduced for GPU memory
    kl_loss_coef: float = 0.5,  # High KL penalty to prevent divergence from SFT
    kl_loss_type: str = "low_var_kl",

    # Logging and checkpointing
    wandb_project: str = "MiniOneRec",
    wandb_run_name: str = "rl_mind_auc",
    save_freq: int = 20,
    test_freq: int = 5,

    # Distributed training
    nnodes: int = 1,
    n_gpus_per_node: int = 8,

    # Legacy parameters (not used for MIND, kept for API compatibility)
    sid_info_file: str = "",
    ada_path: str = "",
    cf_path: str = "",
    sasrec_len_seq: int = 10,
):
    """
    Train MIND news recommendation model with RL using nDCG reward.

    Args:
        model_path: Path to SFT checkpoint (starting point for RL)
        train_parquet: Path to training parquet file (from prepare_mind_rl.py or prepare_mind_rl_cot.py)
        eval_parquet: Path to evaluation parquet file
        output_dir: Directory to save checkpoints and logs
        reward_type: Reward function type
            Standard (short output - just answer number):
            - 'mind_auc': Binary reward for selecting clicked item
            - 'mind_auc_rank': AUC reward with ranking consideration
            - 'mind_ndcg': nDCG-style reward (1/log2(rank+1))
            - 'mind_mrr': MRR-style reward (1/rank)

            Chain-of-Thought (reasoning + answer):
            - 'mind_cot_binary': Binary reward for CoT output
            - 'mind_cot_ndcg': nDCG-style reward with partial credit for category match
            - 'mind_cot_auc': AUC reward for CoT (any clicked item = reward)
            - 'mind_cot_margin': Margin-based reward with penalties for wrong answers
            - 'mind_cot_format': Reward that includes bonus for proper formatting

        num_generations: Number of generations per prompt for GRPO
        max_prompt_length: Maximum prompt length (include full history)
        max_response_length: Maximum response length
            - For standard: 4-8 (just a number)
            - For CoT: 256-512 (reasoning + answer)
        train_batch_size: Training batch size
        learning_rate: Learning rate (use low value for stability)
        total_epochs: Number of training epochs
        temperature: Sampling temperature for generation
        ppo_mini_batch_size: PPO mini-batch size
        ppo_micro_batch_size_per_gpu: Micro-batch size per GPU
        kl_loss_coef: KL divergence coefficient (higher = more conservative)
        kl_loss_type: Type of KL loss computation
        wandb_project: Weights & Biases project name
        wandb_run_name: Weights & Biases run name
        save_freq: Checkpoint save frequency
        test_freq: Evaluation frequency
        nnodes: Number of nodes for distributed training
        n_gpus_per_node: Number of GPUs per node

    Example:
        # Standard RL (short output)
        python src/rl_mind_verl.py \\
            --model_path output_dir/sft_mind_ranking/final_checkpoint \\
            --train_parquet ../data/MIND/train/rl_train.parquet \\
            --eval_parquet ../data/MIND/dev/rl_dev.parquet \\
            --output_dir output_dir/rl_mind_auc \\
            --reward_type mind_auc \\
            --max_response_length 8 \\
            --total_epochs 1

        # Chain-of-Thought RL (reasoning + answer)
        python src/rl_mind_verl.py \\
            --model_path output_dir/sft_mind_ranking/final_checkpoint \\
            --train_parquet ../data/MIND/train/rl_cot_train.parquet \\
            --eval_parquet ../data/MIND/dev/rl_cot_dev.parquet \\
            --output_dir output_dir/rl_mind_cot \\
            --reward_type mind_cot_binary \\
            --max_response_length 256 \\
            --total_epochs 1 \\
            --train_batch_size 32 \\
            --n_gpus_per_node 8

        # CoT with margin-based reward (encourages confident correct answers)
        python src/rl_mind_verl.py \\
            --model_path output_dir/sft_mind_ranking/final_checkpoint \\
            --train_parquet ../data/MIND/train/rl_cot_train.parquet \\
            --eval_parquet ../data/MIND/dev/rl_cot_dev.parquet \\
            --output_dir output_dir/rl_mind_cot_margin \\
            --reward_type mind_cot_margin \\
            --max_response_length 256 \\
            --kl_loss_coef 0.1 \\
            --total_epochs 2
    """
    print("=" * 70)
    print("MIND RL Training with VERL")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Train data: {train_parquet}")
    print(f"Eval data: {eval_parquet}")
    print(f"Output: {output_dir}")
    print(f"Reward type: {reward_type}")
    print(f"Epochs: {total_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"KL coefficient: {kl_loss_coef}")
    print(f"Batch size: {train_batch_size}")
    print(f"GPUs: {n_gpus_per_node}")
    print("=" * 70)
    print()

    # Validate reward type
    valid_reward_types = [
        # Standard ranking (short output)
        'mind_auc', 'mind_auc_rank', 'mind_ndcg', 'mind_mrr',
        # Pointwise (Yes/No classification)
        'pointwise_binary', 'pointwise_weighted', 'pointwise_auc_proxy', 'pointwise_margin',
        'pointwise_asymmetric',
        # Chain-of-Thought (legacy single-answer)
        'mind_cot_binary', 'mind_cot_ndcg', 'mind_cot_auc', 'mind_cot_margin', 'mind_cot_format',
        # Chain-of-Thought (new prob-based <think>/<answer> format)
        'mind_cot_prob_auc', 'mind_cot_prob_combined', 'mind_cot_prob_ndcg', 'mind_cot_prob_ce', 'mind_cot_prob_margin', 'mind_cot_prob_format',
        # Pointwise CoT (<think>/<answer>Yes|No)
        'cot_pointwise_binary', 'cot_pointwise_format', 'cot_pointwise_asymmetric',
    ]
    if reward_type not in valid_reward_types:
        print(f"WARNING: reward_type '{reward_type}' not in {valid_reward_types}")
        print("Using default: mind_auc")
        reward_type = 'mind_auc'

    # Auto-adjust max_response_length for CoT if not explicitly set
    if 'cot' in reward_type and max_response_length < 64:
        print(f"NOTE: CoT reward detected. Consider using --max_response_length 256 or higher for reasoning output.")

    # Call VERL trainer
    train_verl(
        model_path=model_path,
        train_parquet=train_parquet,
        eval_parquet=eval_parquet,
        output_dir=output_dir,
        reward_type=reward_type,
        num_generations=num_generations,
        train_batch_size=train_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        learning_rate=learning_rate,
        total_epochs=total_epochs,
        temperature=temperature,
        rollout_name=rollout_name,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        kl_loss_coef=kl_loss_coef,
        kl_loss_type=kl_loss_type,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        save_freq=save_freq,
        test_freq=test_freq,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        sid_info_file=sid_info_file,
        ada_path=ada_path,
        cf_path=cf_path,
        sasrec_len_seq=sasrec_len_seq,
    )


if __name__ == "__main__":
    Fire(main)
