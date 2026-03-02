import os
import subprocess
import sys


def _reward_name(reward_type):
    mapping = {
        "rule": "compute_score_rule",
        "ranking": "compute_score_ranking",
        "ranking_only": "compute_score_ranking_only",
        "semantic": "compute_score_semantic",
        "sasrec": "compute_score_sasrec",
        "mind_ndcg": "compute_score_mind_ndcg",
        "mind_mrr": "compute_score_mind_mrr",
        "mind_auc": "compute_score_mind_auc",
        "mind_auc_rank": "compute_score_mind_auc_rank",
        # Pointwise rewards
        "pointwise_binary": "compute_score_pointwise_binary",
        "pointwise_weighted": "compute_score_pointwise_weighted",
        "pointwise_auc_proxy": "compute_score_pointwise_auc_proxy",
        "pointwise_margin": "compute_score_pointwise_margin",
        "pointwise_asymmetric": "compute_score_pointwise_asymmetric",
        # Chain-of-Thought rewards (legacy single-answer)
        "mind_cot_binary": "compute_score_mind_cot_binary",
        "mind_cot_ndcg": "compute_score_mind_cot_ndcg",
        "mind_cot_auc": "compute_score_mind_cot_auc",
        "mind_cot_margin": "compute_score_mind_cot_margin",
        "mind_cot_format": "compute_score_mind_cot_format",
        # Chain-of-Thought rewards (new prob-based <think>/<answer> format)
        "mind_cot_prob_auc": "compute_score_mind_cot_prob_auc",
        "mind_cot_prob_ndcg": "compute_score_mind_cot_prob_ndcg",
        "mind_cot_prob_ce": "compute_score_mind_cot_prob_ce",
        "mind_cot_prob_margin": "compute_score_mind_cot_prob_margin",
        "mind_cot_prob_format": "compute_score_mind_cot_prob_format",
        # Pointwise CoT rewards (<think>/<answer>Yes|No)
        "cot_pointwise_binary": "compute_score_cot_pointwise_binary",
        "cot_pointwise_format": "compute_score_cot_pointwise_format",
        "cot_pointwise_asymmetric": "compute_score_cot_pointwise_asymmetric",
    }
    return mapping.get(reward_type, "compute_score_rule")


def _logger_list(use_wandb):
    return '["console","wandb"]' if use_wandb else '["console"]'


def train_verl(
    model_path: str,
    train_parquet: str,
    eval_parquet: str,
    output_dir: str,
    reward_type: str = "rule",
    num_generations: int = 16,
    train_batch_size: int = 1024,
    max_prompt_length: int = 512,
    max_response_length: int = 128,
    learning_rate: float = 1e-6,
    total_epochs: int = 1,
    temperature: float = 1.0,
    rollout_name: str = "vllm",
    ppo_mini_batch_size: int = 256,
    ppo_micro_batch_size_per_gpu: int = 2,
    kl_loss_coef: float = 0.001,
    kl_loss_type: str = "low_var_kl",
    wandb_project: str = "",
    wandb_run_name: str = "",
    save_freq: int = 20,
    test_freq: int = 5,
    nnodes: int = 1,
    n_gpus_per_node: int = 8,
    sid_info_file: str = "",
    ada_path: str = "",
    cf_path: str = "",
    sasrec_len_seq: int = 10,
):
    env = os.environ.copy()
    if sid_info_file:
        env["SID_INFO_FILE"] = sid_info_file
    if ada_path:
        env["ADA_PATH"] = ada_path
    if cf_path:
        env["SASREC_PATH"] = cf_path
    env["SASREC_LEN_SEQ"] = str(sasrec_len_seq)

    reward_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "verl_reward.py"))
    reward_name = _reward_name(reward_type)

    use_wandb = bool(wandb_project)
    project = wandb_project or "minionerec"
    experiment = wandb_run_name or f"minionerec_verl_{reward_type}"

    # Calculate total micro batch size for ref (per_gpu * n_gpus)
    ref_micro_batch_size_per_gpu = ppo_micro_batch_size_per_gpu

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_parquet}",
        f"data.val_files={eval_parquet}",
        f"data.train_batch_size={train_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        f"actor_rollout_ref.rollout.temperature={temperature}",
        f"actor_rollout_ref.rollout.n={num_generations}",
        f"++actor_rollout_ref.rollout.name={rollout_name}",
        "++actor_rollout_ref.rollout.tensor_model_parallel_size=1",  # Disable tensor parallelism for small models
        "++actor_rollout_ref.model.override_config.attn_implementation=sdpa",  # Use PyTorch SDPA instead of flash_attention_2
        "++data.chat_template_kwargs.enable_thinking=False",  # Disable Qwen3 native <think> to avoid conflict with our <think> tags
        "reward_model.enable=False",  # Disable built-in reward model; we use custom_reward_function
        f"++reward_model.rollout.name={rollout_name}",  # Satisfy mandatory field even when disabled
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
        f"actor_rollout_ref.actor.kl_loss_type={kl_loss_type}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu}",
        f"+actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={ref_micro_batch_size_per_gpu}",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.project_name={project}",
        f"trainer.experiment_name={experiment}",
        f"trainer.logger={_logger_list(use_wandb)}",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.nnodes={nnodes}",
        f"trainer.n_gpus_per_node={n_gpus_per_node}",
        f"trainer.default_local_dir={output_dir}",
        f"custom_reward_function.path={reward_path}",
        f"custom_reward_function.name={reward_name}",
    ]

    print("Launching VERL with:")
    print(" ".join(cmd))

    # Run with explicit output handling
    result = subprocess.run(cmd, env=env, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nVERL training failed with return code: {result.returncode}")
        print("Check Ray logs for detailed errors:")
        print("  ls -ltr /tmp/ray/session_latest*/logs/")
        print("  tail /tmp/ray/session_latest*/logs/worker-*.err")
        raise subprocess.CalledProcessError(result.returncode, cmd)
