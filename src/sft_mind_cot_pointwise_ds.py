#!/usr/bin/env python3
"""
SFT training script for Pointwise Chain-of-Thought (CoT) news recommendation.

Teaches a base model the <think>...</think><answer>Yes/No</answer> format
using synthetic training data from prepare_mind_sft_cot_pointwise.py.

Quick SFT — typically 2 epochs on ~5000 samples — to teach the format
before RL (rl_mind_cot_pointwise.sh).

Key differences from ranking CoT SFT (sft_mind_cot_ds.py):
  - Output: <answer>Yes/No</answer> instead of <answer>[1:prob, ...]</answer>
  - Shorter responses (~100 tokens vs ~500)
  - More samples (pointwise expansion)
  - enable_thinking=False for Qwen3 (avoids native <think> tag conflict)

Usage:
    deepspeed --num_gpus 8 src/sft_mind_cot_pointwise_ds.py \
        --base_model Qwen/Qwen3-1.7B \
        --train_file /path/to/sft_cot_pw_train.jsonl \
        --output_dir output_dir/sft_mind_cot_pointwise

Author: MiniOneRec
"""

import os
import sys
import json
import random
import re
import math
from functools import partial

import numpy as np
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
)
import fire


class SaveTokenizerCallback(TrainerCallback):
    """Save tokenizer alongside every checkpoint so it can be loaded standalone."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            self.tokenizer.save_pretrained(checkpoint_dir)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MINDCoTPointwiseSFTDataset:
    """
    Dataset for Pointwise CoT format SFT training.

    Loads JSONL with chat messages (system + user + assistant),
    tokenizes with chat template (enable_thinking=False),
    and masks prompt tokens (-100).

    Masking strategy:
      - Prompt (system + user):  -100  (no loss)
      - "<think>\n":             SUPERVISED  (learn to open think tag)
      - reasoning content:       -100  (no loss — RL will learn this)
      - "\n</think>\n":          SUPERVISED  (learn to close think tag)
      - "<answer>Yes/No</answer>": SUPERVISED  (learn answer format)
      - EOS:                     SUPERVISED
    """

    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_len: int = 1024,
        sample: int = -1,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

        # Load JSONL
        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        if 0 < sample < len(self.samples):
            random.seed(seed)
            self.samples = random.sample(self.samples, sample)

        # Pre-tokenize
        self._tokenized = []
        skipped = 0
        for item in self.samples:
            result = self._tokenize(item)
            if result is not None:
                self._tokenized.append(result)
            else:
                skipped += 1

        if skipped > 0:
            print(f"Skipped {skipped} samples exceeding max_len={max_len}")
        print(f"CoT Pointwise SFT Dataset: {len(self._tokenized)} samples from {data_file}")

    def _tokenize(self, item):
        """
        Tokenize with selective loss masking.

        Supervise: <think>\n, \n</think>\n, <answer>...</answer>, EOS
        Mask: prompt, reasoning content inside <think>...</think>
        """
        messages = item["messages"]

        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        assistant_message = next((m for m in messages if m["role"] == "assistant"), None)
        if assistant_message is None:
            return None

        assistant_content = assistant_message["content"]

        # Apply chat template with enable_thinking=False to avoid Qwen3 native thinking
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        full_text = prompt_text + assistant_content + self.tokenizer.eos_token

        # Tokenize
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full_encoding["input_ids"].squeeze(0)
        attention_mask = full_encoding["attention_mask"].squeeze(0)

        # Check truncation
        full_len = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        if full_len > self.max_len:
            return None

        # Build labels with selective masking
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Parse assistant_content for <think>...</think> boundaries
        think_match = re.search(r'(<think>\n)(.*?)(\n</think>\n)', assistant_content, re.DOTALL)

        if think_match:
            think_open = assistant_content[:think_match.end(1)]    # "<think>\n"
            reasoning = think_match.group(2)                        # reasoning content

            think_open_ids = self.tokenizer.encode(
                prompt_text + think_open, add_special_tokens=False
            )
            reasoning_end_ids = self.tokenizer.encode(
                prompt_text + think_open + reasoning, add_special_tokens=False
            )

            think_open_end = len(think_open_ids)
            reasoning_end = len(reasoning_end_ids)

            labels = input_ids.clone()
            labels[:prompt_len] = -100               # mask prompt
            labels[think_open_end:reasoning_end] = -100  # mask reasoning
        else:
            # No <think> found — supervise entire response
            labels = input_ids.clone()
            labels[:prompt_len] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __len__(self):
        return len(self._tokenized)

    def __getitem__(self, idx):
        return self._tokenized[idx]


def verify_tokenization(dataset, num_samples=3):
    """Quick verification that labels are correctly masked."""
    print("=" * 70)
    print("Tokenization Verification (pointwise CoT selective masking)")
    print("=" * 70)
    print("  Masking strategy:")
    print("    prompt            → -100 (no loss)")
    print("    <think>\\n         → SUPERVISED")
    print("    reasoning content → -100 (no loss)")
    print("    \\n</think>\\n      → SUPERVISED")
    print("    <answer>Yes/No</answer> → SUPERVISED")
    print()

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        input_ids = item["input_ids"]
        labels = item["labels"]
        attention_mask = item["attention_mask"]

        seq_len = attention_mask.sum().item()
        num_target = sum(1 for l in labels if l != -100)
        num_masked = seq_len - num_target

        target_ids = [input_ids[j].item() for j in range(len(labels)) if labels[j] != -100]
        target_text = dataset.tokenizer.decode(target_ids)

        print(f"  Sample {i}: seq_len={seq_len}, supervised={num_target}, masked={num_masked}")

        has_think_open = "<think>" in target_text
        has_think_close = "</think>" in target_text
        has_answer = "<answer>" in target_text
        has_answer_close = "</answer>" in target_text
        print(f"    tags: <think>={has_think_open} </think>={has_think_close} "
              f"<answer>={has_answer} </answer>={has_answer_close}")

        if not all([has_think_open, has_think_close, has_answer, has_answer_close]):
            print(f"    WARNING: Missing format tags!")
            print(f"    Supervised text: {target_text[:300]}")
        else:
            print(f"    Supervised text preview: {target_text[:200]}...")

    print("=" * 70)
    print()


class _StackCollator:
    """Simple collator that stacks pre-tokenized tensors."""
    def __call__(self, batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }


def train(
    # Model
    base_model: str = "",
    train_from_scratch: bool = False,

    # Data
    train_file: str = "",
    eval_file: str = "",

    # Output
    output_dir: str = "output_dir/sft_mind_cot_pointwise",

    # Training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    cutoff_len: int = 1024,
    warmup_steps: int = 20,

    # Sampling
    sample: int = -1,
    eval_sample: int = 500,
    seed: int = 42,

    # WandB
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_run_id: str = "",

    # DeepSpeed
    deepspeed_config: str = "",

    # Misc
    resume_from_checkpoint: str = None,
):
    """
    Quick SFT to teach pointwise CoT output format.

    Trains model to output <think>...</think><answer>Yes/No</answer>.
    Typically 2 epochs is enough to teach the format before RL.
    """
    set_seed(seed)

    if not base_model:
        raise ValueError("Please specify --base_model")
    if not train_file:
        raise ValueError("Please specify --train_file")

    if wandb_project:
        os.environ['WANDB_PROJECT'] = wandb_project
    if wandb_run_id:
        os.environ['WANDB_RUN_ID'] = wandb_run_id
        os.environ['WANDB_RESUME'] = 'allow'

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    print(f"Loading model: {base_model}")

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load datasets
    print(f"Loading training data: {train_file}")
    train_data = MINDCoTPointwiseSFTDataset(
        data_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
    )

    val_data = None
    if eval_file and os.path.exists(eval_file):
        print(f"Loading eval data: {eval_file}")
        val_data = MINDCoTPointwiseSFTDataset(
            data_file=eval_file,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=eval_sample,
            seed=seed,
        )
    elif len(train_data) > 100:
        split_idx = int(len(train_data._tokenized) * 0.9)
        val_items = train_data._tokenized[split_idx:]
        train_data._tokenized = train_data._tokenized[:split_idx]

        val_data = MINDCoTPointwiseSFTDataset.__new__(MINDCoTPointwiseSFTDataset)
        val_data.tokenizer = tokenizer
        val_data.max_len = cutoff_len
        val_data._tokenized = val_items
        print(f"Auto-split: {len(train_data)} train, {len(val_data)} eval")

    print(f"\nPointwise CoT Format SFT Training:")
    print(f"  Base model: {base_model}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data) if val_data else 0}")
    print(f"  Batch size: {batch_size}")
    print(f"  Micro batch: {micro_batch_size}")
    print(f"  Grad accum: {gradient_accumulation_steps}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Cutoff len: {cutoff_len}")
    print()

    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        verify_tokenization(train_data)

    eval_step = max(1, len(train_data) // (batch_size * 4))
    save_step = max(1, len(train_data) // (batch_size * 2))
    if val_data and eval_step > 0:
        save_step = max(eval_step, (save_step // eval_step) * eval_step)

    training_args_dict = {
        "per_device_train_batch_size": micro_batch_size,
        "per_device_eval_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "bf16": True,
        "logging_steps": 1,
        "logging_first_step": True,
        "eval_strategy": "steps" if val_data else "no",
        "save_strategy": "steps",
        "eval_steps": eval_step if val_data else None,
        "save_steps": save_step,
        "output_dir": output_dir,
        "save_total_limit": 2,
        "load_best_model_at_end": True if val_data else False,
        "ddp_find_unused_parameters": False if ddp else None,
        "report_to": "wandb" if wandb_project else "none",
        "run_name": wandb_run_name if wandb_run_name else None,
        "metric_for_best_model": "eval_loss" if val_data else None,
        "greater_is_better": False,
        "disable_tqdm": False,
        "gradient_checkpointing": True,
    }

    if deepspeed_config and os.path.exists(deepspeed_config):
        training_args_dict["deepspeed"] = deepspeed_config
        print(f"Using DeepSpeed config: {deepspeed_config}")

    callbacks = [SaveTokenizerCallback(tokenizer)]
    if val_data:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(**training_args_dict),
        data_collator=_StackCollator(),
        callbacks=callbacks,
    )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    config_info = {
        "base_model": base_model,
        "task": "sft_cot_pointwise_format",
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "train_file": train_file,
        "cutoff_len": cutoff_len,
    }
    with open(os.path.join(final_path, "sft_cot_pointwise_config.json"), 'w') as f:
        json.dump(config_info, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Pointwise CoT Format SFT completed!")
    print(f"Model saved to: {final_path}")
    print(f"{'='*70}")
    print()
    print("Next step: Run RL training with this SFT checkpoint:")
    print(f"  SFT_MODEL={final_path} bash scripts/rl_mind_cot_pointwise.sh")


if __name__ == "__main__":
    fire.Fire(train)
