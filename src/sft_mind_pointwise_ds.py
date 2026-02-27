"""
Train a model on MIND dataset using Point-wise SFT with DeepSpeed support.

This script uses a point-wise approach where each (history, candidate) pair
is scored independently with a Yes/No classification.

Key differences from list-wise (sft_mind_ranking.py):
- Each candidate is evaluated independently
- Binary classification: "Is this article relevant? Yes/No"
- More training signal (every candidate gets a label)
- Shorter context per sample
- No position bias

Training format:
    Prompt: "User History: ... Candidate: [article] Is this relevant? Answer:"
    Target: " Yes" or " No"

Usage:
    deepspeed --hostfile=hostfile src/sft_mind_pointwise_ds.py \
        --base_model Qwen/Qwen3-1.7B \
        --train_behaviors_path ../data/MIND/train/behaviors.tsv \
        --train_news_path ../data/MIND/train/news.tsv \
        --eval_behaviors_path ../data/MIND/dev/behaviors.tsv \
        --eval_news_path ../data/MIND/dev/news.tsv \
        --output_dir output_dir/sft_mind_pointwise \
        --deepspeed_config ds_config_zero2.json
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    default_data_collator,
)
import fire

# Add parent directory and src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import MINDPointwiseSFTDataset
from mind_utils import load_news, build_pointwise_prompt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Custom Trainers ====================


class WeightedCETrainer(transformers.Trainer):
    """Trainer with class-weighted CE loss (higher weight on positive 'Yes' samples)."""

    def __init__(self, pos_weight=2.0, yes_token_id=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        self.yes_token_id = yes_token_id
        print(f"  WeightedCE: pos_weight={pos_weight}, yes_token_id={yes_token_id}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits

        # Standard causal LM shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token CE loss (ignore_index=-100 zeros out pad/prompt positions)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss = loss_fct(flat_logits, flat_labels)

        # Weight: pos_weight for Yes tokens, 1.0 for everything else
        valid_mask = (flat_labels != -100).float()
        weight = torch.ones_like(loss)
        if self.yes_token_id is not None:
            yes_mask = (flat_labels == self.yes_token_id).float()
            weight = weight + yes_mask * (self.pos_weight - 1.0)

        weighted_sum = (loss * weight * valid_mask).sum()
        weight_sum = (weight * valid_mask).sum().clamp(min=1.0)
        loss = weighted_sum / weight_sum

        return (loss, outputs) if return_outputs else loss


class PairwiseTrainer(transformers.Trainer):
    """Trainer with pairwise margin loss + CE loss for ranking-aware training."""

    def __init__(self, margin=1.0, yes_token_id=None, no_token_id=None, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        print(f"  PairwiseMargin: margin={margin}, yes_id={yes_token_id}, no_id={no_token_id}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_input_ids = inputs["pos_input_ids"]
        pos_labels = inputs["pos_labels"]
        pos_attention_mask = inputs["pos_attention_mask"]
        neg_input_ids = inputs["neg_input_ids"]
        neg_labels = inputs["neg_labels"]
        neg_attention_mask = inputs["neg_attention_mask"]

        B = pos_input_ids.size(0)

        # Concatenate pos and neg for a single efficient forward pass
        input_ids = torch.cat([pos_input_ids, neg_input_ids], dim=0)        # (2B, L)
        labels = torch.cat([pos_labels, neg_labels], dim=0)                 # (2B, L)
        attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (2B, L, V)

        # 1. Standard CE loss on all samples (teaches model to say Yes/No)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # 2. Pairwise margin loss (teaches model to rank correctly)
        # Find the answer prediction position for each sample
        answer_mask = (labels != -100)
        first_answer_pos = answer_mask.long().argmax(dim=1).clamp(min=1)
        pred_pos = first_answer_pos - 1  # logit[i] predicts token[i+1]

        batch_indices = torch.arange(2 * B, device=logits.device)
        answer_logits = logits[batch_indices, pred_pos, :]  # (2B, V)

        # Score = log P(Yes) - log P(No)
        log_probs = F.log_softmax(answer_logits, dim=-1)
        scores = log_probs[:, self.yes_token_id] - log_probs[:, self.no_token_id]

        pos_scores = scores[:B]
        neg_scores = scores[B:]

        margin_loss = F.relu(self.margin - (pos_scores - neg_scores)).mean()

        loss = ce_loss + margin_loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle pairwise inputs during evaluation."""
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)


# ==================== Pairwise Dataset ====================


class MINDPairwiseSFTDataset:
    """
    MIND dataset for pairwise margin loss training.
    Each sample is a (positive, negative) pair from the same impression.
    """

    def __init__(
        self,
        behaviors_path,
        news_path,
        tokenizer,
        max_len=2048,
        sample=-1,
        seed=42,
        max_history=0,
        neg_ratio=1.0,
        use_abstract=False,
        use_chat_template=False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history if max_history > 0 else None
        self.use_abstract = use_abstract
        self.use_chat_template = use_chat_template
        self.seed = seed

        self.news = load_news(news_path, use_abstract=use_abstract)
        self.pairs = self._load_and_create_pairs(behaviors_path, sample, neg_ratio)

        print(f"Loaded {len(self.pairs)} pairwise samples")

    def _load_and_create_pairs(self, behaviors_path, sample_limit, neg_ratio):
        all_pairs = []
        rng = random.Random(self.seed)

        with open(behaviors_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue

                history_ids = parts[3].split()
                impressions = parts[4].split()

                if self.max_history is not None:
                    history_ids = history_ids[-self.max_history :]

                history = [self.news[nid] for nid in history_ids if nid in self.news]

                positives = []
                negatives = []
                for imp in impressions:
                    if "-" not in imp:
                        continue
                    news_id, label = imp.rsplit("-", 1)
                    if news_id not in self.news:
                        continue
                    if int(label) == 1:
                        positives.append(news_id)
                    else:
                        negatives.append(news_id)

                if not positives or not negatives:
                    continue

                # 50/50 hard/easy negative split
                pos_cats = set(
                    self.news[pid].get("category", "") for pid in positives
                )
                hard_negs = [
                    n for n in negatives
                    if self.news[n].get("category", "") in pos_cats
                ]
                easy_negs = [
                    n for n in negatives
                    if self.news[n].get("category", "") not in pos_cats
                ]

                for pos_id in positives:
                    num_pairs = max(1, int(neg_ratio))
                    for _ in range(num_pairs):
                        if rng.random() < 0.5 and hard_negs:
                            neg_id = rng.choice(hard_negs)
                        elif easy_negs:
                            neg_id = rng.choice(easy_negs)
                        elif hard_negs:
                            neg_id = rng.choice(hard_negs)
                        else:
                            neg_id = rng.choice(negatives)

                        all_pairs.append({
                            "history": history,
                            "pos_candidate": self.news[pos_id],
                            "neg_candidate": self.news[neg_id],
                        })

        rng.shuffle(all_pairs)
        if 0 < sample_limit < len(all_pairs):
            all_pairs = all_pairs[:sample_limit]
        return all_pairs

    def _tokenize(self, history, candidate, label):
        """Build prompt, tokenize, and create training labels."""
        prompt = build_pointwise_prompt(
            history,
            candidate,
            tokenizer=self.tokenizer if self.use_chat_template else None,
            use_chat_template=self.use_chat_template,
        )
        target = " Yes" if label == 1 else " No"

        if self.use_chat_template:
            full_text = prompt + target
            input_ids = self.tokenizer.encode(
                full_text, max_length=self.max_len, truncation=True,
                add_special_tokens=False,
            )
            prompt_ids = self.tokenizer.encode(
                prompt, max_length=self.max_len, truncation=True,
                add_special_tokens=False,
            )
        else:
            full_text = prompt + target
            input_ids = self.tokenizer.encode(
                full_text, max_length=self.max_len, truncation=True,
                add_special_tokens=True,
            )
            prompt_ids = self.tokenizer.encode(
                prompt, max_length=self.max_len, truncation=True,
                add_special_tokens=True,
            )

        train_labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        # Right-pad to max_len
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            train_labels = train_labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids[: self.max_len], dtype=torch.long),
            "labels": torch.tensor(train_labels[: self.max_len], dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if tok != self.tokenizer.pad_token_id else 0
                 for tok in input_ids[: self.max_len]],
                dtype=torch.long,
            ),
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        pos = self._tokenize(pair["history"], pair["pos_candidate"], label=1)
        neg = self._tokenize(pair["history"], pair["neg_candidate"], label=0)
        return {
            "pos_input_ids": pos["input_ids"],
            "pos_labels": pos["labels"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_labels": neg["labels"],
            "neg_attention_mask": neg["attention_mask"],
        }


def train(
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    eval_behaviors_path: str = "",
    eval_news_path: str = "",
    output_dir: str = "",
    use_abstract: bool = False,
    max_history: int = 0,  # 0 = no limit
    neg_ratio: float = 1.0,  # Negatives per positive
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,  # Shorter than list-wise since single candidate
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_run_id: str = "",
    deepspeed_config: str = "",
    use_chat_template: bool = None,  # Auto-detect if None
    loss_type: str = "ce",  # "ce", "weighted_ce", "pairwise"
    label_smoothing: float = 0.0,  # Label smoothing factor (only for loss_type="ce")
    margin: float = 1.0,  # Margin for pairwise loss
    pos_weight: float = 2.0,  # Weight for positive (Yes) samples in weighted_ce
):
    """Train with point-wise SFT format (Yes/No classification) using DeepSpeed"""

    set_seed(seed)

    # Auto-detect if model is instruct variant (if use_chat_template not explicitly set)
    if use_chat_template is None:
        use_chat_template = "instruct" in base_model.lower() or "chat" in base_model.lower()
        if use_chat_template:
            print(f"Auto-detected instruct model: will use chat template")
        else:
            print(f"Using raw text format (no chat template)")

    # Resume existing WandB run if run_id is provided
    if wandb_run_id:
        os.environ['WANDB_RUN_ID'] = wandb_run_id
        os.environ['WANDB_RESUME'] = 'allow'
        print(f"Resuming WandB run: {wandb_run_id}")

    if not base_model:
        raise ValueError("Please specify --base_model")

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load model
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

    # Detect Yes/No token IDs (needed for weighted_ce and pairwise)
    yes_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]

    # Select dataset class based on loss type
    if loss_type == "pairwise":
        DatasetClass = MINDPairwiseSFTDataset
    else:
        DatasetClass = MINDPointwiseSFTDataset

    train_data = DatasetClass(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        neg_ratio=neg_ratio,
        use_abstract=use_abstract,
        use_chat_template=use_chat_template,
    )

    val_data = DatasetClass(
        behaviors_path=eval_behaviors_path,
        news_path=eval_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=min(5000, len(train_data) // 10) if sample <= 0 else min(1000, sample // 10),
        seed=seed,
        max_history=max_history,
        neg_ratio=neg_ratio,
        use_abstract=use_abstract,
        use_chat_template=use_chat_template,
    )

    print(f"\nTraining with Point-wise SFT ({loss_type}):")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print(f"  Neg ratio: {neg_ratio}")
    print(f"  Cutoff length: {cutoff_len}")
    print(f"  Loss type: {loss_type}")
    if loss_type == "weighted_ce":
        print(f"  Pos weight: {pos_weight}")
    elif loss_type == "pairwise":
        print(f"  Margin: {margin}")
    if label_smoothing > 0:
        print(f"  Label smoothing: {label_smoothing}")
    print(f"  Yes token ID: {yes_token_id}, No token ID: {no_token_id}")
    print(f"  Chat template: {'enabled' if use_chat_template else 'disabled (raw text)'}")

    # Prepare training arguments with optional DeepSpeed
    training_args_dict = {
        "per_device_train_batch_size": micro_batch_size,
        "per_device_eval_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": 100,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "bf16": True,
        "logging_steps": 10,
        "logging_first_step": True,
        "eval_strategy": "steps" if val_data else "no",
        "save_strategy": "steps",
        "eval_steps": 256 if val_data else None,
        "save_steps": 512,
        "output_dir": output_dir,
        "save_total_limit": 3,
        "load_best_model_at_end": True if val_data else False,
        "ddp_find_unused_parameters": False if ddp else None,
        "group_by_length": group_by_length,
        "report_to": "wandb" if wandb_project else "none",
        "run_name": wandb_run_name if wandb_run_name else None,
        "metric_for_best_model": "eval_loss" if val_data else None,
        "greater_is_better": False,
        "disable_tqdm": False,
    }

    # Add label smoothing (only effective for standard CE)
    if label_smoothing > 0 and loss_type == "ce":
        training_args_dict["label_smoothing_factor"] = label_smoothing

    # Pairwise mode needs to keep custom column names
    if loss_type == "pairwise":
        training_args_dict["remove_unused_columns"] = False

    # Add DeepSpeed config if provided
    if deepspeed_config and os.path.exists(deepspeed_config):
        training_args_dict["deepspeed"] = deepspeed_config
        print(f"Using DeepSpeed config: {deepspeed_config}")

    training_args = transformers.TrainingArguments(**training_args_dict)

    # Initialize trainer based on loss type
    callbacks = [EarlyStoppingCallback(early_stopping_patience=64)] if val_data else None
    seq2seq_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    if loss_type == "pairwise":
        trainer = PairwiseTrainer(
            margin=margin,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data if val_data else None,
            args=training_args,
            data_collator=default_data_collator,
            callbacks=callbacks,
        )
    elif loss_type == "weighted_ce":
        trainer = WeightedCETrainer(
            pos_weight=pos_weight,
            yes_token_id=yes_token_id,
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data if val_data else None,
            args=training_args,
            data_collator=seq2seq_collator,
            callbacks=callbacks,
        )
    else:
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data if val_data else None,
            args=training_args,
            data_collator=seq2seq_collator,
            callbacks=callbacks,
        )

    model.config.use_cache = False

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_checkpoint"))

    print(f"\n✓ Point-wise SFT training completed!")
    print(f"  Model saved to: {output_dir}/final_checkpoint")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_pointwise.sh {output_dir}/final_checkpoint dev")


if __name__ == "__main__":
    fire.Fire(train)
