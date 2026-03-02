#!/usr/bin/env python3
"""
Evaluate MIND models trained with Pointwise CoT RL.

For each impression, scores every candidate independently by:
1. Building a pointwise CoT prompt (history + 1 candidate)
2. Generating <think>...</think><answer>Yes/No</answer>
3. Using P("Yes") token logit as ranking score (Method B — continuous scoring)

This produces a ranking over all candidates and computes MIND metrics.

Usage:
    python evaluate_mind_cot_pointwise.py \\
        --model_path output_dir/rl_mind_cot_pointwise/final_checkpoint \\
        --behaviors_path ../data/MIND/dev/behaviors.tsv \\
        --news_path ../data/MIND/dev/news.tsv \\
        --cot_style category --disable_thinking

    # Quick test (100 impressions)
    python evaluate_mind_cot_pointwise.py --model_path ... --max_impressions 100
"""

import argparse
import os
import re
import sys
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mind_utils import load_news, auc_score, mrr_score, ndcg_score
from prepare_mind_rl_cot_pointwise import build_cot_pointwise_prompt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_prompt(messages: list, tokenizer, enable_thinking: bool = True) -> str:
    """Apply chat template to messages."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def score_candidate_generative(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 256,
) -> Tuple[float, str]:
    """
    Generate CoT response and extract score.

    Returns:
        (score, generated_text) where score is:
        - 1.0 if model answered Yes
        - 0.0 if model answered No
        - 0.5 if unparseable (neutral)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract answer
    text = generated_text.strip()
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()

    answer_match = re.search(r'<answer>\s*(Yes|No)\s*</answer>', text, re.IGNORECASE)
    if answer_match:
        return (1.0 if answer_match.group(1).lower() == 'yes' else 0.0), generated_text

    if re.search(r'\byes\b', text, re.IGNORECASE):
        return 1.0, generated_text
    elif re.search(r'\bno\b', text, re.IGNORECASE):
        return 0.0, generated_text

    return 0.5, generated_text


def score_candidate_logit(
    model,
    tokenizer,
    prompt: str,
    device,
    yes_token_id: int,
    no_token_id: int,
) -> float:
    """
    Score a candidate by computing P(Yes) - P(No) from next-token logits.

    This is faster than generation (no autoregressive decoding) and provides
    a continuous score for better ranking.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        score = (log_probs[0, yes_token_id] - log_probs[0, no_token_id]).item()

    return score


def batch_score_logit(
    model,
    tokenizer,
    prompts: List[str],
    device,
    yes_token_id: int,
    no_token_id: int,
    batch_size: int = 4,
) -> List[float]:
    """Score multiple candidates in batches using logit method."""
    all_scores = []

    all_encodings = [
        tokenizer.encode(p, add_special_tokens=False, truncation=True, max_length=4096)
        for p in prompts
    ]

    for batch_start in range(0, len(all_encodings), batch_size):
        batch_ids = all_encodings[batch_start:batch_start + batch_size]

        max_len = max(len(ids) for ids in batch_ids)
        padded_ids = []
        attention_masks = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            scores = (log_probs[:, yes_token_id] - log_probs[:, no_token_id]).cpu().tolist()
            all_scores.extend(scores)

    return all_scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MIND model with Pointwise CoT"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--cot_style", default="standard", choices=["standard", "category", "detailed"])
    parser.add_argument("--max_history", type=int, default=30)
    parser.add_argument("--max_impressions", type=int, default=0, help="0=all")
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="", help="Save predictions for MIND leaderboard")
    parser.add_argument("--scoring", choices=["logit", "generative"], default="logit",
                        help="Scoring method: 'logit' (fast, P(Yes)-P(No)) or 'generative' (slow, full CoT generation)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for logit scoring (only used with --scoring logit)")
    parser.add_argument("--cot_max_tokens", type=int, default=256,
                        help="Max tokens for generative scoring")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"Loaded {len(news)} news articles")

    # Load model
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model on device: {device}")

    # Get Yes/No token IDs (for logit scoring)
    yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("No", add_special_tokens=False)
    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]
    print(f"Yes token: {yes_token_id} ('{tokenizer.decode([yes_token_id])}')")
    print(f"No token: {no_token_id} ('{tokenizer.decode([no_token_id])}')")

    enable_thinking = not args.disable_thinking
    print(f"Scoring: {args.scoring}, Thinking: {'enabled' if enable_thinking else 'disabled'}")
    print(f"CoT style: {args.cot_style}")
    print()

    # Metrics
    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs, mrrs, ndcg5, ndcg10 = [], [], [], []
    predictions = []
    count = 0
    skipped = 0

    # Count lines
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    total = min(total_lines, args.max_impressions) if args.max_impressions > 0 else total_lines

    start_time = time.time()

    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        pbar = tqdm(f, total=total, desc="Evaluating")
        for line in pbar:
            if args.max_impressions > 0 and count >= args.max_impressions:
                break

            parts = line.strip().split('\t')
            if len(parts) < 5:
                skipped += 1
                continue

            impression_id = parts[0]
            history_ids = parts[3].split()[-args.max_history:] if parts[3] else []
            impressions = parts[4].split()

            # Parse candidates
            candidate_ids = []
            labels = []
            for imp in impressions:
                if '-' not in imp:
                    continue
                nid, label = imp.rsplit('-', 1)
                if nid in news:
                    candidate_ids.append(nid)
                    labels.append(int(label))

            if not candidate_ids or sum(labels) == 0:
                skipped += 1
                continue

            # Build history
            history_items = [news[nid] for nid in history_ids if nid in news]
            candidate_objs = [news[nid] for nid in candidate_ids]

            # Score each candidate
            if args.scoring == "logit":
                # Build all prompts, score with logits in batch
                prompts = []
                for cand in candidate_objs:
                    messages = build_cot_pointwise_prompt(history_items, cand, args.cot_style)
                    prompt = format_prompt(messages, tokenizer, enable_thinking)
                    prompts.append(prompt)

                scores = batch_score_logit(
                    model, tokenizer, prompts, device,
                    yes_token_id, no_token_id, args.batch_size
                )
            else:
                # Generative: full CoT for each candidate (slow)
                scores = []
                for cand in candidate_objs:
                    messages = build_cot_pointwise_prompt(history_items, cand, args.cot_style)
                    prompt = format_prompt(messages, tokenizer, enable_thinking)
                    score, _ = score_candidate_generative(
                        model, tokenizer, prompt, device, args.cot_max_tokens
                    )
                    scores.append(score)

            # Compute metrics
            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5.append(ndcg_score(labels, scores, 5))
                ndcg10.append(ndcg_score(labels, scores, 10))

            # Predictions
            if args.output_file:
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_ids))

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{_avg(aucs):.4f}',
                'MRR': f'{_avg(mrrs):.4f}',
                'nDCG@5': f'{_avg(ndcg5):.4f}',
                'nDCG@10': f'{_avg(ndcg10):.4f}'
            })

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print(f"Results ({count} impressions, {elapsed:.1f}s, {count / max(elapsed, 0.1):.1f} imp/s)")
    print("=" * 60)
    print(f"  AUC:     {_avg(aucs):.4f}")
    print(f"  MRR:     {_avg(mrrs):.4f}")
    print(f"  nDCG@5:  {_avg(ndcg5):.4f}")
    print(f"  nDCG@10: {_avg(ndcg10):.4f}")
    print(f"  Skipped: {skipped}")
    print("=" * 60)

    if args.output_file and predictions:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        with open(args.output_file, 'w') as f:
            for imp_id, ranked_ids in predictions:
                ranks = ",".join(ranked_ids)
                f.write(f"{imp_id}\t[{ranks}]\n")
        print(f"Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
