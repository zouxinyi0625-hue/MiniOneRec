#!/usr/bin/env python3
"""
Evaluate MIND models trained with CoT (Chain-of-Thought) SFT / RL.

Key difference from evaluate_mind_ranking.py:
  - Prompt: Uses the same CoT system prompt with <think>/<answer> format
    (identical to prepare_mind_rl_cot.py / prepare_mind_sft_cot.py)
  - Generation: model.generate() to produce full CoT reasoning + answer
  - Answer post-processing: Parse <answer>[1:prob, 2:prob, ...]</answer>
    to extract per-candidate click probabilities
  - Scoring: Use extracted probabilities directly as scores for AUC/MRR/nDCG

Metrics: AUC, MRR, nDCG@5, nDCG@10 — same as other MIND eval scripts.

Usage:
    # Single GPU
    python src/evaluate_mind_cot.py \
        --model_path output_dir/sft_mind_cot/final_checkpoint \
        --behaviors_path /path/to/MIND/dev/behaviors.tsv \
        --news_path /path/to/MIND/dev/news.tsv

    # Multi-GPU via shell script
    bash scripts/eval_mind_cot_prob.sh output_dir/rl_mind_cot/final_checkpoint dev
"""

import argparse
import json
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add src dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mind_utils import auc_score, mrr_score, ndcg_score


# =============================================================================
# Data loading
# =============================================================================

def load_news(news_path: str, use_abstract: bool = False) -> Dict[str, Dict[str, str]]:
    """Load news articles from news.tsv."""
    news = {}
    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            news_id = parts[0]
            category = parts[1] if len(parts) > 1 else ""
            title = parts[3]
            abstract = parts[4] if len(parts) > 4 else ""
            text = f"{title} {abstract}" if use_abstract and abstract else title
            news[news_id] = {"title": title, "text": text, "category": category}
    return news


# =============================================================================
# Prompt building — MUST match prepare_mind_rl_cot.py / prepare_mind_sft_cot.py
# =============================================================================

def build_cot_system_prompt(num_cands: int) -> str:
    """Build CoT system prompt (identical to prepare_mind_rl_cot.py)."""
    system_content = (
        "[Role]\n"
        "You are a news recommendation assistant.\n"
        "\n"
        "[Task]\n"
        "Given a user's [Reading History] and a list of [Candidate Articles], "
        "follow the [Instructions] to analyze user interests and predict "
        "the click probability for each candidate article.\n"
        "\n"
        "[Output Format]\n"
        "You MUST respond with exactly two sections:\n"
        "\n"
        "1. Reasoning section — wrap your step-by-step analysis in <think> tags:\n"
        "   <think>\n"
        "   ... your analysis here ...\n"
        "   </think>\n"
        "\n"
        "2. Answer section — wrap the click probabilities in <answer> tags:\n"
        "   <answer>\n"
    )
    if num_cands <= 3:
        prob_example = ", ".join(f"{i}:prob" for i in range(1, num_cands + 1))
    else:
        prob_example = f"1:prob, 2:prob, ..., {num_cands}:prob"
    system_content += (
        f"   [{prob_example}]\n"
        "   </answer>\n"
        "\n"
        "[Output Rules]\n"
        f"- You MUST list ALL {num_cands} candidates in the answer\n"
        "- Each probability is a float between 0.0 and 1.0\n"
        "- Higher probability = more likely to be clicked\n"
        "- Probabilities do NOT need to sum to 1\n"
        "- Do NOT output anything after </answer>\n"
    )
    return system_content


def build_cot_user_prompt(
    history_items: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
    cot_style: str = "standard",
) -> str:
    """Build CoT user prompt (identical to prepare_mind_rl_cot.py)."""
    lines = []

    lines.append("[Reading History]")
    if history_items:
        for i, item in enumerate(history_items, 1):
            cat = f"[{item.get('category', 'General')}] " if item.get('category') else ""
            lines.append(f"{i}. {cat}{item['text']}")
    else:
        lines.append("(No reading history available)")
    lines.append("")

    lines.append("[Candidate Articles]")
    for i, cand in enumerate(candidates, 1):
        cat = f"[{cand.get('category', 'General')}] " if cand.get('category') else ""
        lines.append(f"{i}. {cat}{cand['text']}")
    lines.append("")

    if cot_style == "category":
        lines.append("[Instructions]")
        lines.append("Analyze the candidates based on category matching:")
        lines.append("1. Identify the main categories/topics from [Reading History]")
        lines.append("2. Note interest patterns (e.g., sports, politics, technology)")
        lines.append("3. For each article in [Candidate Articles], assess category relevance")
        lines.append("4. Assign click probability to EVERY candidate")
    elif cot_style == "detailed":
        lines.append("[Instructions]")
        lines.append("Perform a detailed analysis in four steps:")
        lines.append("Step 1: Summarize user interests from [Reading History]")
        lines.append("Step 2: List the key topics/categories the user prefers")
        lines.append("Step 3: Evaluate each article in [Candidate Articles] for relevance")
        lines.append("Step 4: Assign click probability to EVERY candidate")
    else:  # standard
        lines.append("[Instructions]")
        lines.append("Based on [Reading History], think step by step about what topics interest this user.")
        lines.append("Then estimate the click probability for each article in [Candidate Articles].")

    return "\n".join(lines)


def build_cot_chat_prompt(
    tokenizer,
    history_items: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
    cot_style: str = "standard",
) -> str:
    """Build full prompt string using chat template."""
    system_content = build_cot_system_prompt(len(candidates))
    user_content = build_cot_user_prompt(history_items, candidates, cot_style)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# =============================================================================
# Answer extraction — reuse the same logic as verl_reward.py
# =============================================================================

def extract_cot_probs(solution_str: str, num_candidates: int = None) -> Dict[int, float]:
    """
    Extract click probabilities from <answer>[1:0.8, 2:0.1, ...]</answer>.

    Identical to verl_reward.py::extract_cot_probs.

    Returns:
        dict[int, float]: candidate_id (1-indexed) → probability.
        Empty dict if parsing fails.
    """
    if not solution_str:
        return {}

    text = str(solution_str).strip()

    # Extract content between <answer> and </answer>
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if not answer_match:
        return {}

    answer_content = answer_match.group(1).strip().strip('[]')

    # Parse "id:prob" pairs
    pairs = re.findall(r'(\d+)\s*:\s*([0-9]*\.?[0-9]+)', answer_content)
    if not pairs:
        return {}

    probs = {}
    for cid_str, prob_str in pairs:
        cid = int(cid_str)
        try:
            prob = float(prob_str)
        except ValueError:
            continue
        prob = max(0.0, min(1.0, prob))
        if num_candidates is not None and (cid < 1 or cid > num_candidates):
            continue
        probs[cid] = prob

    return probs


def probs_to_scores(probs: Dict[int, float], num_candidates: int) -> List[float]:
    """
    Convert extracted prob dict to a scores list aligned with candidate indices.

    Args:
        probs: {1: 0.8, 2: 0.3, ...} (1-indexed)
        num_candidates: total number of candidates

    Returns:
        List of length num_candidates with probabilities (0.0 for missing).
    """
    return [probs.get(i + 1, 0.0) for i in range(num_candidates)]


# =============================================================================
# Generation
# =============================================================================

def generate_cot_response(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate full CoT response with greedy decoding.

    Returns the generated text (excluding the prompt).
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# =============================================================================
# Candidate limiting (match training distribution)
# =============================================================================

def limit_candidates(
    candidate_ids: List[str],
    labels: List[int],
    news: Dict,
    max_candidates: int,
    seed: int,
    impression_id: str,
) -> Tuple[List[str], List[int]]:
    """
    Limit candidates to max_candidates (keep all clicked + sample non-clicked).
    Uses the same shuffling logic as prepare_mind_rl_cot.py.
    """
    if max_candidates <= 0 or len(candidate_ids) <= max_candidates:
        return candidate_ids, labels

    clicked_idx = [i for i, l in enumerate(labels) if l == 1]
    non_clicked_idx = [i for i, l in enumerate(labels) if l == 0]

    rng = random.Random(f"{impression_id}-{seed}")

    keep = clicked_idx.copy()
    remaining = max_candidates - len(clicked_idx)
    if remaining > 0 and non_clicked_idx:
        sampled = rng.sample(non_clicked_idx, min(remaining, len(non_clicked_idx)))
        keep.extend(sampled)

    rng.shuffle(keep)

    new_ids = [candidate_ids[i] for i in keep]
    new_labels = [labels[i] for i in keep]
    return new_ids, new_labels


# =============================================================================
# Main evaluation loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoT (Chain-of-Thought) models on MIND dataset"
    )
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--behaviors_path", required=True, help="Path to behaviors.tsv")
    parser.add_argument("--news_path", required=True, help="Path to news.tsv")
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=30)
    parser.add_argument("--max_candidates", type=int, default=30,
                        help="Max candidates per impression (0=unlimited). Should match training.")
    parser.add_argument("--max_impressions", type=int, default=0,
                        help="Limit number of impressions (0=all)")
    parser.add_argument("--cot_style", type=str, default="standard",
                        choices=["standard", "category", "detailed"])
    parser.add_argument("--cot_max_tokens", type=int, default=512,
                        help="Max tokens for CoT generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="",
                        help="Save predictions to file")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Use Flash Attention 2")
    parser.add_argument("--verbose", action="store_true",
                        help="Print sample generations for debugging")
    parser.add_argument("--verbose_count", type=int, default=5,
                        help="Number of verbose samples to print")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load news ──
    print(f"Loading news: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"  Loaded {len(news)} news articles")

    # ── Load model ──
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    device = model.device

    # ── Count lines for progress bar ──
    total_lines = 0
    with open(args.behaviors_path, "r") as f:
        for _ in f:
            total_lines += 1
    total_to_process = min(total_lines, args.max_impressions) if args.max_impressions > 0 else total_lines

    # ── Print config ──
    print()
    print("=" * 60)
    print("MIND CoT Evaluation (Prob-based)")
    print("=" * 60)
    print(f"  Model:          {args.model_path}")
    print(f"  Behaviors:      {args.behaviors_path}")
    print(f"  CoT style:      {args.cot_style}")
    print(f"  Max tokens:     {args.cot_max_tokens}")
    print(f"  Max history:    {args.max_history}")
    print(f"  Max candidates: {args.max_candidates if args.max_candidates > 0 else 'unlimited'}")
    print(f"  Use abstract:   {args.use_abstract}")
    print(f"  Flash Attn:     {args.flash_attn}")
    print(f"  Impressions:    {total_to_process}")
    print("=" * 60)
    print()

    # ── Evaluate ──
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    predictions = []

    # Format diagnostics
    total_generated = 0
    format_ok = 0        # has both <think> and <answer> with parseable probs
    parse_ok = 0         # extracted at least 1 prob
    full_coverage = 0    # extracted probs for ALL candidates

    count = 0
    skipped = 0
    verbose_printed = 0

    _avg = lambda xs: float(np.mean(xs)) if xs else 0.0

    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_to_process, desc="Evaluating", unit="imp")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                skipped += 1
                continue

            impression_id = parts[0]
            history_ids = parts[3].split()
            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]
            impressions = parts[4].split()

            candidate_ids = []
            labels = []
            for imp in impressions:
                if "-" not in imp:
                    continue
                nid, label = imp.rsplit("-", 1)
                candidate_ids.append(nid)
                labels.append(int(label))

            if not candidate_ids or sum(labels) == 0:
                skipped += 1
                continue

            # Limit candidates
            if args.max_candidates > 0:
                candidate_ids, labels = limit_candidates(
                    candidate_ids, labels, news,
                    args.max_candidates, args.seed, impression_id
                )

            # Build candidate / history objects
            candidate_objs = [news.get(nid, {"text": "[MISSING]", "category": ""})
                              for nid in candidate_ids]
            history_objs = [news[nid] for nid in history_ids if nid in news]

            num_cands = len(candidate_objs)

            # ── Build prompt ──
            prompt = build_cot_chat_prompt(
                tokenizer, history_objs, candidate_objs, args.cot_style
            )

            # ── Generate ──
            generated_text = generate_cot_response(
                model, tokenizer, prompt, device, args.cot_max_tokens
            )
            total_generated += 1

            # ── Parse answer ──
            probs = extract_cot_probs(generated_text, num_cands)
            scores = probs_to_scores(probs, num_cands)

            # ── Format diagnostics ──
            has_think = bool(re.search(r'<think>.*?</think>', generated_text, re.DOTALL))
            has_answer = bool(re.search(r'<answer>.*?</answer>', generated_text, re.DOTALL))
            if has_think and has_answer and len(probs) > 0:
                format_ok += 1
            if len(probs) > 0:
                parse_ok += 1
            if len(probs) == num_cands:
                full_coverage += 1

            # ── Verbose output ──
            if args.verbose and verbose_printed < args.verbose_count:
                verbose_printed += 1
                print(f"\n--- Impression {impression_id} ({num_cands} cands) ---")
                print(f"Generated ({len(generated_text)} chars):")
                print(generated_text[:600])
                if len(generated_text) > 600:
                    print("...")
                print(f"Parsed probs: {probs}")
                print(f"Labels:       {labels}")
                print(f"Scores:       {scores}")
                print()

            # ── Metrics ──
            if sum(labels) > 0 and len(probs) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5s.append(ndcg_score(labels, scores, 5))
                ndcg10s.append(ndcg_score(labels, scores, 10))
            elif sum(labels) > 0:
                # No probs extracted → random baseline
                aucs.append(0.5)
                mrrs.append(0.0)
                ndcg5s.append(0.0)
                ndcg10s.append(0.0)

            # ── Predictions ──
            if args.output_file:
                ranked_indices = sorted(range(num_cands), key=lambda i: scores[i], reverse=True)
                ranked_nids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_nids))

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                "AUC": f"{_avg(aucs):.4f}",
                "MRR": f"{_avg(mrrs):.4f}",
                "nDCG@5": f"{_avg(ndcg5s):.4f}",
            })

            if args.max_impressions > 0 and count >= args.max_impressions:
                break

        pbar.close()

    # ── Results ──
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Impressions evaluated: {count}")
    if skipped:
        print(f"Skipped (malformed/no-positive): {skipped}")
    print()

    # Format diagnostics
    print("Format Diagnostics:")
    print(f"  Total generated:   {total_generated}")
    print(f"  Good format:       {format_ok} ({format_ok/max(total_generated,1)*100:.1f}%)"
          f"  (<think>+<answer> with parseable probs)")
    print(f"  Parse OK:          {parse_ok} ({parse_ok/max(total_generated,1)*100:.1f}%)"
          f"  (at least 1 prob extracted)")
    print(f"  Full coverage:     {full_coverage} ({full_coverage/max(total_generated,1)*100:.1f}%)"
          f"  (probs for ALL candidates)")
    print()

    # Metrics
    if aucs:
        print("Metrics:")
        print(f"  AUC:     {_avg(aucs):.4f}")
        print(f"  MRR:     {_avg(mrrs):.4f}")
        print(f"  nDCG@5:  {_avg(ndcg5s):.4f}")
        print(f"  nDCG@10: {_avg(ndcg10s):.4f}")
    else:
        print("No metrics computed (no valid impressions)")
    print("=" * 60)

    # ── Save predictions ──
    if args.output_file and predictions:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_nids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_nids)}\n")
        print(f"\nPredictions saved to: {args.output_file}")
        print(f"  ({len(predictions)} impressions)")

    # ── Save detailed results as JSON ──
    results_json = {
        "model_path": args.model_path,
        "behaviors_path": args.behaviors_path,
        "config": {
            "cot_style": args.cot_style,
            "cot_max_tokens": args.cot_max_tokens,
            "max_history": args.max_history,
            "max_candidates": args.max_candidates,
            "use_abstract": args.use_abstract,
        },
        "num_impressions": count,
        "format_diagnostics": {
            "total_generated": total_generated,
            "format_ok": format_ok,
            "parse_ok": parse_ok,
            "full_coverage": full_coverage,
        },
        "metrics": {
            "AUC": _avg(aucs),
            "MRR": _avg(mrrs),
            "nDCG@5": _avg(ndcg5s),
            "nDCG@10": _avg(ndcg10s),
        },
    }
    results_path = args.output_file.replace(".txt", "_results.json") if args.output_file else ""
    if not results_path:
        results_path = os.path.join("results_mind", "cot_eval_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
