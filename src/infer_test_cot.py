#!/usr/bin/env python3
"""
Simple inference test for CoT news recommendation models.

Loads a model, picks random samples from MIND, runs inference,
and prints the model output alongside ground truth.

Usage:
    python src/infer_test_cot.py --model_dir output_dir/sft_mind_cot/final_checkpoint \
        --mind_root /path/to/MIND_small --num_samples 3

    # Or specify split (default: dev)
    python src/infer_test_cot.py --model_dir output_dir/rl_mind_cot_hf \
        --mind_root /path/to/MIND_small --split dev --num_samples 5
"""

import argparse
import os
import random
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Data loading (same as prepare_mind_sft_cot.py)
# ============================================================================

def load_news(news_path: str) -> Dict[str, Dict[str, str]]:
    """Load news articles from news.tsv."""
    news = {}
    with open(news_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            news_id = parts[0]
            category = parts[1] if len(parts) > 1 else ""
            title = parts[3]
            news[news_id] = {"title": title, "text": title, "category": category}
    return news


def load_behaviors(behaviors_path: str, news: dict, max_history: int = 30, max_candidates: int = 30):
    """Load behaviors and return list of (history_items, candidates, labels)."""
    samples = []
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            # Parse history
            history_ids = parts[3].split() if parts[3] else []
            history_items = []
            for nid in history_ids[-max_history:]:
                if nid in news:
                    history_items.append(news[nid])

            # Parse impressions
            impressions = parts[4].split()
            candidates = []
            labels = []
            for imp in impressions:
                if '-' not in imp:
                    continue
                nid, label = imp.rsplit('-', 1)
                if nid in news and label in ('0', '1'):
                    candidates.append(news[nid])
                    labels.append(int(label))

            if len(candidates) < 2:
                continue

            # Truncate candidates
            if len(candidates) > max_candidates:
                candidates = candidates[:max_candidates]
                labels = labels[:max_candidates]

            samples.append((history_items, candidates, labels))
    return samples


# ============================================================================
# Prompt building (identical to prepare_mind_rl_cot.py / evaluate_mind_cot.py)
# ============================================================================

def build_cot_system_prompt(num_cands: int) -> str:
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
) -> str:
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

    lines.append("[Instructions]")
    lines.append("Based on [Reading History], think step by step about what topics interest this user.")
    lines.append("Then estimate the click probability for each article in [Candidate Articles].")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simple CoT inference test")
    parser.add_argument("--model_dir", required=True, help="Path to model checkpoint")
    parser.add_argument("--mind_root", default=None, help="Path to MIND_small root")
    parser.add_argument("--split", default="dev", choices=["train", "dev"], help="Data split")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to test")
    parser.add_argument("--max_history", type=int, default=30)
    parser.add_argument("--max_candidates", type=int, default=15, help="Smaller for readability")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve mind_root
    mind_root = args.mind_root or os.environ.get("MIND_ROOT")
    if not mind_root:
        raise ValueError("Specify --mind_root or set MIND_ROOT env var")

    random.seed(args.seed)

    # Load data
    news_path = os.path.join(mind_root, args.split, "news.tsv")
    behaviors_path = os.path.join(mind_root, args.split, "behaviors.tsv")
    print(f"Loading news from {news_path}")
    news = load_news(news_path)
    print(f"  {len(news)} articles loaded")

    print(f"Loading behaviors from {behaviors_path}")
    samples = load_behaviors(behaviors_path, news, args.max_history, args.max_candidates)
    print(f"  {len(samples)} valid impressions loaded")

    # Pick random samples
    selected = random.sample(samples, min(args.num_samples, len(samples)))

    # Load model
    print(f"\nLoading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on {model.device}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run inference
    for idx, (history_items, candidates, labels) in enumerate(selected):
        print("\n" + "=" * 80)
        print(f"  SAMPLE {idx + 1} / {len(selected)}")
        print("=" * 80)

        # Ground truth
        num_clicked = sum(labels)
        print(f"\n[Ground Truth] {len(candidates)} candidates, {num_clicked} clicked")
        for i, (cand, label) in enumerate(zip(candidates, labels), 1):
            marker = "CLICK" if label == 1 else "     "
            cat = f"[{cand.get('category', '')}] " if cand.get('category') else ""
            print(f"  {i:2d}. [{marker}] {cat}{cand['text'][:60]}")

        # Build prompt
        system_content = build_cot_system_prompt(len(candidates))
        user_content = build_cot_user_prompt(history_items, candidates)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = prompt_ids["input_ids"].shape[1]
        print(f"\n[Prompt] {prompt_len} tokens")

        # Generate
        with torch.no_grad():
            output = model.generate(
                **prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output[0][prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"\n[Model Output] ({len(generated_ids)} tokens)")
        print("-" * 40)
        print(response)
        print("-" * 40)

        # Try to parse answer
        import re
        answer_match = re.search(r'<answer>\s*\[(.*?)\]\s*</answer>', response, re.DOTALL)
        if answer_match:
            print("\n[Parsed Probs]")
            prob_str = answer_match.group(1)
            for item in prob_str.split(','):
                item = item.strip()
                if ':' in item:
                    cid, prob = item.split(':', 1)
                    cid = int(cid.strip())
                    prob = float(prob.strip())
                    gt = labels[cid - 1] if cid <= len(labels) else "?"
                    marker = "CLICK" if gt == 1 else "     "
                    print(f"  Candidate {cid:2d}: prob={prob:.3f}  GT=[{marker}]")
        else:
            print("\n[WARNING] Could not parse <answer> tags from output!")


if __name__ == "__main__":
    main()
