#!/usr/bin/env python3
"""
Generate SFT training data for Chain-of-Thought (CoT) news recommendation.

This script creates synthetic (prompt, response) pairs to teach a model the
<think>...</think><answer>[1:prob, 2:prob, ...]</answer> output format,
so that subsequent RL training can produce non-zero rewards.

The synthetic reasoning is template-based:
1. Identify user interest categories from reading history
2. For each candidate, assess category overlap
3. Assign plausible probabilities (clicked=high, non-clicked=low)

Usage:
    # Prepare SFT data from MIND dataset
    MIND_ROOT=/path/to/MIND_small python src/prepare_mind_sft_cot.py \
        --split train --output sft_cot_train.jsonl --max_samples 2000

    # Or with explicit paths
    python src/prepare_mind_sft_cot.py \
        --behaviors_path /path/to/behaviors.tsv \
        --news_path /path/to/news.tsv \
        --output sft_cot_data.jsonl \
        --max_samples 2000
"""

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

from tqdm import tqdm


def load_news(news_path: str, use_abstract: bool = False) -> Dict[str, Dict[str, str]]:
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
            abstract = parts[4] if len(parts) > 4 else ""
            text = f"{title} {abstract}" if use_abstract and abstract else title
            news[news_id] = {
                "title": title,
                "text": text,
                "category": category,
            }
    return news


def build_cot_system_prompt(num_cands: int) -> str:
    """Build the system prompt (must match prepare_mind_rl_cot.py exactly)."""
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
    """Build the user prompt (must match prepare_mind_rl_cot.py exactly)."""
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


def generate_synthetic_response(
    history_items: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
    labels: List[int],
    rng: random.Random,
) -> str:
    """
    Generate a synthetic <think>...</think><answer>[...]</answer> response.

    The reasoning is template-based but varied enough to teach format.
    Probabilities are noisy but directionally correct (clicked > non-clicked).
    """
    # --- Analyze user interests from history ---
    history_cats = Counter()
    for item in history_items:
        cat = item.get("category", "")
        if cat:
            history_cats[cat] += 1

    top_cats = [c for c, _ in history_cats.most_common(3)]
    if not top_cats:
        top_cats = ["General"]

    # --- Build reasoning ---
    think_lines = []

    # Step 1: Summarize interests
    interest_templates = [
        "Looking at the user's reading history, they are primarily interested in {cats}.",
        "The user's browsing history shows a preference for {cats} topics.",
        "Based on the reading history, the user frequently reads about {cats}.",
        "The reading history reveals strong interest in {cats}.",
        "Analyzing the reading history, the main interests are {cats}.",
    ]
    cats_str = ", ".join(top_cats[:3])
    think_lines.append(rng.choice(interest_templates).format(cats=cats_str))
    think_lines.append("")

    # Step 2: Evaluate candidates (brief, 1 line each)
    eval_templates_match = [
        "Candidate {i}: [{cat}] \"{title}\" — matches user interest in {match_cat}, likely to click.",
        "Candidate {i}: [{cat}] \"{title}\" — aligns with the user's {match_cat} reading pattern.",
        "Candidate {i}: [{cat}] \"{title}\" — relevant to user's interest in {match_cat}.",
    ]
    eval_templates_partial = [
        "Candidate {i}: [{cat}] \"{title}\" — somewhat related to user interests.",
        "Candidate {i}: [{cat}] \"{title}\" — partially overlaps with user's reading preferences.",
    ]
    eval_templates_no_match = [
        "Candidate {i}: [{cat}] \"{title}\" — does not match user's main interests.",
        "Candidate {i}: [{cat}] \"{title}\" — different from the user's typical reading topics.",
        "Candidate {i}: [{cat}] \"{title}\" — low relevance to user's reading history.",
    ]

    think_lines.append("Evaluating each candidate:")
    for i, (cand, label) in enumerate(zip(candidates, labels), 1):
        cat = cand.get("category", "General")
        title = cand.get("text", "")[:60]  # Truncate for brevity

        if label == 1:
            # Clicked — match
            match_cat = cat if cat in top_cats else top_cats[0]
            think_lines.append(
                rng.choice(eval_templates_match).format(
                    i=i, cat=cat, title=title, match_cat=match_cat
                )
            )
        elif cat in top_cats:
            # Not clicked but same category — partial
            think_lines.append(
                rng.choice(eval_templates_partial).format(i=i, cat=cat, title=title)
            )
        else:
            # Not clicked, different category — no match
            think_lines.append(
                rng.choice(eval_templates_no_match).format(i=i, cat=cat, title=title)
            )

    reasoning = "\n".join(think_lines)

    # --- Build probabilities ---
    probs = []
    for i, (cand, label) in enumerate(zip(candidates, labels), 1):
        cat = cand.get("category", "General")
        if label == 1:
            # Clicked: high probability
            p = rng.uniform(0.65, 0.95)
        elif cat in top_cats:
            # Same category but not clicked: medium
            p = rng.uniform(0.20, 0.45)
        else:
            # Different category and not clicked: low
            p = rng.uniform(0.05, 0.25)
        probs.append(f"{i}:{p:.2f}")

    prob_str = ", ".join(probs)

    # --- Assemble response ---
    response = f"<think>\n{reasoning}\n</think>\n<answer>\n[{prob_str}]\n</answer>"
    return response


def prepare_sft_cot_data(
    behaviors_path: str,
    news_path: str,
    output_path: str,
    max_history: int = 30,
    max_candidates: int = 30,
    min_candidates: int = 2,
    max_samples: int = 2000,
    cot_style: str = "standard",
    use_abstract: bool = False,
    seed: int = 42,
) -> None:
    """
    Convert MIND dataset to SFT JSONL with synthetic CoT responses.

    Each line in the output JSONL is:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    rng = random.Random(seed)

    print("=" * 70)
    print("MIND SFT Data Preparation (Chain-of-Thought Format)")
    print("=" * 70)
    print(f"Behaviors: {behaviors_path}")
    print(f"News: {news_path}")
    print(f"Output: {output_path}")
    print(f"Max samples: {max_samples}")
    print(f"Max history: {max_history}")
    print(f"Max candidates: {max_candidates}")
    print(f"CoT style: {cot_style}")
    print("=" * 70)
    print()

    # Load news
    print("Loading news articles...")
    news = load_news(news_path, use_abstract)
    print(f"Loaded {len(news)} news articles")

    # Count total behaviors
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # Process behaviors
    print("Processing behaviors...")
    data = []
    skipped = 0

    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, total=total_lines, desc="Processing")):
            if max_samples > 0 and len(data) >= max_samples:
                break

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            history_ids = parts[3].split()[-max_history:]
            impressions = parts[4].split()

            # Parse candidates / labels
            candidates = []
            labels = []
            for imp in impressions:
                if '-' not in imp:
                    continue
                news_id, label = imp.rsplit('-', 1)
                if news_id not in news:
                    continue
                candidates.append(news[news_id])
                labels.append(int(label))

            if sum(labels) == 0 or len(candidates) < min_candidates:
                skipped += 1
                continue

            # Limit candidates (keep all clicked + sample non-clicked)
            if len(candidates) > max_candidates:
                clicked_idx = [i for i, l in enumerate(labels) if l == 1]
                non_clicked_idx = [i for i, l in enumerate(labels) if l == 0]

                keep = clicked_idx.copy()
                remaining = max_candidates - len(clicked_idx)
                if remaining > 0 and non_clicked_idx:
                    rng_local = random.Random(line_idx + seed)
                    sampled = rng_local.sample(
                        non_clicked_idx, min(remaining, len(non_clicked_idx))
                    )
                    keep.extend(sampled)

                rng_local = random.Random(line_idx + seed + 1)
                rng_local.shuffle(keep)

                candidates = [candidates[i] for i in keep]
                labels = [labels[i] for i in keep]

            # Build history items
            history_items = [news[nid] for nid in history_ids if nid in news]

            # Build prompts (must match RL prompts exactly)
            system_content = build_cot_system_prompt(len(candidates))
            user_content = build_cot_user_prompt(history_items, candidates, cot_style)

            # Generate synthetic response
            response = generate_synthetic_response(
                history_items, candidates, labels, rng
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]

            data.append({"messages": messages})

    print()
    print(f"Generated {len(data)} SFT samples (skipped {skipped})")

    # Shuffle
    rng.shuffle(data)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")

    # Show a sample
    if data:
        print()
        print("=" * 70)
        print("Sample response (first item):")
        print("=" * 70)
        sample_resp = data[0]["messages"][-1]["content"]
        print(sample_resp[:800])
        if len(sample_resp) > 800:
            print("...")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT data for CoT format training"
    )
    parser.add_argument('--mind_root', type=str, default=None,
                        help='Root directory of MIND dataset (or MIND_ROOT env var)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'dev'])
    parser.add_argument('--behaviors_path', type=str, default=None)
    parser.add_argument('--news_path', type=str, default=None)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSONL path. Default: {mind_root}/{split}/sft_cot_{split}.jsonl')
    parser.add_argument('--max_history', type=int, default=30)
    parser.add_argument('--max_candidates', type=int, default=30)
    parser.add_argument('--min_candidates', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Number of SFT samples to generate (default: 2000)')
    parser.add_argument('--cot_style', choices=['standard', 'category', 'detailed'],
                        default='standard')
    parser.add_argument('--use_abstract', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    mind_root = args.mind_root or os.environ.get('MIND_ROOT', None)

    behaviors_path = args.behaviors_path
    news_path = args.news_path
    output_path = args.output

    if behaviors_path is None or news_path is None:
        if mind_root is None:
            parser.error(
                'Must provide --mind_root (or MIND_ROOT env var), '
                'or explicit --behaviors_path and --news_path'
            )
        split_dir = os.path.join(mind_root, args.split)
        if behaviors_path is None:
            behaviors_path = os.path.join(split_dir, 'behaviors.tsv')
        if news_path is None:
            news_path = os.path.join(split_dir, 'news.tsv')
        if output_path is None:
            output_path = os.path.join(split_dir, f'sft_cot_{args.split}.jsonl')

    if output_path is None:
        parser.error('Must provide --output when not using --mind_root')

    prepare_sft_cot_data(
        behaviors_path=behaviors_path,
        news_path=news_path,
        output_path=output_path,
        max_history=args.max_history,
        max_candidates=args.max_candidates,
        min_candidates=args.min_candidates,
        max_samples=args.max_samples,
        cot_style=args.cot_style,
        use_abstract=args.use_abstract,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
