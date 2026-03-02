#!/usr/bin/env python3
"""
Generate SFT training data for Pointwise Chain-of-Thought (CoT) news recommendation.

Creates synthetic (prompt, response) pairs to teach a model the
<think>...</think><answer>Yes/No</answer> output format.
The prompt format is IDENTICAL to prepare_mind_rl_cot_pointwise.py so that
SFT → RL transition is seamless.

The synthetic reasoning uses template-based category matching:
1. Identify user interest categories from reading history
2. Check candidate category overlap
3. Generate a brief rationale
4. Output Yes/No

Usage:
    python src/prepare_mind_sft_cot_pointwise.py \
        --mind_root /path/to/MIND_small \
        --split train \
        --output /path/to/sft_cot_pw_train.jsonl \
        --max_samples 5000

Author: MiniOneRec
"""

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List

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


# ============================================================
# Prompt builders — MUST match prepare_mind_rl_cot_pointwise.py
# ============================================================

def build_cot_pointwise_system_prompt() -> str:
    """System prompt — identical to RL data prep."""
    return (
        "[Role]\n"
        "You are a news recommendation assistant.\n"
        "\n"
        "[Task]\n"
        "Given a user's [Reading History] and one [Candidate Article], "
        "predict whether the user will click the candidate article.\n"
        "\n"
        "[Output Format]\n"
        "You MUST respond with exactly two sections:\n"
        "\n"
        "1. Reasoning section — wrap your step-by-step analysis in <think> tags:\n"
        "   <think>\n"
        "   ... your analysis here ...\n"
        "   </think>\n"
        "\n"
        "2. Answer section — wrap your prediction in <answer> tags:\n"
        "   <answer>Yes</answer> or <answer>No</answer>\n"
        "\n"
        "[Output Rules]\n"
        "- You MUST answer exactly Yes or No inside <answer> tags\n"
        "- Do NOT output anything after </answer>\n"
    )


def build_cot_pointwise_user_prompt(
    history_items: List[Dict[str, str]],
    candidate: Dict[str, str],
    cot_style: str = "standard",
) -> str:
    """User prompt — identical to RL data prep."""
    lines = []

    lines.append("[Reading History]")
    if history_items:
        for i, item in enumerate(history_items, 1):
            cat = f"[{item.get('category', 'General')}] " if item.get('category') else ""
            lines.append(f"{i}. {cat}{item['text']}")
    else:
        lines.append("(No reading history available)")
    lines.append("")

    lines.append("[Candidate Article]")
    cat = f"[{candidate.get('category', 'General')}] " if candidate.get('category') else ""
    lines.append(f"{cat}{candidate['text']}")
    lines.append("")

    if cot_style == "category":
        lines.append("[Instructions]")
        lines.append("Analyze whether this article matches the user's interests:")
        lines.append("1. Identify the main categories/topics from [Reading History]")
        lines.append("2. Note the user's interest patterns (e.g., sports, politics, technology)")
        lines.append("3. Compare the [Candidate Article]'s category and topic with user interests")
        lines.append("4. Predict: will the user click this article?")
    elif cot_style == "detailed":
        lines.append("[Instructions]")
        lines.append("Perform a detailed analysis:")
        lines.append("Step 1: Summarize user interests from [Reading History]")
        lines.append("Step 2: Identify the topic and category of [Candidate Article]")
        lines.append("Step 3: Assess relevance to user interests")
        lines.append("Step 4: Predict: will the user click this article?")
    else:  # standard
        lines.append("[Instructions]")
        lines.append("Based on [Reading History], think step by step about what topics interest this user.")
        lines.append("Then predict whether the user will click the [Candidate Article].")

    return "\n".join(lines)


# ============================================================
# Synthetic CoT response generation
# ============================================================

def generate_synthetic_response(
    history_items: List[Dict[str, str]],
    candidate: Dict[str, str],
    label: int,
    rng: random.Random,
) -> str:
    """
    Generate a synthetic <think>...</think><answer>Yes/No</answer> response.

    Template-based reasoning that analyses category overlap.
    Quality is deliberately mediocre — RL will learn better reasoning.
    """
    # Analyse user interests from history categories
    cat_counts = Counter()
    for item in history_items:
        cat = item.get('category', '').strip()
        if cat:
            cat_counts[cat] += 1

    top_cats = [c for c, _ in cat_counts.most_common(5)]
    cand_cat = candidate.get('category', '').strip()

    # Generate reasoning
    reasoning_lines = []

    # Step 1: summarise user interests
    if top_cats:
        cat_str = ", ".join(top_cats[:3])
        reasoning_lines.append(
            f"The user's reading history shows interest in: {cat_str}."
        )
        if len(top_cats) > 3:
            reasoning_lines.append(
                f"Other topics include: {', '.join(top_cats[3:])}."
            )
    else:
        reasoning_lines.append("The user's reading history is limited.")

    # Step 2: candidate analysis
    if cand_cat:
        reasoning_lines.append(
            f"The candidate article belongs to the [{cand_cat}] category."
        )
    else:
        reasoning_lines.append("The candidate article has no clear category.")

    # Step 3: match assessment
    if cand_cat and cand_cat in cat_counts:
        count = cat_counts[cand_cat]
        total = sum(cat_counts.values())
        pct = count / total * 100
        reasoning_lines.append(
            f"The user has read {count} article(s) in [{cand_cat}] "
            f"({pct:.0f}% of history), indicating interest in this topic."
        )
        match_strength = "strong" if pct > 20 else "moderate"
    elif cand_cat and cand_cat not in cat_counts:
        reasoning_lines.append(
            f"The [{cand_cat}] category does not appear in the user's history. "
            f"This suggests lower interest."
        )
        match_strength = "weak"
    else:
        reasoning_lines.append("Cannot determine category match.")
        match_strength = "unknown"

    # Step 4: prediction justification — add slight randomness
    answer = "Yes" if label == 1 else "No"
    if label == 1:
        templates = [
            f"Given the {match_strength} category alignment, the user is likely to click this article.",
            f"The topic overlap suggests the user would be interested in this article.",
            f"Based on the user's reading patterns, this article is relevant to their interests.",
        ]
    else:
        templates = [
            f"Given the {match_strength} category alignment, the user is unlikely to click this article.",
            f"The topic does not align well with the user's established interests.",
            f"Based on the reading history, this article falls outside the user's main interests.",
        ]
    reasoning_lines.append(rng.choice(templates))

    reasoning = "\n".join(reasoning_lines)

    return f"<think>\n{reasoning}\n</think>\n<answer>{answer}</answer>"


# ============================================================
# Main data preparation
# ============================================================

def prepare_sft_data(
    behaviors_path: str,
    news_path: str,
    output_file: str,
    max_history: int = 30,
    neg_ratio: float = 2.0,
    use_abstract: bool = False,
    max_samples: int = 0,
    cot_style: str = "standard",
    seed: int = 42,
) -> None:
    """Convert MIND to JSONL for pointwise CoT SFT."""
    rng = random.Random(seed)

    print("=" * 70)
    print("MIND SFT Data Preparation (Pointwise CoT)")
    print("=" * 70)
    print(f"Behaviors: {behaviors_path}")
    print(f"News: {news_path}")
    print(f"Output: {output_file}")
    print(f"Max history: {max_history}")
    print(f"Neg ratio: {neg_ratio}")
    print(f"CoT style: {cot_style}")
    print("=" * 70)
    print()

    # Load news
    news = load_news(news_path, use_abstract)
    print(f"Loaded {len(news)} news articles")

    # Count lines
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    system_content = build_cot_pointwise_system_prompt()
    data = []
    total_pos = 0
    total_neg = 0

    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, total=total_lines, desc="Processing")):
            if max_samples > 0 and len(data) >= max_samples:
                break

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            history_ids = parts[3].split()[-max_history:]
            impressions = parts[4].split()

            positives = []
            negatives = []
            for imp in impressions:
                if '-' not in imp:
                    continue
                nid, lbl = imp.rsplit('-', 1)
                if nid not in news:
                    continue
                if int(lbl) == 1:
                    positives.append(nid)
                else:
                    negatives.append(nid)

            if not positives:
                continue

            history_items = [news[nid] for nid in history_ids if nid in news]

            # Sample negatives: 50% hard + 50% easy
            num_neg = int(len(positives) * neg_ratio)
            if num_neg > 0 and negatives:
                pos_categories = set(news[pid].get('category', '') for pid in positives)
                hard_negs = [n for n in negatives if news[n].get('category', '') in pos_categories]
                easy_negs = [n for n in negatives if news[n].get('category', '') not in pos_categories]

                num_hard = num_neg // 2
                sampled = []
                rng.seed(line_idx + seed)
                if hard_negs:
                    sampled.extend(rng.sample(hard_negs, min(num_hard, len(hard_negs))))
                remaining = num_neg - len(sampled)
                if remaining > 0 and easy_negs:
                    sampled.extend(rng.sample(easy_negs, min(remaining, len(easy_negs))))
                remaining = num_neg - len(sampled)
                if remaining > 0:
                    rest = [n for n in negatives if n not in sampled]
                    sampled.extend(rng.sample(rest, min(remaining, len(rest))))
                negatives = sampled

            # Build samples
            for pos_id in positives:
                user_content = build_cot_pointwise_user_prompt(history_items, news[pos_id], cot_style)
                response = generate_synthetic_response(history_items, news[pos_id], 1, rng)
                data.append({
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": response},
                    ]
                })
                total_pos += 1

            for neg_id in negatives:
                user_content = build_cot_pointwise_user_prompt(history_items, news[neg_id], cot_style)
                response = generate_synthetic_response(history_items, news[neg_id], 0, rng)
                data.append({
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": response},
                    ]
                })
                total_neg += 1

    rng.shuffle(data)

    print()
    print(f"Total samples: {len(data)}")
    print(f"  Positives: {total_pos}")
    print(f"  Negatives: {total_neg}")
    print(f"  Ratio: 1:{total_neg / max(total_pos, 1):.1f}")

    # Save JSONL
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved to: {output_file}")

    # Show sample
    if data:
        print()
        print("Sample:")
        print("-" * 70)
        sample = data[0]
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            print(f"[{role}]")
            print(content[:300] + ("..." if len(content) > 300 else ""))
            print()
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT data for Pointwise CoT format"
    )
    parser.add_argument('--mind_root', type=str, default=None)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'dev'])
    parser.add_argument('--behaviors_path', type=str, default=None)
    parser.add_argument('--news_path', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_history', type=int, default=30)
    parser.add_argument('--neg_ratio', type=float, default=2.0)
    parser.add_argument('--use_abstract', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0, help='0 = all')
    parser.add_argument('--cot_style', choices=['standard', 'category', 'detailed'],
                        default='standard')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    mind_root = args.mind_root or os.environ.get('MIND_ROOT', None)
    behaviors_path = args.behaviors_path
    news_path = args.news_path
    output_file = args.output

    if behaviors_path is None or news_path is None:
        if mind_root is None:
            parser.error('Provide --mind_root or MIND_ROOT env var, or explicit paths')
        split_dir = os.path.join(mind_root, args.split)
        if behaviors_path is None:
            behaviors_path = os.path.join(split_dir, 'behaviors.tsv')
        if news_path is None:
            news_path = os.path.join(split_dir, 'news.tsv')

    if output_file is None:
        if mind_root:
            split_dir = os.path.join(mind_root, args.split)
            output_file = os.path.join(split_dir, f'sft_cot_pw_{args.split}.jsonl')
        else:
            parser.error('Provide --output or --mind_root')

    prepare_sft_data(
        behaviors_path=behaviors_path,
        news_path=news_path,
        output_file=output_file,
        max_history=args.max_history,
        neg_ratio=args.neg_ratio,
        use_abstract=args.use_abstract,
        max_samples=args.max_samples,
        cot_style=args.cot_style,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
