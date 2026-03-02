#!/usr/bin/env python3
"""
Prepare MIND dataset for VERL RL training with Pointwise Chain-of-Thought (CoT).

Each sample contains a user's reading history and ONE candidate article.
The model reasons about whether the user would click, then answers Yes/No.

Key differences from ranking-wise CoT (prepare_mind_rl_cot.py):
- One candidate per sample (not N candidates)
- Output: <think>reasoning</think><answer>Yes/No</answer>
- Binary reward (not AUC-based)
- Shorter responses (256 tokens vs 1024)
- More samples per impression (1 per candidate)

Usage:
    # Option 1: Use MIND_ROOT env var
    MIND_ROOT=/path/to/MIND_small python prepare_mind_rl_cot_pointwise.py \\
        --split train --cot_style category

    # Option 2: Explicit paths
    python prepare_mind_rl_cot_pointwise.py \\
        --behaviors_path ../data/MIND/train/behaviors.tsv \\
        --news_path ../data/MIND/train/news.tsv \\
        --output_parquet ../data/MIND/train/rl_cot_pw_train.parquet

Author: MiniOneRec
"""

import argparse
import os
import random
from typing import Dict, List
import pandas as pd
from tqdm import tqdm


def load_news(news_path: str, use_abstract: bool) -> Dict[str, Dict[str, str]]:
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

            if use_abstract and abstract:
                text = f"{title} {abstract}"
            else:
                text = title
            news[news_id] = {
                "title": title,
                "text": text,
                "category": category,
            }
    return news


def build_cot_pointwise_prompt(
    history_items: List[Dict[str, str]],
    candidate: Dict[str, str],
    cot_style: str = "standard"
) -> List[Dict[str, str]]:
    """
    Build pointwise CoT prompt in chat format.

    Args:
        history_items: List of news dicts in user's reading history
        candidate: Single candidate news dict
        cot_style: Prompt style ("standard", "category", "detailed")

    Returns:
        Chat-style prompt list: [{"role": "system", ...}, {"role": "user", ...}]
    """
    system_content = (
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

    # User prompt
    lines = []

    # Reading history
    lines.append("[Reading History]")
    if history_items:
        for i, item in enumerate(history_items, 1):
            cat = f"[{item.get('category', 'General')}] " if item.get('category') else ""
            lines.append(f"{i}. {cat}{item['text']}")
    else:
        lines.append("(No reading history available)")
    lines.append("")

    # Single candidate
    lines.append("[Candidate Article]")
    cat = f"[{candidate.get('category', 'General')}] " if candidate.get('category') else ""
    lines.append(f"{cat}{candidate['text']}")
    lines.append("")

    # Instructions based on style
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

    user_content = "\n".join(lines)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def prepare_mind_cot_pointwise(
    behaviors_path: str,
    news_path: str,
    output_parquet: str,
    max_history: int = 30,
    neg_ratio: float = 2.0,
    use_abstract: bool = False,
    max_samples: int = 0,
    cot_style: str = "standard",
    seed: int = 42
) -> None:
    """
    Convert MIND to VERL parquet format for pointwise CoT RL.

    Each impression produces: all positives + sampled negatives.
    Negative sampling uses 50% hard (same category) + 50% easy.
    """
    random.seed(seed)

    print("=" * 70)
    print("MIND RL Data Preparation (Pointwise CoT)")
    print("=" * 70)
    print(f"Behaviors: {behaviors_path}")
    print(f"News: {news_path}")
    print(f"Output: {output_parquet}")
    print(f"Max history: {max_history}")
    print(f"Neg ratio: {neg_ratio}")
    print(f"Use abstracts: {use_abstract}")
    print(f"CoT style: {cot_style}")
    print(f"Seed: {seed}")
    print("=" * 70)
    print()

    # Load news
    print("Loading news articles...")
    news = load_news(news_path, use_abstract)
    print(f"Loaded {len(news)} news articles")
    print()

    # Process behaviors
    print("Processing behaviors...")
    data = []
    total_positives = 0
    total_negatives = 0
    skipped_impressions = 0

    with open(behaviors_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, total=total_lines, desc="Processing")):
            if max_samples > 0 and len(data) >= max_samples:
                break

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            impression_id = parts[0]
            user_id = parts[1]
            timestamp = parts[2]
            history_ids = parts[3].split()[-max_history:]
            impressions = parts[4].split()

            # Parse candidates and labels
            positives = []
            negatives = []

            for imp in impressions:
                if '-' not in imp:
                    continue
                news_id, label = imp.rsplit('-', 1)
                if news_id not in news:
                    continue
                if int(label) == 1:
                    positives.append(news_id)
                else:
                    negatives.append(news_id)

            if not positives:
                skipped_impressions += 1
                continue

            # Build history
            history_items = [news[nid] for nid in history_ids if nid in news]

            # Sample negatives: 50% hard (same category) + 50% easy
            num_neg = int(len(positives) * neg_ratio)
            if num_neg > 0 and negatives:
                pos_categories = set(news[pid].get('category', '') for pid in positives)
                hard_negs = [nid for nid in negatives if news[nid].get('category', '') in pos_categories]
                easy_negs = [nid for nid in negatives if news[nid].get('category', '') not in pos_categories]

                num_hard = num_neg // 2
                num_easy = num_neg - num_hard

                sampled_negs = []
                random.seed(line_idx + seed)
                if hard_negs:
                    sampled_negs.extend(random.sample(hard_negs, min(num_hard, len(hard_negs))))
                if len(sampled_negs) < num_neg and easy_negs:
                    remaining = num_neg - len(sampled_negs)
                    sampled_negs.extend(random.sample(easy_negs, min(remaining, len(easy_negs))))
                # If still not enough, fill from remaining negatives
                if len(sampled_negs) < num_neg:
                    remaining_negs = [n for n in negatives if n not in sampled_negs]
                    remaining = num_neg - len(sampled_negs)
                    sampled_negs.extend(random.sample(remaining_negs, min(remaining, len(remaining_negs))))

                negatives = sampled_negs

            # Create positive samples
            for pos_id in positives:
                prompt = build_cot_pointwise_prompt(history_items, news[pos_id], cot_style)
                data.append({
                    'prompt': prompt,
                    'data_source': 'mind_cot_pointwise',
                    'reward_model': {
                        'ground_truth': 'Yes'
                    },
                    'extra_info': {
                        'label': 1,
                        'news_id': pos_id,
                        'impression_id': impression_id,
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'num_history': len(history_items),
                        'candidate_text': news[pos_id]['text'],
                        'candidate_category': news[pos_id].get('category', ''),
                        'cot_style': cot_style,
                    }
                })
                total_positives += 1

            # Create negative samples
            for neg_id in negatives:
                prompt = build_cot_pointwise_prompt(history_items, news[neg_id], cot_style)
                data.append({
                    'prompt': prompt,
                    'data_source': 'mind_cot_pointwise',
                    'reward_model': {
                        'ground_truth': 'No'
                    },
                    'extra_info': {
                        'label': 0,
                        'news_id': neg_id,
                        'impression_id': impression_id,
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'num_history': len(history_items),
                        'candidate_text': news[neg_id]['text'],
                        'candidate_category': news[neg_id].get('category', ''),
                        'cot_style': cot_style,
                    }
                })
                total_negatives += 1

    # Shuffle
    random.shuffle(data)

    print()
    print("=" * 70)
    print("Processing Summary:")
    print("=" * 70)
    print(f"Total impressions: {total_lines}")
    print(f"Skipped (no positives): {skipped_impressions}")
    print(f"Total samples: {len(data)}")
    print(f"  Positives: {total_positives}")
    print(f"  Negatives: {total_negatives}")
    print(f"  Pos:Neg ratio: 1:{total_negatives / max(total_positives, 1):.1f}")
    print("=" * 70)
    print()

    if not data:
        print("ERROR: No valid samples created!")
        return

    # Save to parquet
    print(f"Saving to: {output_parquet}")
    os.makedirs(os.path.dirname(output_parquet) or '.', exist_ok=True)
    df = pd.DataFrame(data)

    def _prompt_len(p):
        if isinstance(p, list) and p:
            return sum(len(m.get("content", "")) for m in p)
        return len(str(p))

    avg_prompt_len = df['prompt'].apply(_prompt_len).mean()
    print(f"Average prompt length: {avg_prompt_len:.0f} chars")

    df.to_parquet(output_parquet, index=False, engine='pyarrow')
    print(f"Saved {len(df)} samples")
    print()

    # Show sample
    print("Sample prompt:")
    print("-" * 70)
    sample = df.iloc[0]
    sample_prompt = sample['prompt']
    if isinstance(sample_prompt, list):
        for msg in sample_prompt:
            role = msg.get('role', '?')
            content = msg.get('content', '')
            print(f"[{role}]")
            print(content[:300] + ("..." if len(content) > 300 else ""))
            print()
    print("-" * 70)
    print(f"Ground truth: {sample['reward_model']['ground_truth']}")
    print(f"Label: {sample['extra_info']['label']}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIND dataset for VERL RL training with Pointwise CoT"
    )
    parser.add_argument('--mind_root', type=str, default=None,
                        help='Root directory of MIND dataset (or set MIND_ROOT env var)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'dev', 'test'])
    parser.add_argument('--behaviors_path', type=str, default=None)
    parser.add_argument('--news_path', type=str, default=None)
    parser.add_argument('--output_parquet', type=str, default=None)
    parser.add_argument('--max_history', type=int, default=30)
    parser.add_argument('--neg_ratio', type=float, default=2.0,
                        help='Negative to positive ratio (default: 2.0)')
    parser.add_argument('--use_abstract', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0, help='0 = all')
    parser.add_argument('--cot_style', choices=['standard', 'category', 'detailed'],
                        default='standard')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    mind_root = args.mind_root or os.environ.get('MIND_ROOT', None)

    behaviors_path = args.behaviors_path
    news_path = args.news_path
    output_parquet = args.output_parquet

    if behaviors_path is None or news_path is None:
        if mind_root is None:
            parser.error(
                'Must provide either --mind_root (or MIND_ROOT env var), '
                'or explicit --behaviors_path and --news_path'
            )
        split_dir = os.path.join(mind_root, args.split)
        if behaviors_path is None:
            behaviors_path = os.path.join(split_dir, 'behaviors.tsv')
        if news_path is None:
            news_path = os.path.join(split_dir, 'news.tsv')

    if output_parquet is None:
        neg_tag = str(args.neg_ratio).replace('.', 'p')
        if mind_root:
            split_dir = os.path.join(mind_root, args.split)
            output_parquet = os.path.join(split_dir, f'rl_cot_pw_neg{neg_tag}_{args.split}.parquet')
        else:
            parser.error('Must provide --output_parquet when not using --mind_root')

    prepare_mind_cot_pointwise(
        behaviors_path=behaviors_path,
        news_path=news_path,
        output_parquet=output_parquet,
        max_history=args.max_history,
        neg_ratio=args.neg_ratio,
        use_abstract=args.use_abstract,
        max_samples=args.max_samples,
        cot_style=args.cot_style,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
