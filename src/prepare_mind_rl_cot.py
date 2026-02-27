#!/usr/bin/env python3
"""
Prepare MIND dataset for VERL RL training with Chain-of-Thought (CoT) prompts.

This version encourages the model to generate reasoning before providing an answer,
allowing RL to learn reasoning patterns without explicit CoT training data.

Key differences from prepare_mind_rl.py:
- Prompt explicitly asks for step-by-step reasoning
- Model is expected to output: <reasoning> ... Answer: <number>
- Reward functions in verl_reward.py extract answer from CoT output

Usage:
    # Option 1: Use MIND_ROOT env var (recommended)
    MIND_ROOT=/path/to/MIND_small python prepare_mind_rl_cot.py \\
        --split train \\
        --output_parquet /path/to/rl_cot_train.parquet

    # Option 2: Use --mind_root argument
    python prepare_mind_rl_cot.py \\
        --mind_root /path/to/MIND_small \\
        --split train \\
        --output_parquet /path/to/rl_cot_train.parquet

    # Option 3: Explicit paths (legacy)
    python prepare_mind_rl_cot.py \\
        --behaviors_path ../data/MIND/train/behaviors.tsv \\
        --news_path ../data/MIND/train/news.tsv \\
        --output_parquet ../data/MIND/train/rl_cot_train.parquet

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


def build_cot_prompt(
    history_items: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
    cot_style: str = "standard"
) -> List[Dict[str, str]]:
    """
    Build Chain-of-Thought ranking prompt.

    Args:
        history_items: List of news dicts in user's reading history
        candidates: List of candidate news dicts
        cot_style: Style of CoT prompt
            - "standard": Basic step-by-step reasoning
            - "category": Focus on category matching
            - "detailed": More detailed analysis

    Returns:
        Chat-style prompt list for VERL.
        System message defines the output format (<think> + <answer>).
        User message contains history + candidates + style-specific instructions.
    """
    # System prompt: role + task + output format (clear structure)
    num_cands = len(candidates)
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
        f"   [1:prob, 2:prob, ..., {num_cands}:prob]\n"
        "   </answer>\n"
        "\n"
        "[Output Rules]\n"
        f"- You MUST list ALL {num_cands} candidates in the answer\n"
        "- Each probability is a float between 0.0 and 1.0\n"
        "- Higher probability = more likely to be clicked\n"
        "- Probabilities do NOT need to sum to 1\n"
        "- Do NOT output anything after </answer>\n"
    )

    # User prompt: structured with [Section] markers
    lines = []

    # User history
    lines.append("[Reading History]")
    if history_items:
        for i, item in enumerate(history_items, 1):
            cat = f"[{item.get('category', 'General')}] " if item.get('category') else ""
            lines.append(f"{i}. {cat}{item['text']}")
    else:
        lines.append("(No reading history available)")
    lines.append("")

    # Candidate articles
    lines.append("[Candidate Articles]")
    for i, cand in enumerate(candidates, 1):
        cat = f"[{cand.get('category', 'General')}] " if cand.get('category') else ""
        lines.append(f"{i}. {cat}{cand['text']}")
    lines.append("")

    # CoT instruction based on style
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

    user_content = "\n".join(lines)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def prepare_mind_for_rl_cot(
    behaviors_path: str,
    news_path: str,
    output_parquet: str,
    max_history: int = 30,
    use_abstract: bool = False,
    max_samples: int = 0,
    min_candidates: int = 2,
    max_candidates: int = 50,
    cot_style: str = "standard",
    seed: int = 42
) -> None:
    """
    Convert MIND behaviors and news to VERL parquet format with CoT prompts.

    Args:
        behaviors_path: Path to behaviors.tsv
        news_path: Path to news.tsv
        output_parquet: Output parquet file path
        max_history: Maximum number of history items to include
        use_abstract: Whether to include news abstracts
        max_samples: Maximum number of samples (0 = all)
        min_candidates: Minimum number of candidates per impression
        max_candidates: Maximum number of candidates per impression
        cot_style: CoT prompt style ("standard", "category", "detailed")
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print("=" * 70)
    print("MIND RL Data Preparation (Chain-of-Thought)")
    print("=" * 70)
    print(f"Behaviors: {behaviors_path}")
    print(f"News: {news_path}")
    print(f"Output: {output_parquet}")
    print(f"Max history: {max_history}")
    print(f"Max candidates: {max_candidates}")
    print(f"Use abstracts: {use_abstract}")
    print(f"CoT style: {cot_style}")
    print("=" * 70)
    print()

    # Load news
    print("Loading news articles...")
    news = load_news(news_path, use_abstract)
    print(f"✓ Loaded {len(news)} news articles")
    print()

    # Process behaviors
    print("Processing behaviors...")
    data = []
    skipped = 0
    skipped_reasons = {
        'no_clicks': 0,
        'too_few_candidates': 0,
        'missing_news': 0
    }

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
            candidates = []
            candidate_ids = []
            labels = []
            clicked_indices = []

            for imp in impressions:
                if '-' not in imp:
                    continue
                news_id, label = imp.rsplit('-', 1)

                if news_id not in news:
                    skipped_reasons['missing_news'] += 1
                    continue

                candidates.append(news[news_id])
                candidate_ids.append(news_id)
                label_int = int(label)
                labels.append(label_int)

                if label_int == 1:
                    clicked_indices.append(len(candidates) - 1)

            # Validation
            if not clicked_indices:
                skipped += 1
                skipped_reasons['no_clicks'] += 1
                continue

            if len(candidates) < min_candidates:
                skipped += 1
                skipped_reasons['too_few_candidates'] += 1
                continue

            # Limit candidates if needed (keep all clicked + sample non-clicked)
            if len(candidates) > max_candidates:
                non_clicked_indices = [i for i in range(len(candidates)) if i not in clicked_indices]

                keep_indices = clicked_indices.copy()
                remaining_slots = max_candidates - len(clicked_indices)

                if remaining_slots > 0 and non_clicked_indices:
                    random.seed(line_idx + seed)
                    sampled = random.sample(
                        non_clicked_indices,
                        min(remaining_slots, len(non_clicked_indices))
                    )
                    keep_indices.extend(sampled)

                # Shuffle to avoid position bias
                random.shuffle(keep_indices)

                # Rebuild with selected indices
                new_candidates = []
                new_labels = []
                new_clicked_idx = None

                for new_idx, old_idx in enumerate(keep_indices):
                    new_candidates.append(candidates[old_idx])
                    new_labels.append(labels[old_idx])
                    if old_idx in clicked_indices and new_clicked_idx is None:
                        new_clicked_idx = new_idx

                candidates = new_candidates
                labels = new_labels
                clicked_idx = new_clicked_idx if new_clicked_idx is not None else 0
            else:
                # Use first clicked item as target
                clicked_idx = clicked_indices[0]

            # Build history items
            history_items = [news[nid] for nid in history_ids if nid in news]

            # Build CoT prompt
            prompt = build_cot_prompt(history_items, candidates, cot_style)

            # Ground truth is the 1-indexed position
            ground_truth = str(clicked_idx + 1)

            # Categories for partial credit reward
            categories = [c.get('category', '') for c in candidates]

            # Extra info for reward computation
            extra_info = {
                'candidates': [c['text'] for c in candidates],
                'labels': labels,
                'categories': categories,
                'clicked_idx': clicked_idx + 1,  # 1-indexed
                'num_candidates': len(candidates),
                'num_clicked': sum(labels),
                'impression_id': impression_id,
                'user_id': user_id,
                'timestamp': timestamp,
                'num_history': len(history_items),
                'cot_style': cot_style,
            }

            data.append({
                'prompt': prompt,
                'data_source': 'mind_cot',
                'reward_model': {
                    'ground_truth': ground_truth
                },
                'extra_info': extra_info
            })

    print()
    print("=" * 70)
    print("Processing Summary:")
    print("=" * 70)
    print(f"Total impressions: {total_lines}")
    print(f"Valid samples: {len(data)}")
    print(f"Skipped: {skipped}")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"  - {reason}: {count}")
    print("=" * 70)
    print()

    if not data:
        print("ERROR: No valid samples created!")
        return

    # Save to parquet
    print(f"Saving to: {output_parquet}")
    df = pd.DataFrame(data)

    # Statistics
    def _prompt_len(p):
        if isinstance(p, list) and p:
            return len(p[0].get("content", ""))
        return len(str(p))

    avg_prompt_len = df['prompt'].apply(_prompt_len).mean()
    print(f"Average prompt length: {avg_prompt_len:.0f} chars")

    df.to_parquet(output_parquet, index=False, engine='pyarrow')
    print(f"✓ Saved {len(df)} samples")
    print()

    # Sample output
    print("Sample prompt (truncated):")
    print("-" * 70)
    sample = df.iloc[0]
    sample_prompt = sample['prompt']
    if isinstance(sample_prompt, list) and sample_prompt:
        sample_prompt = sample_prompt[0].get("content", "")
    print(sample_prompt[:500] + "...")
    print("-" * 70)
    print(f"Ground truth: {sample['reward_model']['ground_truth']}")
    print(f"Num candidates: {sample['extra_info']['num_candidates']}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIND dataset for VERL RL training with CoT"
    )
    # Path options: either --mind_root + --split, or explicit --behaviors_path + --news_path
    parser.add_argument('--mind_root', type=str, default=None,
                        help='Root directory of MIND dataset. Can also be set via MIND_ROOT env var. '
                             'Paths are derived as {mind_root}/{split}/behaviors.tsv and news.tsv')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'dev', 'test'],
                        help='Dataset split (used with --mind_root)')
    parser.add_argument('--behaviors_path', type=str, default=None,
                        help='Explicit path to behaviors.tsv (overrides --mind_root)')
    parser.add_argument('--news_path', type=str, default=None,
                        help='Explicit path to news.tsv (overrides --mind_root)')
    parser.add_argument('--output_parquet', type=str, default=None,
                        help='Output parquet path. If not set, defaults to {mind_root}/{split}/rl_cot_{split}.parquet')
    parser.add_argument('--max_history', type=int, default=30)
    parser.add_argument('--max_candidates', type=int, default=50)
    parser.add_argument('--min_candidates', type=int, default=2)
    parser.add_argument('--use_abstract', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--cot_style', choices=['standard', 'category', 'detailed'],
                        default='standard', help='CoT prompt style')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Resolve MIND_ROOT: arg > env var
    mind_root = args.mind_root or os.environ.get('MIND_ROOT', None)

    # Resolve behaviors_path and news_path
    behaviors_path = args.behaviors_path
    news_path = args.news_path
    output_parquet = args.output_parquet

    if behaviors_path is None or news_path is None:
        if mind_root is None:
            parser.error(
                'Must provide either --mind_root (or MIND_ROOT env var) with --split, '
                'or explicit --behaviors_path and --news_path'
            )
        split_dir = os.path.join(mind_root, args.split)
        if behaviors_path is None:
            behaviors_path = os.path.join(split_dir, 'behaviors.tsv')
        if news_path is None:
            news_path = os.path.join(split_dir, 'news.tsv')
        if output_parquet is None:
            output_parquet = os.path.join(split_dir, f'rl_cot_{args.split}.parquet')

    if output_parquet is None:
        parser.error('Must provide --output_parquet when not using --mind_root')

    prepare_mind_for_rl_cot(
        behaviors_path=behaviors_path,
        news_path=news_path,
        output_parquet=output_parquet,
        max_history=args.max_history,
        use_abstract=args.use_abstract,
        max_samples=args.max_samples,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        cot_style=args.cot_style,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
