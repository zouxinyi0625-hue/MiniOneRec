"""
Evaluate MIND models trained with ranking-aware SFT (multiple-choice format).

This evaluation uses the SAME multiple-choice prompt format as training,
ensuring perfect alignment between training and evaluation.

Key difference from evaluate_mind.py:
- Uses multiple-choice format: "1. Title1\n2. Title2\n..." (numeric options)
- Scores option numbers (1, 2, 3, ...) instead of full text
- Matches training format exactly
- Uses KV-cache for efficient multi-token scoring
- Supports Flash Attention 2 for faster long-context processing

Usage:
    python evaluate_mind_ranking.py \
        --model_path output_dir/sft_mind_ranking_*/final_checkpoint \
        --behaviors_path ../data/MIND/dev/behaviors.tsv \
        --news_path ../data/MIND/dev/news.tsv \
        --flash_attn \
        --max_impressions 1000  # Optional: for quick testing
"""

import argparse
import json
import math
import os
import random
import re
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mind_utils import build_ranking_prompt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_news(news_path: str, use_abstract: bool) -> dict:
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

            if use_abstract and abstract:
                text = f"{title} {abstract}"
            else:
                text = title

            news[news_id] = {
                'title': title,
                'text': text,
                'category': category
            }
    return news


def build_prompt_content(history: List[dict], candidates: List[dict]) -> str:
    """
    Build prompt content (without final formatting).

    NOTE: For standard evaluation, use build_ranking_prompt from mind_utils instead.
    This function is kept for CoT mode and backward compatibility.
    """
    lines = []
    lines.append("A user read these news articles:")

    # User history - limit to last 30 for token efficiency
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get('category', 'General')
            lines.append(f"{i}. [{cat}] {h['text']}")
    else:
        lines.append("(No reading history)")

    lines.append("")

    # Candidate articles - category first in brackets
    lines.append("Candidate articles:")
    for i, cand in enumerate(candidates):
        option_num = i + 1  # 1-indexed
        cat = cand.get('category', 'General')
        lines.append(f"{option_num}. [{cat}] {cand['text']}")

    lines.append("")
    lines.append("Which article will this user read? Answer with the number.")

    return "\n".join(lines)


def format_prompt_for_eval(content: str, tokenizer, use_chat_template: bool, enable_thinking: bool = True) -> str:
    """
    Format content for evaluation (used for CoT mode).

    NOTE: For standard evaluation, use build_ranking_prompt from mind_utils instead.
    This function is kept for CoT mode which has a different prompt structure.

    Args:
        enable_thinking: If False, disable Qwen3's <think> mode for faster inference.
    """
    if use_chat_template:
        system_prompt = (
            "You are a news recommendation assistant. "
            "Based on a user's reading history, select the article they are most likely to read. "
            "Each article includes its category and title. "
            "Answer with the article number."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        # Try passing enable_thinking (supported by Qwen3 chat template)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    else:
        prompt = content + "\n\nAnswer:"

    return prompt


def build_multiple_choice_prompt(history: List[dict], candidates: List[dict]) -> str:
    """
    Build multiple-choice ranking prompt (backwards compatible).

    Returns the prompt with "Answer:" at the end.
    For chat template support, use build_prompt_content() + format_prompt_for_eval().
    """
    content = build_prompt_content(history, candidates)
    return content + "\n\nAnswer:"


def build_cot_prompt_content(history: List[dict], candidates: List[dict], cot_style: str = "standard") -> str:
    """
    Build Chain-of-Thought prompt content for evaluation.

    This prompt encourages the model to reason before answering.
    Must be aligned with prepare_mind_rl_cot.py::build_cot_prompt().

    Args:
        history: List of news dicts in user's reading history
        candidates: List of candidate news dicts
        cot_style: Style of CoT prompt ("standard", "category", "detailed")
    """
    lines = []

    lines.append("You are a news recommendation assistant.")
    lines.append("Your task is to predict which article a user will click based on their reading history.")
    lines.append("")

    # User history
    lines.append("=== User Reading History ===")
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = f"[{h.get('category', 'General')}]" if h.get('category') else ""
            lines.append(f"{i}. {cat} {h['text']}")
    else:
        lines.append("(No reading history available)")
    lines.append("")

    # Candidates
    lines.append("=== Candidate Articles ===")
    for i, cand in enumerate(candidates, 1):
        cat = f"[{cand.get('category', 'General')}]" if cand.get('category') else ""
        lines.append(f"{i}. {cat} {cand['text']}")
    lines.append("")

    # CoT instruction based on style — aligned with prepare_mind_rl_cot.py
    if cot_style == "category":
        lines.append("=== Instructions ===")
        lines.append("1. First, identify the main categories/topics in the user's reading history")
        lines.append("2. Note any patterns (e.g., sports, politics, technology)")
        lines.append("3. For each candidate, assess how well it matches the user's interests")
        lines.append("4. Select the article most likely to be clicked")
        lines.append("")
        lines.append("Think step by step, then provide your final answer as: Answer: <number>")
    elif cot_style == "detailed":
        lines.append("=== Instructions ===")
        lines.append("Analyze this recommendation task step by step:")
        lines.append("")
        lines.append("Step 1: Summarize the user's interests based on their reading history")
        lines.append("Step 2: List the key topics/categories they seem interested in")
        lines.append("Step 3: Evaluate each candidate article for relevance")
        lines.append("Step 4: Identify the best match and explain why")
        lines.append("")
        lines.append("After your analysis, provide the final answer in this exact format:")
        lines.append("Answer: <number>")
    else:  # standard
        lines.append("=== Instructions ===")
        lines.append("Think step by step about what topics interest this user based on their history.")
        lines.append("Then select the article they are most likely to click.")
        lines.append("")
        lines.append("Provide your reasoning, then give your final answer as: Answer: <number>")

    return "\n".join(lines)


def extract_cot_answer(generated_text: str, num_candidates: int) -> Optional[int]:
    """
    Extract answer number from CoT output.

    Tries multiple patterns:
    - "Answer: X"
    - "The answer is X"
    - Last number in text

    Returns:
        1-indexed answer number or None if not found
    """
    if not generated_text:
        return None

    text = generated_text.strip()

    # If model used <think> tags (Qwen instruct models), only look after </think>
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()

    # Pattern 1: "Answer: X"
    match = re.search(r'[Aa]nswer\s*:\s*(\d+)', text)
    if match:
        ans = int(match.group(1))
        if 1 <= ans <= num_candidates:
            return ans

    # Pattern 2: "The answer is X"
    match = re.search(r'[Tt]he\s+answer\s+is\s+(\d+)', text)
    if match:
        ans = int(match.group(1))
        if 1 <= ans <= num_candidates:
            return ans

    # Pattern 3: "I choose X" or similar
    match = re.search(r'[Ii]\s+(?:choose|select|pick)\s+(\d+)', text)
    if match:
        ans = int(match.group(1))
        if 1 <= ans <= num_candidates:
            return ans

    # Pattern 4: Last number in text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        ans = int(numbers[-1])
        if 1 <= ans <= num_candidates:
            return ans

    return None


def generate_cot_response(
    model,
    tokenizer,
    prompt: str,
    num_candidates: int,
    device,
    max_new_tokens: int = 256
) -> Tuple[int, str]:
    """
    Generate CoT response and extract answer.

    Args:
        model: The LLM model
        tokenizer: Tokenizer
        prompt: The CoT prompt
        num_candidates: Number of candidates (for validation)
        device: Torch device
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (predicted_index_0_based, generated_text)
        predicted_index is 0-based (for use with labels array)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for reproducibility
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    generated_ids = outputs[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract answer
    answer = extract_cot_answer(generated_text, num_candidates)

    # Convert to 0-indexed (for labels array)
    predicted_idx = (answer - 1) if answer is not None else 0

    return predicted_idx, generated_text


def cot_scores_from_prediction(predicted_idx: int, num_candidates: int) -> List[float]:
    """
    Convert CoT prediction to scores array.

    The predicted candidate gets score 1.0, all others get 0.0.
    This allows using existing metric functions.
    """
    scores = [0.0] * num_candidates
    if 0 <= predicted_idx < num_candidates:
        scores[predicted_idx] = 1.0
    return scores


def score_candidates_multiple_choice(
    model,
    tokenizer,
    prompt: str,
    num_candidates: int,
    device,
) -> List[float]:
    """
    Score candidates using multiple-choice format with numeric options.

    Scores the probability of each option number (1, 2, 3, ...).
    Handles multi-digit numbers by computing the joint probability of all tokens.
    Uses KV-cache for efficiency.

    Returns:
        List of scores (log probabilities of each number)
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Pre-tokenize all candidate numbers to get their token sequences
    candidate_tokens = []
    for i in range(1, num_candidates + 1):
        # Tokenize " {number}" (with leading space)
        number_text = f" {i}"
        token_ids = tokenizer.encode(number_text, add_special_tokens=False)
        candidate_tokens.append(token_ids)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    scores = []
    with torch.no_grad():
        # Get initial logits with KV-cache
        outputs = model(input_ids=input_ids, use_cache=True)
        first_logits = outputs.logits[0, -1, :]
        first_log_probs = torch.nn.functional.log_softmax(first_logits, dim=-1)
        base_past_kv = outputs.past_key_values

        # Group candidates by first token to batch second-token scoring
        # This reduces forward passes for multi-digit numbers
        first_token_groups = {}
        for idx, token_seq in enumerate(candidate_tokens):
            first_tok = token_seq[0]
            if first_tok not in first_token_groups:
                first_token_groups[first_tok] = []
            first_token_groups[first_tok].append((idx, token_seq))

        # Initialize scores array
        scores = [0.0] * num_candidates

        for first_tok, group in first_token_groups.items():
            first_tok_log_prob = first_log_probs[first_tok].item()

            # Check if any in group need second token
            needs_second = [(idx, seq) for idx, seq in group if len(seq) > 1]
            single_token = [(idx, seq) for idx, seq in group if len(seq) == 1]

            # Handle single-token candidates
            for idx, seq in single_token:
                scores[idx] = first_tok_log_prob

            if not needs_second:
                continue

            # For multi-token: run one forward pass with first token to get second token probs
            first_tok_tensor = torch.tensor([[first_tok]], dtype=torch.long, device=device)
            outputs2 = model(input_ids=first_tok_tensor, past_key_values=base_past_kv, use_cache=True)
            second_logits = outputs2.logits[0, -1, :]
            second_log_probs = torch.nn.functional.log_softmax(second_logits, dim=-1)
            second_past_kv = outputs2.past_key_values

            # Check if any need third token
            needs_third = [(idx, seq) for idx, seq in needs_second if len(seq) > 2]
            two_token = [(idx, seq) for idx, seq in needs_second if len(seq) == 2]

            # Handle two-token candidates
            for idx, seq in two_token:
                scores[idx] = first_tok_log_prob + second_log_probs[seq[1]].item()

            # Handle three+ token candidates (rare: numbers >= 100)
            if needs_third:
                # Group by second token
                second_token_groups = {}
                for idx, seq in needs_third:
                    second_tok = seq[1]
                    if second_tok not in second_token_groups:
                        second_token_groups[second_tok] = []
                    second_token_groups[second_tok].append((idx, seq))

                for second_tok, group3 in second_token_groups.items():
                    second_tok_log_prob = second_log_probs[second_tok].item()

                    # Run forward pass for third token
                    second_tok_tensor = torch.tensor([[second_tok]], dtype=torch.long, device=device)
                    outputs3 = model(input_ids=second_tok_tensor, past_key_values=second_past_kv, use_cache=True)
                    third_logits = outputs3.logits[0, -1, :]
                    third_log_probs = torch.nn.functional.log_softmax(third_logits, dim=-1)

                    for idx, seq in group3:
                        score = first_tok_log_prob + second_tok_log_prob
                        if len(seq) > 2:
                            score += third_log_probs[seq[2]].item()
                        # For 4+ tokens (1000+), just use first 3 tokens as approximation
                        scores[idx] = score

    return scores



def auc_score(labels: List[int], scores: List[float]) -> float:
    """Compute AUC score using sklearn's roc_auc_score."""
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return 0.5
    return roc_auc_score(labels, scores)


def mrr_score(labels: List[int], scores: List[float]) -> float:
    """Compute MRR score (Mean Reciprocal Rank)."""
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rr_scores = []
    for rank, idx in enumerate(sorted_idx, start=1):
        if labels[idx] == 1:
            rr_scores.append(1.0 / rank)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def ndcg_score(labels: List[int], scores: List[float], k: int) -> float:
    """Compute nDCG@k score."""
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    dcg = 0.0
    for rank, idx in enumerate(sorted_idx[:k], start=1):
        if labels[idx] == 1:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(sum(labels), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def limit_candidates(candidate_ids: List[str], labels: List[int], news: dict,
                     neg_ratio: float, max_candidates: int, seed: int, impression_id: str) -> Tuple[List[str], List[int]]:
    """
    Limit candidates to match training distribution.
    Uses same hard negative sampling strategy as training.
    """
    import random

    if neg_ratio <= 0 and max_candidates <= 0:
        return candidate_ids, labels

    clicked_indices = [i for i, l in enumerate(labels) if l == 1]
    non_clicked_indices = [i for i, l in enumerate(labels) if l == 0]
    rng = random.Random(f"{impression_id}-{seed}")

    # Determine number of negatives to use
    if neg_ratio > 0:
        num_negatives = int(len(clicked_indices) * neg_ratio)
        num_negatives = min(num_negatives, len(non_clicked_indices))
    elif max_candidates > 0 and len(candidate_ids) > max_candidates:
        num_negatives = max_candidates - len(clicked_indices)
        num_negatives = max(0, min(num_negatives, len(non_clicked_indices)))
    else:
        num_negatives = len(non_clicked_indices)

    # Hard negative sampling (50% same category, 50% different)
    if num_negatives > 0 and num_negatives < len(non_clicked_indices):
        clicked_categories = set()
        for idx in clicked_indices:
            if candidate_ids[idx] in news:
                cat = news[candidate_ids[idx]].get('category', '')
                clicked_categories.add(cat)

        hard_neg_indices = []
        easy_neg_indices = []

        for idx in non_clicked_indices:
            if candidate_ids[idx] in news:
                neg_cat = news[candidate_ids[idx]].get('category', '')
                if neg_cat in clicked_categories:
                    hard_neg_indices.append(idx)
                else:
                    easy_neg_indices.append(idx)
            else:
                easy_neg_indices.append(idx)

        num_hard = num_negatives // 2
        num_easy = num_negatives - num_hard

        sampled_neg_indices = []
        if hard_neg_indices:
            sampled_neg_indices.extend(rng.sample(hard_neg_indices, min(num_hard, len(hard_neg_indices))))
        if len(sampled_neg_indices) < num_negatives and easy_neg_indices:
            remaining = num_negatives - len(sampled_neg_indices)
            sampled_neg_indices.extend(rng.sample(easy_neg_indices, min(remaining, len(easy_neg_indices))))
        if len(sampled_neg_indices) < num_negatives and hard_neg_indices:
            remaining = num_negatives - len(sampled_neg_indices)
            available = [idx for idx in hard_neg_indices if idx not in sampled_neg_indices]
            if available:
                sampled_neg_indices.extend(rng.sample(available, min(remaining, len(available))))
    else:
        sampled_neg_indices = non_clicked_indices[:num_negatives]

    # Combine positives + sampled negatives
    selected_indices = clicked_indices + sampled_neg_indices

    # Apply max_candidates cap
    if max_candidates > 0 and len(selected_indices) > max_candidates:
        if len(clicked_indices) >= max_candidates:
            selected_indices = rng.sample(clicked_indices, max_candidates)
        else:
            remaining = max_candidates - len(clicked_indices)
            selected_indices = clicked_indices + sampled_neg_indices[:remaining]

    rng.shuffle(selected_indices)

    new_candidate_ids = [candidate_ids[i] for i in selected_indices]
    new_labels = [labels[i] for i in selected_indices]
    return new_candidate_ids, new_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=0, help="Max history items (0=unlimited)")
    parser.add_argument("--max_candidates", type=int, default=0, help="Max candidates per impression (0=unlimited). Set to match training.")
    parser.add_argument("--neg_ratio", type=float, default=0, help="Neg ratio to match training (e.g., 4.0 = 4 negs per pos). 0=unlimited.")
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2 for faster inference")
    parser.add_argument("--use_chat_template", action="store_true", help="Use chat template (must match training)")
    parser.add_argument("--load_training_config", action="store_true", help="Load config from training_config.json in model_path")
    parser.add_argument("--use_cot", action="store_true", help="Use Chain-of-Thought generation (slower but may be more accurate)")
    parser.add_argument("--cot_style", type=str, default="standard", choices=["standard", "category", "detailed"], help="CoT prompt style (must match training COT_STYLE)")
    parser.add_argument("--cot_max_tokens", type=int, default=256, help="Max tokens for CoT generation (default: 256)")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable Qwen3 thinking mode for faster inference")
    args = parser.parse_args()

    # Load training config if requested
    if args.load_training_config:
        import json
        config_path = os.path.join(args.model_path, "training_config.json")
        if os.path.exists(config_path):
            print(f"Loading training config from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Override args with training config (but keep command line overrides if explicitly set)
            if args.max_history == 0:
                args.max_history = config.get('max_history', 0)
            if args.max_candidates == 0:
                args.max_candidates = config.get('max_candidates', 0)
            if args.neg_ratio == 0:
                args.neg_ratio = config.get('neg_ratio', 0)
            if not args.use_abstract:
                args.use_abstract = config.get('use_abstract', False)
            if not args.use_chat_template:
                args.use_chat_template = config.get('use_chat_template', False)
            if args.seed == 42:
                args.seed = config.get('seed', 42)
            print(f"  max_history: {args.max_history}")
            print(f"  max_candidates: {args.max_candidates}")
            print(f"  neg_ratio: {args.neg_ratio}")
            print(f"  use_chat_template: {args.use_chat_template}")
            print(f"  seed: {args.seed}")
        else:
            print(f"Warning: No training config found at {config_path}")

    set_seed(args.seed)

    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"✓ Loaded {len(news)} news articles")

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load model with optional Flash Attention 2
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    device = model.device

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs = []
    mrrs = []
    ndcg5 = []
    ndcg10 = []
    predictions = []

    # Count total lines
    import subprocess
    total_lines = None
    if not args.max_impressions:
        try:
            total_lines = int(subprocess.check_output(['wc', '-l', args.behaviors_path]).split()[0])
        except:
            pass

    total_to_process = total_lines or args.max_impressions or None

    count = 0
    skipped_malformed = 0

    eval_mode = "CoT generation" if args.use_cot else "multiple-choice scoring"
    print(f"\nEvaluating with {eval_mode}...")
    print(f"Use abstract: {args.use_abstract}")
    print(f"Max candidates: {'unlimited' if args.max_candidates <= 0 else args.max_candidates}")
    print(f"Neg ratio: {'unlimited' if args.neg_ratio <= 0 else args.neg_ratio}")
    print(f"Chat template: {args.use_chat_template}")
    print(f"Max history: {args.max_history if args.max_history > 0 else 'unlimited'}")
    if args.use_cot:
        print(f"CoT max tokens: {args.cot_max_tokens}")
    print()

    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_to_process, desc="Evaluating impressions", unit="impression")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                skipped_malformed += 1
                continue

            impression_id = parts[0]
            history_ids = parts[3].split()
            # Use all history if max_history is 0 or not set
            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]
            impressions = parts[4].split()

            labels = []
            candidate_ids = []

            for imp in impressions:
                if "-" not in imp:
                    continue
                nid, label = imp.rsplit("-", 1)
                candidate_ids.append(nid)
                labels.append(int(label))

            # Skip if no candidates or no positives
            if not candidate_ids or sum(labels) == 0:
                continue

            # Limit candidates to match training distribution (if specified)
            if args.neg_ratio > 0 or args.max_candidates > 0:
                candidate_ids, labels = limit_candidates(
                    candidate_ids, labels, news,
                    args.neg_ratio, args.max_candidates, args.seed, impression_id
                )

            # Build candidate objects
            candidate_objs = []
            for nid in candidate_ids:
                if nid not in news:
                    candidate_objs.append({'text': '[MISSING_NEWS]', 'category': ''})
                else:
                    candidate_objs.append(news[nid])

            # Skip if no candidates after filtering
            if not candidate_objs:
                continue

            # Build history (pass full news objects to include category)
            history_objs = [news[nid] for nid in history_ids if nid in news]

            # Build prompt and get scores
            if args.use_cot:
                # Chain-of-Thought: generate reasoning and extract answer
                content = build_cot_prompt_content(history_objs, candidate_objs, cot_style=args.cot_style)
                enable_thinking = not getattr(args, 'disable_thinking', False)
                prompt = format_prompt_for_eval(content, tokenizer, args.use_chat_template, enable_thinking=enable_thinking)
                predicted_idx, _ = generate_cot_response(
                    model, tokenizer, prompt, len(candidate_objs), device, args.cot_max_tokens
                )
                scores = cot_scores_from_prediction(predicted_idx, len(candidate_objs))
            else:
                # Standard: score each option by probability
                # Use shared prompt builder from mind_utils for consistency with training
                prompt = build_ranking_prompt(
                    history_objs, candidate_objs,
                    tokenizer=tokenizer, use_chat_template=args.use_chat_template
                )
                scores = score_candidates_multiple_choice(
                    model, tokenizer, prompt, len(candidate_objs), device
                )

            # Compute metrics (only if we have positive labels)
            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5.append(ndcg_score(labels, scores, 5))
                ndcg10.append(ndcg_score(labels, scores, 10))

            # Generate ranked predictions
            if args.output_file:
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

            count += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{_avg(aucs):.4f}',
                'MRR': f'{_avg(mrrs):.4f}',
                'nDCG@5': f'{_avg(ndcg5):.4f}',
                'nDCG@10': f'{_avg(ndcg10):.4f}'
            })

            if args.max_impressions and count >= args.max_impressions:
                break

        pbar.close()

    eval_type = "CoT Generation" if args.use_cot else "Numeric Options"
    print(f"\nMIND Evaluation (Ranking-Aware, {eval_type})")
    print(f"Impressions processed: {count}")
    if skipped_malformed > 0:
        print(f"⚠️  Skipped malformed lines: {skipped_malformed}")

    if aucs:
        print(f"AUC:  {_avg(aucs):.4f}")
        print(f"MRR:  {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (test set has no labels)")

    # Write predictions
    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")
        print(f"✓ Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
