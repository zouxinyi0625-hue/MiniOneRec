#!/usr/bin/env python3
"""
Simple inference script for MIND CoT model.

Runs a few samples, prints the prompt, model output, and ground truth.
Useful for quick sanity check and speed estimation before full evaluation.

Prompt is aligned with prepare_mind_rl_cot.py (training prompt).
Output format: <think>...</think><answer>[1:prob, 2:prob, ...]</answer>

Usage:
    python simple_infer.py \
        --model_path output_dir/rl_mind_cot_global_step_1420_hf \
        --behaviors_path /path/to/MIND/dev/behaviors.tsv \
        --news_path /path/to/MIND/dev/news.tsv \
        --cot_style category \
        --max_candidates 10 \
        --cot_max_tokens 1024 \
        --num_samples 5

    # With thinking disabled (faster):
    python simple_infer.py \
        --model_path output_dir/rl_mind_cot_global_step_1420_hf \
        --behaviors_path /path/to/MIND/dev/behaviors.tsv \
        --news_path /path/to/MIND/dev/news.tsv \
        --cot_style category \
        --max_candidates 10 \
        --cot_max_tokens 256 \
        --disable_thinking \
        --num_samples 5
"""

import argparse
import os
import sys
import re
import time
import random
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse training prompt builder to guarantee alignment
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from prepare_mind_rl_cot import build_cot_prompt


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


def parse_behaviors(behaviors_path: str, news: dict, max_history: int, max_candidates: int, seed: int):
    """Parse behaviors.tsv and yield (impression_id, history_objs, candidate_objs, labels)."""
    rng = random.Random(seed)
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            impression_id = parts[0]
            history_ids = parts[3].split()
            if max_history > 0:
                history_ids = history_ids[-max_history:]
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
                continue

            # Limit candidates (same logic as training: keep all positives + sample negatives)
            if max_candidates > 0 and len(candidate_ids) > max_candidates:
                pos_indices = [i for i, l in enumerate(labels) if l == 1]
                neg_indices = [i for i, l in enumerate(labels) if l == 0]
                n_neg = max(0, max_candidates - len(pos_indices))
                if n_neg < len(neg_indices):
                    neg_indices = rng.sample(neg_indices, n_neg)
                selected = sorted(pos_indices + neg_indices)
                candidate_ids = [candidate_ids[i] for i in selected]
                labels = [labels[i] for i in selected]

            history_objs = [news[nid] for nid in history_ids if nid in news]
            candidate_objs = []
            for nid in candidate_ids:
                if nid in news:
                    candidate_objs.append(news[nid])
                else:
                    candidate_objs.append({"text": "[MISSING_NEWS]", "category": ""})

            yield impression_id, history_objs, candidate_objs, labels


def extract_cot_probs(generated_text: str, num_candidates: int) -> Optional[List[float]]:
    """
    Extract click probabilities from <answer>[1:prob, 2:prob, ...]</answer> format.

    Returns list of probabilities (0-indexed) or None if parsing fails.
    """
    if not generated_text:
        return None

    text = generated_text.strip()

    # If model used <think> tags, only look after </think>
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()

    # Try <answer>...</answer> tags
    answer_match = re.search(r'<answer>\s*\[(.*?)\]\s*</answer>', text, re.DOTALL)
    if not answer_match:
        # Fallback: look for [...] directly
        answer_match = re.search(r'\[([\d:., \n]+)\]', text, re.DOTALL)

    if not answer_match:
        return None

    prob_str = answer_match.group(1)
    probs = [0.0] * num_candidates

    for item in prob_str.split(','):
        item = item.strip()
        if ':' not in item:
            continue
        try:
            cid_str, prob_str_val = item.split(':', 1)
            cid = int(cid_str.strip())
            prob = float(prob_str_val.strip())
            if 1 <= cid <= num_candidates:
                probs[cid - 1] = prob
        except (ValueError, IndexError):
            continue

    return probs


def main():
    parser = argparse.ArgumentParser(description="Simple MIND CoT inference for sanity check & speed estimation")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--behaviors_path", required=True, help="Path to behaviors.tsv")
    parser.add_argument("--news_path", required=True, help="Path to news.tsv")
    parser.add_argument("--cot_style", default="standard", choices=["standard", "category", "detailed"],
                        help="CoT prompt style (must match training)")
    parser.add_argument("--max_candidates", type=int, default=10, help="Max candidates per impression")
    parser.add_argument("--max_history", type=int, default=30, help="Max history items")
    parser.add_argument("--cot_max_tokens", type=int, default=1024, help="Max tokens for CoT generation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to run")
    parser.add_argument("--use_chat_template", action="store_true", default=True, help="Use chat template")
    parser.add_argument("--no_chat_template", action="store_true", help="Disable chat template")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable Qwen3 thinking mode for faster inference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_chat_template:
        args.use_chat_template = False

    enable_thinking = not args.disable_thinking

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print(f"\nLoading model: {args.model_path}")
    t0 = time.time()
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Thinking mode: {'ENABLED' if enable_thinking else 'DISABLED'}")

    # Load news
    print(f"Loading news: {args.news_path}")
    news_data = load_news(args.news_path)
    print(f"Loaded {len(news_data)} news articles")

    # Parse behaviors
    print(f"Parsing behaviors: {args.behaviors_path}")
    samples = []
    for item in parse_behaviors(args.behaviors_path, news_data, args.max_history, args.max_candidates, args.seed):
        samples.append(item)
        if len(samples) >= args.num_samples:
            break
    print(f"Collected {len(samples)} samples\n")

    # Run inference
    total_tokens_generated = 0
    total_gen_time = 0.0
    correct = 0

    print("=" * 80)
    for idx, (impression_id, history_objs, candidate_objs, labels) in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"Sample {idx+1}/{len(samples)} | Impression: {impression_id}")
        print(f"{'='*80}")

        # Build prompt using TRAINING prompt builder (aligned!)
        messages = build_cot_prompt(history_objs, candidate_objs, cot_style=args.cot_style)

        # Apply chat template
        if args.use_chat_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = "\n".join(f"[{m['role']}]\n{m['content']}" for m in messages) + "\n\n"

        # Print prompt (truncated)
        prompt_tokens = len(tokenizer.encode(prompt))
        print(f"\n--- PROMPT ({len(prompt)} chars, ~{prompt_tokens} tokens) ---")
        if len(prompt) > 2000:
            print(prompt[:800] + "\n... [truncated] ...\n" + prompt[-400:])
        else:
            print(prompt)

        # Ground truth
        gt_positives = [i+1 for i, l in enumerate(labels) if l == 1]
        print(f"\n--- GROUND TRUTH ---")
        print(f"Labels: {labels}")
        print(f"Positive article(s): {gt_positives}")
        for pos_idx in gt_positives:
            cand = candidate_objs[pos_idx - 1]
            cat = cand.get("category", "")
            print(f"  #{pos_idx}: [{cat}] {cand['text'][:100]}")

        # Generate
        print(f"\n--- MODEL OUTPUT ---")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.cot_max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_end = time.time()
        gen_time = t_end - t_start

        generated_ids = outputs[0, input_ids.shape[1]:]
        n_tokens = len(generated_ids)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        total_tokens_generated += n_tokens
        total_gen_time += gen_time

        print(generated_text)

        # Extract probabilities
        probs = extract_cot_probs(generated_text, len(candidate_objs))

        print(f"\n--- RESULT ---")
        if probs:
            # Find predicted answer (highest prob)
            predicted = max(range(len(probs)), key=lambda i: probs[i]) + 1
            print(f"Parsed probabilities:")
            for i, (prob, label) in enumerate(zip(probs, labels)):
                marker = "CLICK" if label == 1 else "     "
                print(f"  #{i+1}: prob={prob:.3f}  [{marker}]")
            print(f"Predicted: #{predicted} (prob={probs[predicted-1]:.3f})")
            print(f"Correct: {predicted in gt_positives}")
            if predicted in gt_positives:
                correct += 1
        else:
            print("[WARNING] Could not parse <answer> tags from output!")
            predicted = None

        print(f"Generated {n_tokens} tokens in {gen_time:.2f}s ({n_tokens/gen_time:.1f} tok/s)")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Samples: {len(samples)}")
    print(f"Correct: {correct}/{len(samples)} ({100*correct/len(samples):.1f}%)")
    print(f"Thinking mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Total generation time: {total_gen_time:.2f}s")
    if total_gen_time > 0:
        print(f"Average speed: {total_tokens_generated/total_gen_time:.1f} tok/s")
        print(f"Average per sample: {total_gen_time/len(samples):.2f}s")

    # Estimate full eval time
    try:
        with open(args.behaviors_path, "r") as f:
            total_impressions = sum(1 for _ in f)
        avg_time = total_gen_time / len(samples)
        est_total = avg_time * total_impressions
        est_total_8gpu = est_total / 8
        print(f"\n--- SPEED ESTIMATE ---")
        print(f"Total impressions in file: {total_impressions}")
        print(f"Estimated full eval time (1 GPU): {est_total/60:.1f} min ({est_total/3600:.2f} hr)")
        print(f"Estimated full eval time (8 GPU): {est_total_8gpu/60:.1f} min ({est_total_8gpu/3600:.2f} hr)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
