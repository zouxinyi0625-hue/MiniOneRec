#!/usr/bin/env python3
"""
Simple Pointwise CoT inference for sanity check & speed estimation.

Runs a few samples, prints the prompt, model output, and ground truth.
Compare model's Yes/No answer with actual click labels.

Usage:
    python simple_infer_pointwise.py \\
        --model_path output_dir/rl_mind_cot_pointwise/final_checkpoint \\
        --behaviors_path $MIND_ROOT/dev/behaviors.tsv \\
        --news_path $MIND_ROOT/dev/news.tsv \\
        --cot_style category --disable_thinking --num_samples 5
"""

import argparse
import os
import re
import sys
import random
import time
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from prepare_mind_rl_cot_pointwise import build_cot_pointwise_prompt, load_news


def format_prompt(messages: list, tokenizer, enable_thinking: bool = True) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def extract_answer(text: str) -> str:
    """Extract Yes/No from generated text."""
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()
    m = re.search(r'<answer>\s*(Yes|No)\s*</answer>', text, re.IGNORECASE)
    if m:
        return m.group(1)
    if re.search(r'\byes\b', text, re.IGNORECASE):
        return "Yes"
    elif re.search(r'\bno\b', text, re.IGNORECASE):
        return "No"
    return "???"


def main():
    parser = argparse.ArgumentParser(description="Simple Pointwise CoT inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--cot_style", default="category", choices=["standard", "category", "detailed"])
    parser.add_argument("--max_history", type=int, default=30)
    parser.add_argument("--cot_max_tokens", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load news
    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, use_abstract=False)
    print(f"Loaded {len(news)} articles")

    # Load model
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    enable_thinking = not args.disable_thinking
    print(f"Device: {device}, Thinking: {'on' if enable_thinking else 'off'}")
    print()

    # Collect samples (positive + negative pairs from same impression)
    samples = []
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)

    for line in lines:
        if len(samples) >= args.num_samples:
            break

        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue

        history_ids = parts[3].split()[-args.max_history:] if parts[3] else []
        impressions = parts[4].split()

        pos_ids = []
        neg_ids = []
        for imp in impressions:
            if '-' not in imp:
                continue
            nid, label = imp.rsplit('-', 1)
            if nid not in news:
                continue
            if int(label) == 1:
                pos_ids.append(nid)
            else:
                neg_ids.append(nid)

        if not pos_ids or not neg_ids:
            continue

        history_items = [news[nid] for nid in history_ids if nid in news]

        # Pick one positive and one negative
        pos_id = random.choice(pos_ids)
        neg_id = random.choice(neg_ids)

        samples.append({
            'history': history_items,
            'candidate': news[pos_id],
            'label': 1,
            'news_id': pos_id,
        })
        samples.append({
            'history': history_items,
            'candidate': news[neg_id],
            'label': 0,
            'news_id': neg_id,
        })

    print(f"Running {len(samples)} samples ({len(samples)//2} impressions)...\n")

    times = []
    correct = 0
    total = 0

    for i, sample in enumerate(samples):
        messages = build_cot_pointwise_prompt(sample['history'], sample['candidate'], args.cot_style)
        prompt = format_prompt(messages, tokenizer, enable_thinking)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs["input_ids"].to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"].to(device),
                max_new_tokens=args.cot_max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        t1 = time.time()

        generated_ids = outputs[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer = extract_answer(generated_text)
        expected = "Yes" if sample['label'] == 1 else "No"
        is_correct = answer.lower() == expected.lower()
        if is_correct:
            correct += 1
        total += 1

        elapsed = t1 - t0
        times.append(elapsed)
        num_tokens = len(generated_ids)

        print(f"{'='*60}")
        print(f"Sample {i+1}/{len(samples)} | {sample['candidate'].get('category', '?')} | "
              f"Label: {expected} | Pred: {answer} | {'OK' if is_correct else 'WRONG'}")
        print(f"Candidate: {sample['candidate']['text'][:80]}")
        print(f"Time: {elapsed:.2f}s | Tokens: {num_tokens} | {num_tokens/elapsed:.0f} tok/s")
        print(f"--- Response ---")
        print(generated_text[:500])
        if len(generated_text) > 500:
            print(f"... ({len(generated_text)} chars total)")
        print()

    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  Accuracy: {correct}/{total} ({100*correct/max(total,1):.1f}%)")
    print(f"  Avg time: {sum(times)/len(times):.2f}s/sample")
    print(f"  Total: {sum(times):.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
