#!/usr/bin/env python3
"""
Read and print parquet samples for inspection.

Usage:
    python src/read_parquet.py /path/to/data.parquet          # print 2 samples
    python src/read_parquet.py /path/to/data.parquet -n 5     # print 5 samples
    python src/read_parquet.py /path/to/data.parquet --cols prompt reward_model  # specific columns
"""

import argparse
import json
import pandas as pd


def pretty_print(obj, indent=2):
    """Pretty print nested structures."""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Read and print parquet samples")
    parser.add_argument("parquet_path", help="Path to parquet file")
    parser.add_argument("-n", "--num_samples", type=int, default=2, help="Number of samples to print (default: 2)")
    parser.add_argument("--cols", nargs="*", default=None, help="Specific columns to print (default: all)")
    parser.add_argument("--summary", action="store_true", help="Also print dataset summary stats")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path, engine="pyarrow")

    print("=" * 80)
    print(f"Parquet: {args.parquet_path}")
    print(f"Rows: {len(df)}  |  Columns: {list(df.columns)}")
    print("=" * 80)

    if args.summary:
        print("\n--- Column Types ---")
        for col in df.columns:
            sample_val = df[col].iloc[0] if len(df) > 0 else None
            val_type = type(sample_val).__name__
            print(f"  {col}: {val_type}")
        print()

    cols = args.cols if args.cols else df.columns.tolist()
    n = min(args.num_samples, len(df))

    for i in range(n):
        print(f"\n{'─' * 80}")
        print(f"  Sample {i + 1} / {n}")
        print(f"{'─' * 80}")
        row = df.iloc[i]
        for col in cols:
            if col not in df.columns:
                print(f"\n  [!] Column '{col}' not found")
                continue
            val = row[col]
            print(f"\n  [{col}]")
            print(f"  {pretty_print(val)}")
        print()

    print("=" * 80)
    print(f"Printed {n} / {len(df)} samples")
    print("=" * 80)


if __name__ == "__main__":
    main()
