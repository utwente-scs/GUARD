"""
Merge extracted Chef + Puppet training data into existing GUARD training CSVs.

File structure expected:
    training_data/
        chef_comments_positives.csv      chef_comments_negatives.csv
        chef_secrets_positives.csv       chef_secrets_negatives.csv
        puppet_comments_positives.csv    puppet_comments_negatives.csv
        puppet_secrets_positives.csv     puppet_secrets_negatives.csv

Each file has: word,label[,source_file,line_number]
Positives have label=1, negatives have label=0.

Usage (from data_extraction/):
    python merge_training_data.py --dry-run          # preview stats
    python merge_training_data.py --balance          # write balanced 1:1 CSVs
    python merge_training_data.py                    # write combined CSVs (unbalanced)
"""

import argparse
import os
import pandas as pd


LANGUAGES = ["puppet", "chef"]
CATEGORIES = {
    "comments": "susp_comments_dataset.csv",
    "secrets": "hardcoded_secrets_dataset.csv",
}


def load_extracted(path: str, expected_label: int) -> pd.DataFrame:
    """Load an extracted CSV, keep only word + label."""
    if not os.path.exists(path):
        print(f"    SKIP: {path} not found")
        return pd.DataFrame(columns=["word", "label"])

    df = pd.read_csv(path)

    # Normalize column name
    if "text" in df.columns and "word" not in df.columns:
        df = df.rename(columns={"text": "word"})

    df = df[["word", "label"]].copy()

    # Sanity check: verify labels match the filename
    mismatched = (df["label"] != expected_label).sum()
    if mismatched > 0:
        print(f"    ⚠ {path}: {mismatched}/{len(df)} rows have label != {expected_label}")

    print(f"    {os.path.basename(path):>40s}: {len(df):>6} rows")
    return df


def print_stats(name: str, base_n: int, combined_df: pd.DataFrame):
    pos = (combined_df["label"] == 1).sum()
    neg = (combined_df["label"] == 0).sum()
    added = len(combined_df) - base_n

    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"    Base (Ansible):    {base_n:>8}")
    print(f"    Added:             {added:>8}")
    print(f"    Combined total:    {len(combined_df):>8}")
    print(f"    Positives (1):     {pos:>8}")
    print(f"    Negatives (0):     {neg:>8}")

    ratio = pos / neg if neg > 0 else float("inf")
    marker = "✓" if 0.8 <= ratio <= 1.25 else "⚠"
    print(f"    {marker} Pos/neg ratio:   {ratio:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default="training_data",
                        help="Directory with per-language pos/neg CSVs")
    parser.add_argument("--base-dir", default="combined",
                        help="Directory with existing base datasets")
    parser.add_argument("--output-dir", default="combined",
                        help="Output directory (defaults to overwriting base)")
    parser.add_argument("--balance", action="store_true",
                        help="Undersample majority class to match minority (1:1 ratio)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only, don't write files")
    args = parser.parse_args()

    for category, out_filename in CATEGORIES.items():
        print(f"\n{'═'*55}")
        print(f"  {category.upper()}")
        print(f"{'═'*55}")

        # Load base dataset
        base_path = os.path.join(args.base_dir, out_filename)
        if os.path.exists(base_path):
            base_df = pd.read_csv(base_path, usecols=["word", "label"])
            print(f"  Base: {len(base_df)} rows from {base_path}")
        else:
            print(f"  Base: {base_path} not found — starting from empty")
            base_df = pd.DataFrame(columns=["word", "label"])

        base_n = len(base_df)
        parts = [base_df]

        # Load extracted per-language files
        for lang in LANGUAGES:
            pos_path = os.path.join(args.training_dir, f"{lang}_{category}_positives.csv")
            neg_path = os.path.join(args.training_dir, f"{lang}_{category}_negatives.csv")
            parts.append(load_extracted(pos_path, expected_label=1))
            parts.append(load_extracted(neg_path, expected_label=0))

        # Combine + deduplicate
        combined = pd.concat(parts, ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(subset=["word", "label"], keep="first")
        dupes = before - len(combined)
        if dupes > 0:
            print(f"  Removed {dupes} duplicates")

        # Balance classes if requested
        if args.balance:
            min_class = combined["label"].value_counts().min()
            combined = combined.groupby("label").sample(n=min_class, random_state=42).reset_index(drop=True)
            print(f"  Balanced to {min_class} per class ({len(combined)} total)")

        print_stats(category.title(), base_n, combined)

        # Save
        if not args.dry_run:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, out_filename)
            combined.to_csv(out_path, index=False)
            print(f"  → Saved to {out_path}")

    if args.dry_run:
        print(f"\n[DRY RUN] No files written. Drop --dry-run to save.")


if __name__ == "__main__":
    main()