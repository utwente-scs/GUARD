"""
Build a per-language training dataset from extracted positives/negatives.

Unlike merge_training_data.py, this does NOT mix in the Ansible base —
it produces a standalone dataset from one language only.

File structure expected:
    training_data/
        {lang}_comments_positives.csv    {lang}_comments_negatives.csv
        {lang}_secrets_positives.csv     {lang}_secrets_negatives.csv

Each file has: word,label[,source_file,line_number]
Positives have label=1, negatives have label=0.

Usage (from data_extraction/):
    python merge_per_language.py --lang puppet --dry-run
    python merge_per_language.py --lang puppet --balance
    python merge_per_language.py --lang chef --balance
"""

import argparse
import os
import pandas as pd


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

    if "text" in df.columns and "word" not in df.columns:
        df = df.rename(columns={"text": "word"})

    df = df[["word", "label"]].copy()

    mismatched = (df["label"] != expected_label).sum()
    if mismatched > 0:
        print(f"    ⚠ {path}: {mismatched}/{len(df)} rows have label != {expected_label}")

    print(f"    {os.path.basename(path):>40s}: {len(df):>6} rows")
    return df


def print_stats(name: str, combined_df: pd.DataFrame):
    pos = (combined_df["label"] == 1).sum()
    neg = (combined_df["label"] == 0).sum()

    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"    Total:             {len(combined_df):>8}")
    print(f"    Positives (1):     {pos:>8}")
    print(f"    Negatives (0):     {neg:>8}")

    ratio = pos / neg if neg > 0 else float("inf")
    marker = "✓" if 0.8 <= ratio <= 1.25 else "⚠"
    print(f"    {marker} Pos/neg ratio:   {ratio:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["puppet", "chef"],
                        help="Language to build dataset for")
    parser.add_argument("--training-dir", default="training_data",
                        help="Directory with per-language pos/neg CSVs")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (defaults to per_language/{lang})")
    parser.add_argument("--balance", action="store_true",
                        help="Undersample majority class to match minority (1:1 ratio)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only, don't write files")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("per_language", args.lang)

    print(f"\nBuilding {args.lang.upper()}-only training datasets")
    print(f"Source: {args.training_dir}/")
    print(f"Output: {args.output_dir}/")

    for category, out_filename in CATEGORIES.items():
        print(f"\n{'═'*55}")
        print(f"  {category.upper()}")
        print(f"{'═'*55}")

        pos_path = os.path.join(args.training_dir, f"{args.lang}_{category}_positives.csv")
        neg_path = os.path.join(args.training_dir, f"{args.lang}_{category}_negatives.csv")

        pos_df = load_extracted(pos_path, expected_label=1)
        neg_df = load_extracted(neg_path, expected_label=0)

        combined = pd.concat([pos_df, neg_df], ignore_index=True)

        # Deduplicate
        before = len(combined)
        combined = combined.drop_duplicates(subset=["word", "label"], keep="first")
        dupes = before - len(combined)
        if dupes > 0:
            print(f"  Removed {dupes} duplicates")

        # Balance
        if args.balance:
            min_class = combined["label"].value_counts().min()
            combined = combined.groupby("label").sample(n=min_class, random_state=42).reset_index(drop=True)
            print(f"  Balanced to {min_class} per class ({len(combined)} total)")

        print_stats(f"{args.lang.title()} {category.title()}", combined)

        if not args.dry_run:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, out_filename)
            combined.to_csv(out_path, index=False)
            print(f"  → Saved to {out_path}")

    if args.dry_run:
        print(f"\n[DRY RUN] No files written. Drop --dry-run to save.")


if __name__ == "__main__":
    main()