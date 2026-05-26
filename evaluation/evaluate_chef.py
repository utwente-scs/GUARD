"""
Compare GUARD output against GLITCH Chef oracle annotations.
Computes precision, recall, and F1-score per smell type and overall.

Example usage:

    # Baseline evaluation
    python evaluate_chef.py \
        -g results/baseline_chef.csv \
        -o datasets/chef/oracle-dataset-analysis/oracle_dataset.csv

    # Extended GUARD evaluation
    python evaluate_chef.py \
        -g results/extended_chef.csv \
        -o datasets/chef/oracle-dataset-analysis/oracle_dataset.csv
"""

import csv
import argparse


def load_guard_output(path):
    """Load GUARD output CSV."""
    detections = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) >= 3 and row[1] != "none":
                filename = row[0]
                smell = row[1].strip()
                line_num = int(row[2])

                detections.append((filename, smell, line_num))

    return detections


def load_oracle(path):
    """
    Load Chef oracle CSV.

    Expected format:
        PATH,LINE,CATEGORY,AGREEMENT
    """
    oracle = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        next(reader)  # Skip header

        for row in reader:
            if len(row) >= 4:
                filename = row[0].strip()
                line_num = int(row[1])
                category = row[2].strip()

                if category != "none":
                    oracle.append((filename, category, line_num))

    return oracle


# GUARD smell names → oracle category names
GUARD_TO_ORACLE = {
    "hardcoded-secret": [
        "Hard-coded secret",
        "Admin by default", 
        "Empty password",
    ],
    "use of http": [
        "Use of HTTP without SSL/TLS",
    ],
    "improper ip address binding": [
        "Unrestricted IP Address",
    ],
    "weak crypto": [
        "Use of weak cryptography algorithms",
    ],
    "suspicious comment": [
        "Suspicious comment",
    ],
    "empty password": [
        "Empty password",
    ],
}


# Oracle categories outside GUARD's scope
NON_GUARD_CATEGORIES = {
    "Missing Default in Case Statement",
    "No integrity check",
}


def evaluate(guard_detections, oracle_entries):
    """Compute TP, FP, FN, precision, recall, and F1-score."""

    # Keep only smells GUARD is designed to detect
    oracle_compatible = [
        (f, s, l)
        for f, s, l in oracle_entries
        if s not in NON_GUARD_CATEGORIES
    ]

    oracle_incompatible = [
        (f, s, l)
        for f, s, l in oracle_entries
        if s in NON_GUARD_CATEGORIES
    ]

    # Match detections using (file, line)
    oracle_set = set()
    oracle_by_key = {}

    for f, s, l in oracle_compatible:
        key = (f, l)

        oracle_set.add(key)
        oracle_by_key[key] = s

    guard_set = set()
    guard_by_key = {}

    for f, s, l in guard_detections:
        key = (f, l)

        guard_set.add(key)
        guard_by_key[key] = s

    tp_keys = guard_set & oracle_set
    fp_keys = guard_set - oracle_set
    fn_keys = oracle_set - guard_set

    tp = len(tp_keys)
    fp = len(fp_keys)
    fn = len(fn_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_keys": tp_keys,
        "fp_keys": fp_keys,
        "fn_keys": fn_keys,
        "oracle_compatible": oracle_compatible,
        "oracle_incompatible": oracle_incompatible,
        "guard_by_key": guard_by_key,
        "oracle_by_key": oracle_by_key,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GUARD Chef detections against oracle annotations."
    )

    parser.add_argument(
        "-g",
        "--guard",
        required=True,
        help="Path to GUARD output CSV.",
    )

    parser.add_argument(
        "-o",
        "--oracle",
        required=True,
        help="Path to Chef oracle CSV.",
    )

    args = parser.parse_args()

    print("Loading GUARD output...")
    guard_detections = load_guard_output(args.guard)
    print(f"  GUARD detections: {len(guard_detections)}")

    print("Loading oracle annotations...")
    oracle_entries = load_oracle(args.oracle)
    print(f"  Oracle entries: {len(oracle_entries)}")

    results = evaluate(guard_detections, oracle_entries)

    # ---------------------------------------------------------
    # ORACLE OVERVIEW
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("ORACLE OVERVIEW")
    print(f"{'=' * 60}")

    print(f"  Total oracle smells: {len(oracle_entries)}")
    print(f"  GUARD-compatible:    {len(results['oracle_compatible'])}")
    print(f"  Non-GUARD:           {len(results['oracle_incompatible'])}")

    smell_counts = {}

    for _, s, _ in oracle_entries:
        smell_counts[s] = smell_counts.get(s, 0) + 1

    print("\n  Oracle smell distribution:")

    for s in sorted(smell_counts, key=smell_counts.get, reverse=True):
        compat = "  " if s not in NON_GUARD_CATEGORIES else " *"
        print(f"    {s}: {smell_counts[s]}{compat}")

    print("    (* = not in GUARD's scope)")

    # ---------------------------------------------------------
    # OVERALL RESULTS
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESULTS (GUARD-compatible smells only)")
    print(f"{'=' * 60}")

    print(f"  True Positives:  {results['tp']}")
    print(f"  False Positives: {results['fp']}")
    print(f"  False Negatives: {results['fn']}")

    print(f"  Precision:       {results['precision']:.4f}")
    print(f"  Recall:          {results['recall']:.4f}")
    print(f"  F1-score:        {results['f1']:.4f}")

    # ---------------------------------------------------------
    # PER-SMELL BREAKDOWN
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PER-SMELL BREAKDOWN")
    print(f"{'=' * 60}")

    guard_smell_counts = {}

    for _, s, _ in guard_detections:
        guard_smell_counts[s] = guard_smell_counts.get(s, 0) + 1

    print("\n  GUARD detections by type:")

    for s, c in sorted(guard_smell_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    # ---------------------------------------------------------
    # TRUE POSITIVES
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TRUE POSITIVES")
    print(f"{'=' * 60}")

    for key in sorted(results["tp_keys"]):
        guard_s = results["guard_by_key"].get(key, "?")
        oracle_s = results["oracle_by_key"].get(key, "?")

        print(
            f"  {key[0]} line {key[1]}: "
            f"GUARD={guard_s}, Oracle={oracle_s}"
        )

    # ---------------------------------------------------------
    # FALSE POSITIVES
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FALSE POSITIVES")
    print(f"{'=' * 60}")

    for key in sorted(results["fp_keys"]):
        guard_s = results["guard_by_key"].get(key, "?")

        print(f"  {key[0]} line {key[1]}: GUARD={guard_s}")

    # ---------------------------------------------------------
    # FALSE NEGATIVES
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FALSE NEGATIVES")
    print(f"{'=' * 60}")

    for key in sorted(results["fn_keys"]):
        oracle_s = results["oracle_by_key"].get(key, "?")

        print(f"  {key[0]} line {key[1]}: Oracle={oracle_s}")


if __name__ == "__main__":
    main()