"""
RQ3: Evaluate GLITCH detection results against oracle datasets.

Uses the same per-line matching methodology as GUARD's evaluators
(evaluate_chef.py, evaluate_puppet.py) for fair comparison.

Usage:
    python evaluate_glitch.py \
        --glitch datasets/chef-data/oracle-dataset-analysis/GLITCH-chef-oracle.csv \
        --oracle datasets/chef-data/oracle-dataset-analysis/oracle_dataset.csv \
        --guard results/retrained_chef_v2.csv \
        --lang chef

    python evaluate_glitch.py \
        --glitch datasets/puppet/oracle-dataset-analysis/GLITCH-puppet-oracle.csv \
        --oracle datasets/puppet/oracle-dataset-analysis/oracle-dataset.csv \
        --guard results/retrained_puppet_v2.csv \
        --lang puppet
"""

import csv
import argparse


# Oracle categories outside GUARD's 7-smell scope.
# These are excluded so GLITCH is evaluated on the same smell set as GUARD.
NON_GUARD_CATEGORIES = {
    "Missing Default in Case Statement",
    "No integrity check",
}

# GLITCH error types that correspond to non-GUARD oracle categories.
# Filter these from GLITCH detections so it isn't penalised for detecting
# smells that GUARD structurally cannot detect.
NON_GUARD_GLITCH_ERRORS = {
    "no-integrity-check",
    "missing default case statement",
}

# GLITCH reports these as separate sub-types, but they are not independent
# oracle categories — they co-occur with hardcoded-secret on the same lines.
# We keep them so they contribute to per-line matching, but note them.
GLITCH_SUBTYPES = {"hardcoded-password", "hardcoded-username"}


def load_guard_output(path):
    """Load GUARD output CSV (filename, smell, line_num — no header)."""
    detections = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3 and row[1].strip() != "none":
                filename = row[0].strip()
                smell = row[1].strip()
                line_num = int(row[2])
                detections.append((filename, smell, line_num))
    return detections


def load_glitch_output(path):
    """Load GLITCH detection CSV (PATH, LINE, ERROR)."""
    detections = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 3:
                filename = row[0].strip()
                line_num = int(row[1])
                error = row[2].strip()
                detections.append((filename, error, line_num))
    return detections


def load_oracle(path):
    """Load oracle CSV (PATH, LINE, CATEGORY, AGREEMENT)."""
    entries = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 4:
                filename = row[0].strip()
                line_num = int(row[1])
                category = row[2].strip()
                if category != "none":
                    entries.append((filename, category, line_num))
    return entries


def evaluate_per_line(detections, oracle_entries):
    """
    Per-line evaluation: match on (file, line) pairs.
    Same methodology as GUARD's evaluate_chef.py / evaluate_puppet.py.
    """
    # Filter oracle to GUARD-compatible smells
    oracle_compatible = [
        (f, s, l) for f, s, l in oracle_entries
        if s not in NON_GUARD_CATEGORIES
    ]
    oracle_incompatible = [
        (f, s, l) for f, s, l in oracle_entries
        if s in NON_GUARD_CATEGORIES
    ]

    # Filter detections to GUARD-compatible error types
    detections_filtered = [
        (f, e, l) for f, e, l in detections
        if e not in NON_GUARD_GLITCH_ERRORS
    ]

    # Deduplicate to (file, line) pairs
    oracle_set = set()
    oracle_by_key = {}
    for f, s, l in oracle_compatible:
        key = (f, l)
        oracle_set.add(key)
        oracle_by_key[key] = s

    detection_set = set()
    detection_by_key = {}
    for f, e, l in detections_filtered:
        key = (f, l)
        detection_set.add(key)
        if key not in detection_by_key:
            detection_by_key[key] = []
        if e not in detection_by_key[key]:
            detection_by_key[key].append(e)

    tp_keys = detection_set & oracle_set
    fp_keys = detection_set - oracle_set
    fn_keys = oracle_set - detection_set

    tp = len(tp_keys)
    fp = len(fp_keys)
    fn = len(fn_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "tp_keys": tp_keys, "fp_keys": fp_keys, "fn_keys": fn_keys,
        "oracle_compatible": oracle_compatible,
        "oracle_incompatible": oracle_incompatible,
        "detection_by_key": detection_by_key,
        "oracle_by_key": oracle_by_key,
        "detections_filtered": detections_filtered,
        "detections_excluded": [
            (f, e, l) for f, e, l in detections
            if e in NON_GUARD_GLITCH_ERRORS
        ],
    }


def evaluate_full_scope(detections, oracle_entries):
    """
    Evaluate GLITCH on ALL oracle smells (its full capability).
    Shows what GLITCH can detect beyond GUARD's scope.
    """
    oracle_set = set()
    oracle_by_key = {}
    for f, s, l in oracle_entries:
        key = (f, l)
        oracle_set.add(key)
        oracle_by_key[key] = s

    detection_set = set()
    for f, e, l in detections:
        detection_set.add((f, l))

    tp_keys = detection_set & oracle_set
    fp_keys = detection_set - oracle_set
    fn_keys = oracle_set - detection_set

    tp = len(tp_keys)
    fp = len(fp_keys)
    fn = len(fn_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GLITCH detections against oracle (RQ3)."
    )
    parser.add_argument("-g", "--glitch", required=True,
                        help="Path to GLITCH output CSV.")
    parser.add_argument("-o", "--oracle", required=True,
                        help="Path to oracle CSV.")
    parser.add_argument("-l", "--lang", required=True,
                        choices=["chef", "puppet"],
                        help="Language being evaluated.")
    parser.add_argument("--guard", default=None,
                        help="Path to GUARD output CSV (for side-by-side comparison).")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"RQ3: GLITCH EVALUATION — {args.lang.upper()}")
    print(f"{'=' * 60}")

    # Load data
    detections = load_glitch_output(args.glitch)
    oracle_entries = load_oracle(args.oracle)

    print(f"\n  Raw GLITCH detections: {len(detections)}")
    print(f"  Oracle entries (excl. 'none'): {len(oracle_entries)}")

    # ---------------------------------------------------------
    # ORACLE OVERVIEW
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("ORACLE SMELL DISTRIBUTION")
    print(f"{'=' * 60}")

    smell_counts = {}
    for _, s, _ in oracle_entries:
        smell_counts[s] = smell_counts.get(s, 0) + 1

    for s in sorted(smell_counts, key=smell_counts.get, reverse=True):
        scope = "  " if s not in NON_GUARD_CATEGORIES else " *"
        print(f"    {s}: {smell_counts[s]}{scope}")
    print("    (* = outside GUARD's scope)")

    guard_compat = sum(c for s, c in smell_counts.items()
                       if s not in NON_GUARD_CATEGORIES)
    non_guard = sum(c for s, c in smell_counts.items()
                    if s in NON_GUARD_CATEGORIES)
    print(f"\n    GUARD-compatible: {guard_compat}")
    print(f"    Non-GUARD:        {non_guard}")

    # ---------------------------------------------------------
    # GLITCH DETECTION DISTRIBUTION
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("GLITCH DETECTION DISTRIBUTION")
    print(f"{'=' * 60}")

    error_counts = {}
    for _, e, _ in detections:
        error_counts[e] = error_counts.get(e, 0) + 1

    for e in sorted(error_counts, key=error_counts.get, reverse=True):
        scope = "  " if e not in NON_GUARD_GLITCH_ERRORS else " *"
        sub = " (sub)" if e in GLITCH_SUBTYPES else ""
        print(f"    {e}: {error_counts[e]}{scope}{sub}")
    print("    (* = excluded from GUARD-scope evaluation)")
    print("    (sub) = sub-type, co-occurs with hardcoded-secret")

    # ---------------------------------------------------------
    # GUARD-SCOPE EVALUATION (apples-to-apples with GUARD)
    # ---------------------------------------------------------
    results = evaluate_per_line(detections, oracle_entries)

    print(f"\n{'=' * 60}")
    print("RESULTS — GUARD-COMPATIBLE SMELLS ONLY (per-line matching)")
    print(f"{'=' * 60}")

    print(f"  Detections after filtering: "
          f"{len(set((f,l) for f,_,l in results['detections_filtered']))}"
          f" unique lines "
          f"({len(results['detections_excluded'])} excluded as non-GUARD)")
    print(f"\n  True Positives:  {results['tp']}")
    print(f"  False Positives: {results['fp']}")
    print(f"  False Negatives: {results['fn']}")
    print(f"  Precision:       {results['precision']:.4f}")
    print(f"  Recall:          {results['recall']:.4f}")
    print(f"  F1-score:        {results['f1']:.4f}")

    # ---------------------------------------------------------
    # FULL-SCOPE EVALUATION (GLITCH's full capability)
    # ---------------------------------------------------------
    full = evaluate_full_scope(detections, oracle_entries)

    print(f"\n{'=' * 60}")
    print("RESULTS — ALL ORACLE SMELLS (GLITCH full scope)")
    print(f"{'=' * 60}")

    print(f"  True Positives:  {full['tp']}")
    print(f"  False Positives: {full['fp']}")
    print(f"  False Negatives: {full['fn']}")
    print(f"  Precision:       {full['precision']:.4f}")
    print(f"  Recall:          {full['recall']:.4f}")
    print(f"  F1-score:        {full['f1']:.4f}")

    # ---------------------------------------------------------
    # TRUE POSITIVES (detailed)
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TRUE POSITIVES")
    print(f"{'=' * 60}")

    for key in sorted(results["tp_keys"]):
        glitch_e = ", ".join(results["detection_by_key"].get(key, ["?"]))
        oracle_s = results["oracle_by_key"].get(key, "?")
        print(f"  {key[0]} line {key[1]}: GLITCH={glitch_e}, Oracle={oracle_s}")

    # ---------------------------------------------------------
    # FALSE NEGATIVES
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FALSE NEGATIVES (oracle smells GLITCH missed)")
    print(f"{'=' * 60}")

    for key in sorted(results["fn_keys"]):
        oracle_s = results["oracle_by_key"].get(key, "?")
        print(f"  {key[0]} line {key[1]}: Oracle={oracle_s}")

    # ---------------------------------------------------------
    # FALSE POSITIVES (first 30)
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"FALSE POSITIVES (GLITCH flagged, not in oracle)"
          f" — showing first 30 of {results['fp']}")
    print(f"{'=' * 60}")

    for i, key in enumerate(sorted(results["fp_keys"])):
        if i >= 30:
            print(f"  ... and {results['fp'] - 30} more")
            break
        glitch_e = ", ".join(results["detection_by_key"].get(key, ["?"]))
        print(f"  {key[0]} line {key[1]}: GLITCH={glitch_e}")

    # ---------------------------------------------------------
    # GUARD EVALUATION (if --guard provided)
    # ---------------------------------------------------------
    guard_results = None
    if args.guard:
        print(f"\n{'=' * 60}")
        print("GUARD EVALUATION (same oracle, same methodology)")
        print(f"{'=' * 60}")

        guard_detections = load_guard_output(args.guard)
        print(f"  Raw GUARD detections: {len(guard_detections)}")

        guard_results = evaluate_per_line(guard_detections, oracle_entries)

        print(f"\n  True Positives:  {guard_results['tp']}")
        print(f"  False Positives: {guard_results['fp']}")
        print(f"  False Negatives: {guard_results['fn']}")
        print(f"  Precision:       {guard_results['precision']:.4f}")
        print(f"  Recall:          {guard_results['recall']:.4f}")
        print(f"  F1-score:        {guard_results['f1']:.4f}")

    # ---------------------------------------------------------
    # COMPARISON TABLE
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"RQ3 COMPARISON TABLE — {args.lang.upper()}")
    print(f"{'=' * 60}")

    if guard_results:
        g = guard_results
        print(f"\n  | Metric    | GLITCH (GUARD-scope) | GLITCH (full) | GUARD  |")
        print(f"  |-----------|---------------------:|--------------:|-------:|")
        print(f"  | TP        | {results['tp']:>20} | {full['tp']:>13} | {g['tp']:>6} |")
        print(f"  | FP        | {results['fp']:>20} | {full['fp']:>13} | {g['fp']:>6} |")
        print(f"  | FN        | {results['fn']:>20} | {full['fn']:>13} | {g['fn']:>6} |")
        print(f"  | Precision | {results['precision']:>20.4f} | {full['precision']:>13.4f} | {g['precision']:.4f} |")
        print(f"  | Recall    | {results['recall']:>20.4f} | {full['recall']:>13.4f} | {g['recall']:.4f} |")
        print(f"  | F1        | {results['f1']:>20.4f} | {full['f1']:>13.4f} | {g['f1']:.4f} |")

        # Overlap analysis
        glitch_tp = results["tp_keys"]
        guard_tp = guard_results["tp_keys"]
        both_tp = glitch_tp & guard_tp
        only_glitch = glitch_tp - guard_tp
        only_guard = guard_tp - glitch_tp

        print(f"\n  Detection overlap:")
        print(f"    Both found:       {len(both_tp)}")
        print(f"    Only GLITCH:      {len(only_glitch)}")
        print(f"    Only GUARD:       {len(only_guard)}")

        if only_glitch:
            print(f"\n  Smells only GLITCH caught:")
            for key in sorted(only_glitch):
                oracle_s = results["oracle_by_key"].get(key, "?")
                print(f"    {key[0]} line {key[1]}: {oracle_s}")

        if only_guard:
            print(f"\n  Smells only GUARD caught:")
            for key in sorted(only_guard):
                oracle_s = guard_results["oracle_by_key"].get(key, "?")
                print(f"    {key[0]} line {key[1]}: {oracle_s}")
    else:
        print(f"\n  | Metric    | GLITCH (GUARD-scope) | GLITCH (full) |")
        print(f"  |-----------|---------------------:|--------------:|")
        print(f"  | TP        | {results['tp']:>20} | {full['tp']:>13} |")
        print(f"  | FP        | {results['fp']:>20} | {full['fp']:>13} |")
        print(f"  | FN        | {results['fn']:>20} | {full['fn']:>13} |")
        print(f"  | Precision | {results['precision']:>20.4f} | {full['precision']:>13.4f} |")
        print(f"  | Recall    | {results['recall']:>20.4f} | {full['recall']:>13.4f} |")
        print(f"  | F1        | {results['f1']:>20.4f} | {full['f1']:>13.4f} |")
        print(f"\n  (Add --guard <path> to auto-populate GUARD column)")


if __name__ == "__main__":
    main()