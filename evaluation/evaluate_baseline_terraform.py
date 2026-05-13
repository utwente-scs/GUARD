"""
Compare GUARD baseline output against GLITCH Terraform oracle annotations.
Computes precision, recall, and F1-score per smell type and overall.

Run from GUARD/ root:
    python evaluate_baseline.py
"""

import csv
import os
import glob


def load_guard_output(path):
    """Load GUARD's baseline output CSV."""
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


def load_oracle(annotations_dir):
    """Load all oracle annotation CSVs."""
    oracle = []
    for csv_path in glob.glob(os.path.join(annotations_dir, "*.csv")):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    filepath = row[0].strip()
                    # Extract just the filename (oracle uses files/xxx.tf)
                    filename = filepath.replace("files/", "")
                    line_num = int(row[1])
                    smell_type = row[2].strip()
                    tp_fp = row[5].strip()
                    if tp_fp == "TP":
                        oracle.append((filename, smell_type, line_num))
    return oracle


# Map GUARD smell names to oracle smell names
GUARD_TO_ORACLE = {
    "hardcoded-secret": ["sec_hard_secr", "sec_hard_pass", "sec_hard_user", "sec_empty_pass"],
    "use of http": ["sec_https", "sec_ssl_tls_policy"],
    "improper ip address binding": ["sec_invalid_bind"],
    "weak crypto": ["sec_weak_crypt"],
    "suspicious comment": ["sec_susp_comm"],
}

# Reverse mapping: oracle smell -> GUARD smell
ORACLE_TO_GUARD = {}
for guard_smell, oracle_smells in GUARD_TO_ORACLE.items():
    for os_name in oracle_smells:
        ORACLE_TO_GUARD[os_name] = guard_smell

# Oracle smells that GUARD doesn't attempt to detect
NON_GUARD_SMELLS = {
    "sec_logging", "sec_naming", "sec_missing_encryption",
    "sec_access_control", "sec_key_management", "sec_firewall_misconfig",
    "sec_integrity_policy", "sec_def_admin",
    "sec_replication", "sec_versioning", "sec_public_ip",
    "sec_authentication", "sec_sensitive_iam_action",
    "sec_network_security_rules", "sec_weak_password_key_policy",
    "sec_threats_detection_alerts", "sec_dnssec",
    "sec_permission_iam_policies", "sec_attached_resource",
}


def evaluate(guard_detections, oracle_entries):
    """Compute TP, FP, FN and metrics."""
    # Filter oracle to only GUARD-compatible smells
    oracle_compatible = [
        (f, s, l) for f, s, l in oracle_entries if s not in NON_GUARD_SMELLS
    ]
    oracle_incompatible = [
        (f, s, l) for f, s, l in oracle_entries if s in NON_GUARD_SMELLS
    ]

    # Build sets for matching (file, line_number)
    # We match on file + line since smell names differ between tools
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "tp_keys": tp_keys, "fp_keys": fp_keys, "fn_keys": fn_keys,
        "oracle_compatible": oracle_compatible,
        "oracle_incompatible": oracle_incompatible,
        "guard_by_key": guard_by_key,
        "oracle_by_key": oracle_by_key,
    }


def main():
    guard_output = "baseline_terraform.csv"
    annotations_dir = "datasets/new/terraform-GLITCH-replication-package/oracle-dataset/glitch-results-final-annotations"

    print("Loading GUARD baseline output...")
    guard_detections = load_guard_output(guard_output)
    print(f"  GUARD detections: {len(guard_detections)}")

    print("Loading oracle annotations...")
    oracle_entries = load_oracle(annotations_dir)
    print(f"  Oracle entries (TP only): {len(oracle_entries)}")

    results = evaluate(guard_detections, oracle_entries)

    # --- Overall results ---
    print(f"\n{'='*60}")
    print(f"ORACLE OVERVIEW")
    print(f"{'='*60}")
    print(f"  Total oracle smells (TP): {len(oracle_entries)}")
    print(f"  GUARD-compatible smells:  {len(results['oracle_compatible'])}")
    print(f"  Non-GUARD smells:         {len(results['oracle_incompatible'])}")

    # Smell distribution in oracle
    smell_counts = {}
    for _, s, _ in oracle_entries:
        smell_counts[s] = smell_counts.get(s, 0) + 1
    print(f"\n  Oracle smell distribution:")
    for s in sorted(smell_counts, key=smell_counts.get, reverse=True):
        compat = "  " if s not in NON_GUARD_SMELLS else " *"
        print(f"    {s}: {smell_counts[s]}{compat}")
    print(f"    (* = not in GUARD's scope)")

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS (GUARD-compatible smells only)")
    print(f"{'='*60}")
    print(f"  True Positives:  {results['tp']}")
    print(f"  False Positives: {results['fp']}")
    print(f"  False Negatives: {results['fn']}")
    print(f"  Precision:       {results['precision']:.4f}")
    print(f"  Recall:          {results['recall']:.4f}")
    print(f"  F1-score:        {results['f1']:.4f}")

    # --- Per-smell breakdown ---
    print(f"\n{'='*60}")
    print(f"PER-SMELL BREAKDOWN")
    print(f"{'='*60}")

    guard_smell_counts = {}
    for _, s, _ in guard_detections:
        guard_smell_counts[s] = guard_smell_counts.get(s, 0) + 1
    print(f"\n  GUARD detections by type:")
    for s, c in sorted(guard_smell_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    # --- Detailed TP/FP/FN lists ---
    print(f"\n{'='*60}")
    print(f"TRUE POSITIVES (GUARD detection matched oracle)")
    print(f"{'='*60}")
    for key in sorted(results["tp_keys"]):
        guard_s = results["guard_by_key"].get(key, "?")
        oracle_s = results["oracle_by_key"].get(key, "?")
        print(f"  {key[0]} line {key[1]}: GUARD={guard_s}, Oracle={oracle_s}")

    print(f"\n{'='*60}")
    print(f"FALSE POSITIVES (GUARD detected, not in oracle)")
    print(f"{'='*60}")
    for key in sorted(results["fp_keys"]):
        guard_s = results["guard_by_key"].get(key, "?")
        print(f"  {key[0]} line {key[1]}: GUARD={guard_s}")

    print(f"\n{'='*60}")
    print(f"FALSE NEGATIVES (in oracle, GUARD missed)")
    print(f"{'='*60}")
    for key in sorted(results["fn_keys"]):
        oracle_s = results["oracle_by_key"].get(key, "?")
        print(f"  {key[0]} line {key[1]}: Oracle={oracle_s}")


if __name__ == "__main__":
    main()