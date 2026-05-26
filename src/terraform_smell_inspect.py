"""
Extract actual file content for non-GUARD Terraform oracle categories.

Shows the raw lines so we can determine which smells are detectable
with line-level parsing (key = value patterns) vs which require
resource-level analysis (missing attributes).

Run from GUARD/:
    python src/inspect_terraform_smells.py
"""

import csv
import os
import glob


# Categories to investigate (top non-GUARD by count)
TARGET_CATEGORIES = [
    'sec_logging',
    'sec_missing_encryption',
    'sec_firewall_misconfig',
    'sec_access_control',
    'sec_key_management',
    'sec_public_ip',
    'sec_versioning',
    'sec_replication',
    'sec_authentication',
    'sec_weak_password_key_policy',
]


def load_oracle(annotations_dir):
    """Load all oracle annotation CSVs with full detail."""
    oracle = []
    for csv_path in glob.glob(os.path.join(annotations_dir, "*.csv")):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    filepath = row[0].strip()
                    filename = filepath.replace("files/", "")
                    line_num = int(row[1])
                    smell_type = row[2].strip()
                    tp_fp = row[5].strip()
                    if tp_fp == "TP":
                        oracle.append((filename, smell_type, line_num))
    return oracle


def get_file_line(files_dir, filename, line_num, context=1):
    """Get a specific line from a file with optional context."""
    filepath = os.path.join(files_dir, filename)
    if not os.path.exists(filepath):
        return None, []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if line_num > len(lines):
        return None, []

    target = lines[line_num - 1].rstrip()
    ctx = []
    for i in range(max(0, line_num - 1 - context), min(len(lines), line_num + context)):
        marker = ">>>" if i == line_num - 1 else "   "
        ctx.append(f"  {marker} {i+1}: {lines[i].rstrip()}")

    return target, ctx


def main():
    annotations_dir = "datasets/terraform/oracle-dataset/glitch-results-final-annotations"
    files_dir = "datasets/terraform/oracle-dataset/files"

    # Try alternate paths
    if not os.path.exists(annotations_dir):
        annotations_dir = "datasets/new/terraform-GLITCH-replication-package/oracle-dataset/glitch-results-final-annotations"
        files_dir = "datasets/new/terraform-GLITCH-replication-package/oracle-dataset/files"

    print("Loading oracle...")
    oracle = load_oracle(annotations_dir)
    print(f"Total oracle entries: {len(oracle)}\n")

    for category in TARGET_CATEGORIES:
        entries = [(f, s, l) for f, s, l in oracle if s == category]
        if not entries:
            continue

        print(f"\n{'=' * 70}")
        print(f"{category} ({len(entries)} instances)")
        print(f"{'=' * 70}")

        # Track patterns
        detectable = 0  # Lines with clear key=value pattern
        missing_attr = 0  # Lines that are resource declarations (missing attribute)
        unclear = 0

        for filename, smell, line_num in sorted(entries):
            target, ctx = get_file_line(files_dir, filename, line_num)
            if target is None:
                print(f"\n  ✗ {filename} line {line_num}: FILE NOT FOUND")
                continue

            print(f"\n  {filename} line {line_num}:")
            for line in ctx:
                print(f"    {line}")

            # Quick classification
            stripped = target.strip()
            if '=' in stripped and not stripped.startswith('resource') and not stripped.startswith('data'):
                detectable += 1
            elif stripped.startswith('resource') or stripped.startswith('data') or stripped.strip() in ('{', '}', ''):
                missing_attr += 1
            else:
                unclear += 1

        print(f"\n  --- Summary for {category} ---")
        print(f"  Likely detectable (has = assignment): {detectable}")
        print(f"  Likely missing-attribute (resource/block header): {missing_attr}")
        print(f"  Unclear: {unclear}")
        print(f"  Total: {len(entries)}")

    print(f"\n{'=' * 70}")
    print("Done. Review the lines above to identify detectable patterns.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()