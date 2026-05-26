"""
Extract training data candidates from Chef (.rb) or Puppet (.pp) files
using GLITCH's keyword lists for suspicious comments and hardcoded secrets.

Produces two CSV files for manual verification:
    - {tech}_comments_candidates.csv   (word, label, source_file, line_number)
    - {tech}_secrets_candidates.csv    (word, label, source_file, line_number)

The 'label' column is pre-filled based on GLITCH keyword matching:
    1 = candidate match (likely positive — verify manually)
    0 = no keyword match (likely negative — verify manually)

After manual review, the verified entries can be appended to the
existing training datasets (susp_comments_dataset.csv and
hardcoded_secrets_dataset.csv) for retraining.

Usage:
    python extract_training_data.py --tech chef  /path/to/chef-repos
    python extract_training_data.py --tech puppet /path/to/puppet-repos
"""

import os
import re
import csv
import sys
import argparse


# ---------------------------------------------------------------------------
# GLITCH keyword lists (from configs/default.ini)
# ---------------------------------------------------------------------------

SUSPICIOUS_WORDS = [
    "bug", "debug", "todo", "to-do", "to_do", "to be implemented", "fix",
    "issue", "issues", "problem", "solve", "hack", "ticket", "later",
    "incorrect", "fixme",
]

# Keys that suggest hardcoded secrets / credentials
SECRET_KEYWORDS = {
    "passwords": [
        "pass", "pwd", "password", "passwd", "passno",
        "pass-no", "pass_no",
    ],
    "users": [
        "root", "user", "uname", "username", "user-name", "user_name",
        "owner-name", "owner_name", "admin", "login", "userid", "loginid",
    ],
    "secrets": [
        "auth_token", "authentication_token", "auth-token",
        "authentication-token", "secret", "uuid", "crypt", "certificate",
        "token", "ssh_key", "md5", "rsa", "ssl_content", "ca_content",
        "ssl-content", "ca-content", "ssh_key_content", "ssh-key-content",
        "ssh_key_public", "ssh-key-public", "ssh_key_private",
        "ssh-key-private", "ssh_key_public_content",
        "ssh_key_private_content", "ssh-key-public-content",
        "ssh-key-private-content",
    ],
    "misc_secrets": ["key", "cert"],
    "admin": ["admin", "root"],
}

# Flatten all secret keywords into a single set for matching
ALL_SECRET_WORDS = set()
for category in SECRET_KEYWORDS.values():
    ALL_SECRET_WORDS.update(category)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

TECH_EXTENSIONS = {
    'chef': '.rb',
    'puppet': '.pp',
}


def find_iac_files(folder, extension):
    """Find all files with the given extension in a directory tree."""
    files = []
    for root, dirs, filenames in os.walk(folder, followlinks=True):
        for f in filenames:
            if f.endswith(extension):
                files.append(os.path.join(root, f))
    return files


# ---------------------------------------------------------------------------
# Comment extraction (works for both Chef and Puppet — both use #)
# ---------------------------------------------------------------------------

def extract_comments(filepath):
    """Extract comments from a Chef .rb or Puppet .pp file.

    Both languages use # for comments.
    Returns list of (comment_text, line_number) tuples.
    """
    comments = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()

            # Full-line comment
            if stripped.startswith('#'):
                comment = stripped.lstrip('#').strip()
                if comment:
                    comments.append((comment, i))
                continue

            # Inline comment (simple heuristic: # not inside quotes)
            in_single = False
            in_double = False
            for ci, ch in enumerate(stripped):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif ch == '#' and not in_single and not in_double:
                    # Skip Ruby string interpolation #{
                    if ci + 1 < len(stripped) and stripped[ci + 1] == '{':
                        continue
                    comment = stripped[ci + 1:].strip()
                    if comment:
                        comments.append((comment, i))
                    break

    return comments


# ---------------------------------------------------------------------------
# Key extraction — Chef (.rb)
# ---------------------------------------------------------------------------

def extract_keys_chef(filepath):
    """Extract key names from Chef .rb key-value patterns.

    Returns list of (key_name, line_number) tuples.
    """
    keys = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            # Pattern: key => value (hash rocket)
            m = re.match(r":?(\w[\w-]*)\s*=>", stripped)
            if m:
                keys.append((m.group(1), i))
                continue

            # Pattern: key: value (symbol-key hash)
            m = re.match(r"(\w[\w-]*):\s+", stripped)
            if m:
                keys.append((m.group(1), i))
                continue

            # Pattern: default/node/override[...]['key'] = value
            if '=' in stripped and '=>' not in stripped:
                m = re.match(r"(.+?)\s*(?<![!=><])=(?!=)\s*(.+)", stripped)
                if m:
                    lhs = m.group(1).strip()
                    key_m = re.search(r"""[\['":]([\w]+)['"\]]?\]?\s*$""", lhs)
                    if key_m:
                        keys.append((key_m.group(1), i))
                        continue

            # Pattern: key 'value' / key value (space-separated)
            if re.search(r'\bdo(\s*\|.*\|)?\s*$', stripped):
                continue
            m = re.match(r"(\w[\w-]*)\s+(['\"].*|:?\w.*)", stripped)
            if m:
                key = m.group(1)
                if key not in {
                    'end', 'if', 'unless', 'case', 'when', 'else', 'elsif',
                    'class', 'module', 'def', 'return', 'raise', 'include',
                    'require', 'extend', 'include_recipe', 'depends',
                    'not_if', 'only_if', 'notifies', 'subscribes',
                }:
                    keys.append((key, i))

    return keys


# ---------------------------------------------------------------------------
# Key extraction — Puppet (.pp)
# ---------------------------------------------------------------------------

def extract_keys_puppet(filepath):
    """Extract key names from Puppet .pp key-value patterns.

    Handles:
        key => value     (resource attributes)
        $var = value     (variable assignments)

    Returns list of (key_name, line_number) tuples.
    """
    keys = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            # Pattern: key => value (resource attributes)
            m = re.match(r"(\w[\w-]*)\s*=>", stripped)
            if m:
                keys.append((m.group(1), i))
                continue

            # Pattern: $variable = value
            m = re.match(r"\$(\w[\w:]*)\s*=\s*", stripped)
            if m:
                keys.append((m.group(1), i))
                continue

            # Pattern: class/define parameter defaults
            # class myclass ( Type $param = 'default' )
            params = re.findall(r"\$(\w[\w:]*)\s*=\s*", stripped)
            for param in params:
                keys.append((param, i))

    return keys


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label_comment(comment_text):
    """Check if a comment contains any GLITCH suspicious words."""
    lower = comment_text.lower()
    for word in SUSPICIOUS_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", lower):
            return 1
    return 0


def tokenize_key(key_name):
    """Split a key name into constituent terms.

    Handles snake_case, camelCase, and kebab-case.
    """
    parts = re.split(r'[_\-]', key_name)
    tokens = []
    for part in parts:
        tokens.extend(re.findall(r'[a-z]+|[A-Z][a-z]*|[A-Z]+(?=[A-Z]|$)', part))
    return [t.lower() for t in tokens if t]


def label_key(key_name):
    """Check if a key name matches any GLITCH secret keywords."""
    lower = key_name.lower()
    tokens = tokenize_key(key_name)

    # Direct match
    if lower in ALL_SECRET_WORDS:
        return 1

    # Token match
    for token in tokens:
        if token in ALL_SECRET_WORDS:
            return 1

    # Substring match
    for keyword in ALL_SECRET_WORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            return 1

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract training data candidates from Chef or Puppet IaC files."
    )
    parser.add_argument(
        "--tech",
        required=True,
        choices=["chef", "puppet"],
        help="IaC technology (chef or puppet).",
    )
    parser.add_argument(
        "path",
        help="Path to the dataset folder (repos, NOT oracle).",
    )

    args = parser.parse_args()
    tech = args.tech
    dataset_path = args.path
    extension = TECH_EXTENSIONS[tech]

    iac_files = find_iac_files(dataset_path, extension)

    if not iac_files:
        print(f"No {extension} files found in {dataset_path}")
        sys.exit(1)

    print(f"Found {len(iac_files)} {extension} files")

    # --- Extract comments ---
    all_comments = []
    for filepath in iac_files:
        rel_path = os.path.relpath(filepath, dataset_path)
        for comment_text, line_num in extract_comments(filepath):
            label = label_comment(comment_text)
            all_comments.append((comment_text, label, rel_path, line_num))

    # --- Extract keys ---
    extract_keys = extract_keys_chef if tech == 'chef' else extract_keys_puppet

    all_keys = []
    for filepath in iac_files:
        rel_path = os.path.relpath(filepath, dataset_path)
        for key_name, line_num in extract_keys(filepath):
            label = label_key(key_name)
            all_keys.append((key_name, label, rel_path, line_num))

    # --- Write outputs ---
    comments_file = f'{tech}_comments_candidates.csv'
    secrets_file = f'{tech}_secrets_candidates.csv'

    with open(comments_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'label', 'source_file', 'line_number'])
        for row in all_comments:
            writer.writerow(row)

    with open(secrets_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'label', 'source_file', 'line_number'])
        for row in all_keys:
            writer.writerow(row)

    # --- Summary ---
    pos_comments = sum(1 for _, l, _, _ in all_comments if l == 1)
    neg_comments = sum(1 for _, l, _, _ in all_comments if l == 0)
    pos_keys = sum(1 for _, l, _, _ in all_keys if l == 1)
    neg_keys = sum(1 for _, l, _, _ in all_keys if l == 0)

    print(f"\n=== COMMENTS ({tech.upper()}) ===")
    print(f"  Total:     {len(all_comments)}")
    print(f"  Positive:  {pos_comments} (contain suspicious words)")
    print(f"  Negative:  {neg_comments}")

    print(f"\n=== KEYS ({tech.upper()}) ===")
    print(f"  Total:     {len(all_keys)}")
    print(f"  Positive:  {pos_keys} (match secret keywords)")
    print(f"  Negative:  {neg_keys}")

    print(f"\nOutput written to:")
    print(f"  {comments_file}")
    print(f"  {secrets_file}")
    print(f"\nReview these files manually, then append verified entries")
    print(f"to susp_comments_dataset.csv and hardcoded_secrets_dataset.csv")


if __name__ == '__main__':
    main()