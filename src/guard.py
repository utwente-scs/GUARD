"""
GUARD: Robust Detection of Security Smells in Infrastructure as Code.

Extended to support Ansible, Puppet, and Terraform through a dispatcher
architecture. Language-specific parsing is handled by parser modules in
src/parsers/; detection logic (RegEx + ML) is shared across all languages.

Usage:
    python -m src.guard -p <path> [-f] [-c] [-o output.csv] [--lang puppet|chef]

    -p, --path     Path to a single IaC file or project folder.
    -f, --folder   Specify that path is a folder (required for -c).
    -c, --context  Activate contextualization (variable resolution).
    -o, --output   Output CSV filename (default: output.csv).
    --lang         Use per-language models from pretrainedModels/{lang}/
                   (default: use base models in pretrainedModels/).
"""

import csv
import re
import importlib
import pickle
import argparse
import os
import shutil
import numpy as np
import time

from setfit import SetFitModel
from src.parsers import ParsedLine, get_parser


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = ('.yml', '.yaml', '.pp', '.tf', '.rb')


def find_iac_files(folder):
    """Find all supported IaC files in a directory tree."""
    file_list = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(SUPPORTED_EXTENSIONS):
                file_list.append(os.path.join(root, f))
    return file_list


def writer_csv(dataset, filepath):
    """Write detection results to a CSV file."""
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in dataset:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Detection logic (shared across all languages)
# ---------------------------------------------------------------------------

# RegEx patterns for pattern-matching smells
ERRORS = {
    'invalidBinding': r'0\.0\.0\.0',
    'weakCrypto': [r'md5', r'sha1', r'arcfour'],
    'http': r'http://',
}

HTTP_WHITELIST = ['localhost', '127.0.0.1']


def detect_smells(parsed_lines, filename, smells, model_comments, model_secrets, vectorizer):
    """Run detection on a list of ParsedLine objects.

    Applies both pattern matching (RegEx) and semantic analysis (ML) to
    each parsed line. This function is language-agnostic — all language-
    specific logic is handled upstream by the parser.

    Args:
        parsed_lines:   List of ParsedLine objects from a language parser.
        filename:       Display filename for output (relative path).
        smells:         List to append detections to (mutated in place).
        model_comments: Trained SetFit model for suspicious comment detection.
        model_secrets:  Trained Label Propagation model for hardcoded secrets.
        vectorizer:     Word2Vec vectorizer module.
    """
    smell_detected = False

    for pl in parsed_lines:
        # --- Comment detection (SetFit) ---
        if pl.comment:
            prediction = model_comments.predict([pl.comment])[0]
            if prediction > 0:
                smells.append([filename, 'suspicious comment', pl.line_number, pl.raw])
                smell_detected = True

        # --- Key-value detection (RegEx + ML) ---
        if pl.key is not None and pl.value is not None:
            value = pl.value
            key = pl.key

            # RegEx: invalid IP binding (0.0.0.0)
            if re.search(ERRORS['invalidBinding'], value):
                smells.append([filename, 'improper ip address binding', pl.line_number, pl.raw])
                smell_detected = True

            # RegEx: HTTP without TLS
            elif 'http' in value.lower():
                if re.search(ERRORS['http'], value) and not any(item in value for item in HTTP_WHITELIST):
                    smells.append([filename, 'use of http', pl.line_number, pl.raw])
                    smell_detected = True
                # Terraform-specific: protocol = "HTTP" (not a URL, but unencrypted protocol)
                elif key.strip() == 'protocol' and value.strip().strip('"') == 'HTTP':
                    smells.append([filename, 'use of http', pl.line_number, pl.raw])
                    smell_detected = True

            # RegEx: weak cryptographic algorithms
            elif any(re.search(weak, value) for weak in ERRORS['weakCrypto']):
                smells.append([filename, 'weak crypto', pl.line_number, pl.raw])
                smell_detected = True

            # ML: hardcoded secrets + empty password (Word2Vec + Label Propagation)
            else:
                vec = vectorizer.getVector(key)
                prediction = model_secrets.predict(np.array([vec]))
                if prediction > 0:
                    value_stripped = value.strip().strip("'\"")
                    if value_stripped in ('', 'undef'):
                        # Key looks like a credential but value is empty/undefined
                        smells.append([filename, 'empty password', pl.line_number, pl.raw])
                        smell_detected = True
                    elif not pl.is_parameterized and len(value) > 0:
                        smells.append([filename, 'hardcoded-secret', pl.line_number, pl.raw])
                        smell_detected = True

    if not smell_detected:
        smells.append([filename, 'none', 'none'])


# ---------------------------------------------------------------------------
# Contextualization dispatcher
# ---------------------------------------------------------------------------

def run_contextualization(folder, extensions_present):
    """Run language-appropriate contextualization on the working directory.

    Args:
        folder: Path to the working directory (./temp).
        extensions_present: Set of file extensions found in the folder.
    """
    if '.yml' in extensions_present or '.yaml' in extensions_present:
        from src.contextualization import ansible_context
        ansible_context.main(folder)

    if '.pp' in extensions_present:
        from src.contextualization import puppet_context
        puppet_context.main(folder)

    if '.tf' in extensions_present:
        from src.contextualization import terraform_context
        terraform_context.main(folder)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def guard_files(output, model_comments, model_secrets, vectorizer):
    """Run GUARD on all IaC files in the working directory.

    Args:
        output:          Path for the output CSV.
        model_comments:  Trained SetFit model.
        model_secrets:   Trained Label Propagation model.
        vectorizer:      Word2Vec vectorizer module.
    """
    smells = []

    for filepath in find_iac_files('./temp'):
        filename = filepath.replace('./temp/', '')
        ext = os.path.splitext(filepath)[1]

        try:
            parser = get_parser(ext)
        except ValueError:
            # No parser for this extension — skip
            continue

        parsed_lines = parser.parse_file(filepath)
        detect_smells(parsed_lines, filename, smells, model_comments, model_secrets, vectorizer)

    writer_csv(smells, output)


def main():
    parser = argparse.ArgumentParser(
        description="GUARD: Robust Detection of Security Smells in Infrastructure as Code."
    )
    parser.add_argument("-p", "--path", help="Path to the file or folder to analyze.", required=True)
    parser.add_argument("-f", "--folder", action="store_true", help="Specify that path is a folder.")
    parser.add_argument("-c", "--context", action="store_true", help="Activate contextualization (variable resolution).")
    parser.add_argument("-o", "--output", help="Output CSV filename.", default='output.csv')
    parser.add_argument("--lang", choices=["puppet", "chef"],
                        help="Use per-language models from pretrainedModels/{lang}/. "
                             "Default: use base models in pretrainedModels/.")

    args = parser.parse_args()

    if args.context and not args.folder:
        parser.error("Contextualization only works with projects (folders), not single files.")

    # --- Load models ---
    load_start = time.time()

    vectorizer = importlib.import_module("src.vectorizer")

    # Model paths — per-language subdirectory if --lang is specified
    base_model_dir = os.path.join(os.path.dirname(__file__), 'pretrainedModels')
    if args.lang:
        model_dir = os.path.join(base_model_dir, args.lang)
        print(f"Using {args.lang}-specific models from {model_dir}/")
    else:
        model_dir = base_model_dir
        print(f"Using base models from {model_dir}/")

    comments_path = os.path.join(model_dir, 'suspCommentsModel')
    secrets_path = os.path.join(model_dir, 'hardCodedSecretModel.pkl')

    if not os.path.exists(comments_path):
        print(f"ERROR: SetFit model not found at {comments_path}")
        return
    if not os.path.exists(secrets_path):
        print(f"ERROR: LP model not found at {secrets_path}")
        return

    model_comments = SetFitModel._from_pretrained(comments_path)

    with open(secrets_path, 'rb') as f:
        model_secrets = pickle.load(f)

    load_time = time.time() - load_start
    print(f"Models loaded in {load_time:.2f}s")

    # --- Prepare working directory ---
    working_dir = './temp'

    try:
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)

        if args.folder:
            shutil.copytree(args.path, working_dir)

            if args.context:
                # Detect which languages are present
                extensions = set()
                for f in find_iac_files(working_dir):
                    extensions.add(os.path.splitext(f)[1])
                run_contextualization(working_dir, extensions)
        else:
            os.mkdir(working_dir)
            shutil.copy(args.path, working_dir)

        # --- Run detection ---
        start_time = time.time()
        guard_files(args.output, model_comments, model_secrets, vectorizer)
        detection_time = time.time() - start_time
        print(f"Detection completed in {detection_time:.2f}s → {args.output}")

        shutil.rmtree(working_dir)

    except IsADirectoryError as e:
        print(e)
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)


if __name__ == '__main__':
    main()