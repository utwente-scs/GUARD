import csv
import re
import pickle
import argparse
import os
import shutil
import time

import numpy as np

import contexualizationPhase
import model as guard_model  # makes model.GUARD importable for unpickling

MODEL_PATH = '../resources/pretrainedModels/guard_model.pkl'


def find_ansible_file(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('.yml', '.yaml')):
                full_path = os.path.join(root, file)
                if os.path.isfile(full_path):  # skip broken symlinks
                    file_list.append(full_path)
    return file_list


def fileReader(filePath):
    with open(filePath, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()
    return lines


def writer_csv(dataset, file):
    with open(file, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        for line in dataset:
            writer.writerow(line)


# ---------------------------------------------------------------------------
# Embedding with a global cache: each distinct key name is embedded once
# per run, regardless of how many files it appears in.
# ---------------------------------------------------------------------------
_embedding_cache = {}


def embed_keys(keys):
    """Return a (len(keys), dim) matrix of embeddings, using the cache."""
    unseen = [k for k in dict.fromkeys(keys) if k not in _embedding_cache]
    if unseen:
        vectors = embedder(unseen)
        for k, v in zip(unseen, vectors):
            _embedding_cache[k] = v
    return np.stack([_embedding_cache[k] for k in keys])


LEAF_FIELDS = ['file', 'line', 'key', 'content', 'label', 'stage', 'score']


def guard_file(filePath, smells, leaves=None):
    lines = fileReader(filePath)
    filename = filePath.replace("./temp/", "")
    smell_detected = False
    leaf_index = {} 

    secret_candidates = []  # (line_number, line, key)

    for line_number, row in enumerate(lines, start=1):
        row = row.rstrip()
        if '---' in row or not row.strip():
            continue

        if '#' in row:
            divided = row.split('#', 1)
            line = divided[0]
        else:
            line = row.lstrip('-').lstrip()

        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.lstrip()
            label, stage = 'none', 'skip:other'  

            if re.search(errors['invalindBinding'], value):
                smells.append([filename, 'improper ip address binding', line_number, line])
                smell_detected = True
                label, stage = 'improper ip address binding', 'regex:binding'  
            elif 'http' in value:
                if re.search(errors['http'], value) and not any(item in value for item in httpWhiteList):
                    smells.append([filename, 'use of http', line_number, line])
                    smell_detected = True
                    label, stage = 'use of http', 'regex:http'  
                else:
                    stage = 'skip:http_branch' 
            elif any(re.search(weak, value) for weak in errors['weakCrypto']):
                smells.append([filename, 'weak crypto', line_number, line])
                smell_detected = True
                label, stage = 'weak crypto', 'regex:weak_crypto'  
            elif ('{' not in value) and len(value) > 0 and len(key) > 0:
                secret_candidates.append((line_number, line, key))
                stage = 'ml:pending' 
            elif '{' in value:
                stage = 'skip:brace' 

            if leaves is not None: 
                leaf_index[line_number] = len(leaves) 
                leaves.append([filename, line_number, key, line, label, stage, ''])  

    if secret_candidates:
        X = embed_keys([key for _, _, key in secret_candidates])
        preds = modelSecrets.predict_raw(X)
        for pred, (line_number, line, _) in zip(preds, secret_candidates):
            if pred > 0:
                smells.append([filename, 'hardcoded-secret', line_number, line])
                smell_detected = True
            if leaves is not None and line_number in leaf_index:  
                leaf = leaves[leaf_index[line_number]]  
                leaf[4] = 'hardcoded-secret' if pred > 0 else 'none'  
                leaf[5] = 'ml:positive' if pred > 0 else 'ml:negative'  
                leaf[6] = float(pred)  

    if not smell_detected:
        smells.append([filename, 'none', 'none', '']) 


def guard_files(output, leaves_output=None):
    smells = []
    leaves = [] if leaves_output else None  

    global embedder
    global modelSecrets

    loadtime = time.time()
    with open(MODEL_PATH, 'rb') as f:
        modelSecrets = pickle.load(f)  # model.GUARD instance (scaler + clf)
    embedder = guard_model.get_embedder()  # frozen sentence-transformers model
    loadModelTime = time.time() - loadtime
    print(f"models loaded in {loadModelTime:.2f}s")

    for filePath in find_ansible_file('./temp'):
        guard_file(filePath, smells, leaves)  

    writer_csv(smells, output)

    if leaves_output:  
        writer_csv([LEAF_FIELDS] + leaves, leaves_output)  
        print(f"{len(leaves)} leaves -> {leaves_output}")


def main():
    parser = argparse.ArgumentParser(description="GUARD: Robust Detection of Ansible Security Smells.")
    parser.add_argument("-p", "--path", help="path is the file or folder to analyze.", required=True)
    parser.add_argument("-f", "--folder", action="store_true", help="specify path is a folder.")
    parser.add_argument("-c", "--context", action="store_true", help="activate contextualizzation procedure.")
    parser.add_argument("-o", "--output", help="name of output file.", default='output.csv')
    parser.add_argument("--leaves", help="optional CSV with one row per analyzed leaf.")  

    args = parser.parse_args()

    global errors
    global httpWhiteList

    errors = {
        'invalindBinding': r'(?<![\d.])0\.0\.0\.0(?![\d.])',
        'weakCrypto': [r'md5', r'sha1', r'arcfour'],
        'http': r'http://'
    }

    httpWhiteList = ['localhost', '127.0.0.1']

    if args.context and not args.folder:
        parser.error("Contextualization phase only work with projects not single files.")

    if not os.path.isfile(MODEL_PATH):
        parser.error(f"pretrained model not found at {MODEL_PATH}. "
                     "Run `python trainer.py -d dataset.csv` first.")

    workingDirectory = './temp'

    try:
        if args.folder:
            shutil.copytree(args.path, workingDirectory, symlinks=True,
                            ignore=shutil.ignore_patterns('.git'))
            if args.context:
                contexualizationPhase.main('./temp')
        else:
            os.mkdir(workingDirectory)
            shutil.copy(args.path, workingDirectory)

        start_time = time.time()
        guard_files(args.output, args.leaves)
        final_time = time.time() - start_time
        print(f"analysis completed in {final_time:.2f}s")

    finally:
        if os.path.isdir(workingDirectory):
            shutil.rmtree(workingDirectory)


if __name__ == '__main__':
    main()