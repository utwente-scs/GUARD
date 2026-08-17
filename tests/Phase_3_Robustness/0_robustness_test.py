
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (
    GUARD,
    DECISION_THRESHOLD,
    LABEL_COL,
    LEAF_COL,
    get_embedder,
    load_dataset,
    load_embedding_cache,
)

# ----------------- The Active GLITCH Keywords -----------------
# Only the 38 keywords that actually generated hits in GLITCH's output.

DEFAULT_KEYWORDS = [
    "key", "pass", "password", "user", "token", "cert", "secret", "id", 
    "username", "sh", "crypt", "uuid", "admin", "passwd", "source", 
    "file", "login", "path", "certificate", "ssh_key", "root", 
    "auth_token", "pwd", "bin", "user_name", "fix", "md5", "tar", 
    "dest", "gpg", "owner", "role", "iso", "passno", "userid", 
    "issue", "src", "destination"
]

# ----------------- Inlined split functions -----------------

def get_groups(df, group_col, leaf_col):
    
    if group_col in df.columns:
        return df[group_col].to_numpy(), f"column '{group_col}'"
    else:
        
        groups = df[leaf_col].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).to_numpy()
        return groups, "normalized leaf"

def make_partition(df, y, group_col, leaf_col, holdout_fraction=0.20, seed=42):
    
    labeled_idx = np.where(y != -1)[0]
    groups, _ = get_groups(df, group_col, leaf_col)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed)
    labeled_groups = groups[labeled_idx]
    
    for tr_rel, te_rel in gss.split(labeled_idx, y[labeled_idx], labeled_groups):
        return labeled_idx[tr_rel], labeled_idx[te_rel]
    return np.array([]), np.array([])

def mask_labels(y, eval_idx):
    
    y_masked = y.copy()
    y_masked[eval_idx] = -1
    return y_masked

# ---------------------------------------------------------------------------------

def load_glitch_predictions(path, leaf_col, label_col, positive_value):
    
    import re

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if leaf_col not in fields:
            raise SystemExit(f"GLITCH file has no column '{leaf_col}'. "
                             f"Columns: {fields}")
        rows = list(reader)

    has_label = label_col is not None and label_col in fields
    positives_only = (not has_label) or positive_value == "presence"

    if positives_only:
        pred = {}
        for r in rows:
            if has_label and positive_value not in (None, "presence"):
                if (r.get(label_col) or "").strip() != positive_value:
                    continue
            pred[r[leaf_col]] = 1
        return pred, "positives-only"

    raw = [(r[leaf_col], (r[label_col] or "").strip()) for r in rows]
    values = [v for _, v in raw]
    numeric = all(re.fullmatch(r"-?\d+(\.\d+)?", v or "0") for v in values)

    def is_pos(v):
        if numeric:
            return float(v or 0) > 0
        if positive_value is not None:
            return v == positive_value
        if any(x == "hardcoded-secret" for x in values):
            return v == "hardcoded-secret"
        return v != ""

    pred = {}
    for leaf, v in raw:
        pred[leaf] = max(pred.get(leaf, 0), 1 if is_pos(v) else 0)
    return pred, "labeled-column"


def matched_keywords(leaf, keywords):
    low = str(leaf).lower()
    return {k for k in keywords if k in low}


def metrics(y_true, pred):
    return (precision_score(y_true, pred, zero_division=0),
            recall_score(y_true, pred, zero_division=0),
            f1_score(y_true, pred, zero_division=0))


def one_run(removed, select_idx, holdout_idx, y_all, X_all, leaves_all,
            matched, glitch_pred):
    
    keep = np.asarray(
        [i for i in select_idx if not (matched[leaves_all[i]] & removed)],
        dtype=int,
    )
    y_keep = y_all[keep] if len(keep) else np.array([], dtype=int)
    train_pos = int((y_keep == 1).sum())
    classes = np.unique(y_keep)

    y_true = y_all[holdout_idx]

        
    y_fit = mask_labels(y_all, holdout_idx)
    pipe = GUARD().fit(
        X_all, y_fit, train_idx=keep
    )
    proba = pipe.predict_proba_raw(X_all[holdout_idx])
    guard_pred = (proba >= DECISION_THRESHOLD).astype(int)

    glitch = []
    for i in holdout_idx:
        leaf = leaves_all[i]
        if glitch_pred.get(leaf, 0) != 1:
            glitch.append(0)
        else:
            mk = matched[leaf]
            glitch.append(0 if (mk and mk <= removed) else 1)
    glitch_pred_vec = np.asarray(glitch, dtype=int)

    return metrics(y_true, glitch_pred_vec), metrics(y_true, guard_pred), train_pos


def main():
    parser = argparse.ArgumentParser(
        description="Keyword-removal degradation study (GLITCH vs GUARD) on "
                    "the shared SELECT/HOLDOUT partition.")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-e", "--embeddings", default=None)
    parser.add_argument("--glitch-file", required=True)
    parser.add_argument("--glitch-leaf-col", default="CODE")
    parser.add_argument("--glitch-label-col", default=None)
    parser.add_argument("--positive-value", default=None)
    parser.add_argument("--group-col", default=None,
                        help="override the project column name; omit to let "
                             "script pick project, else normalized leaf")
    parser.add_argument("--keywords", nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=30,
                        help="random keyword subsets per removal count")
    parser.add_argument("--seed-base", type=int, default=200)
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--detail-out", default=None)
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    y_all = df[LABEL_COL].to_numpy()
    leaves_all = df[LEAF_COL].to_numpy()
    X_all = (load_embedding_cache(args.embeddings, df) if args.embeddings
             else get_embedder()(df[LEAF_COL].tolist()))

    group_col = args.group_col if args.group_col is not None else "project"
    
    select_idx, holdout_idx = make_partition(
        df, y_all, group_col=group_col, leaf_col=LEAF_COL
    )
    _, gdesc = get_groups(df, group_col, LEAF_COL)

    glitch_pred, gmode = load_glitch_predictions(
        args.glitch_file, args.glitch_leaf_col, args.glitch_label_col,
        args.positive_value)

    keywords = list(args.keywords if args.keywords else DEFAULT_KEYWORDS)
    matched = {l: matched_keywords(l, keywords) for l in set(leaves_all)}

    print(f"[kw-abl] GLITCH mode: {gmode}; grouping by {gdesc}")
    print(f"[kw-abl] keyword pool ({len(keywords)}): {keywords}")
    print(f"[kw-abl] {args.runs} subsets/removal-count "
          f"(seeds {args.seed_base}-{args.seed_base + args.runs - 1}); "
          f"train on SELECT, score on HOLDOUT, threshold {DECISION_THRESHOLD}\n")

    print("[kw-abl] {:>4} {:>7} | {:>21} | {:>21}".format(
        "#rm", "tr_pos", "GLITCH F1 (mean+/-sd)", "GUARD F1 (mean+/-sd)"))
    print("[kw-abl] " + "-" * 62)

    curve_rows, detail_rows = [], []
    for k in range(len(keywords) + 1):
        gl_runs, gu_runs, trpos_runs = [], [], []
        for r in range(args.runs):
            seed = args.seed_base + r
            subset = set(random.Random(seed).sample(keywords, k))
            gl, gu, trpos = one_run(
                subset, select_idx, holdout_idx, y_all, X_all, leaves_all,
                matched, glitch_pred)
            gl_runs.append(gl)
            gu_runs.append(gu)
            trpos_runs.append(trpos)
            if args.detail_out:
                detail_rows.append({
                    "n_removed": k, "run": r, "seed": seed,
                    "removed_keywords": "|".join(sorted(subset)),
                    "train_pos": trpos,
                    "glitch_precision": gl[0], "glitch_recall": gl[1],
                    "glitch_f1": gl[2], "guard_precision": gu[0],
                    "guard_recall": gu[1], "guard_f1": gu[2]})

        gl_runs = np.array(gl_runs)
        gu_runs = np.array(gu_runs)
        gl_mean, gl_std = gl_runs.mean(0), gl_runs.std(0)
        gu_mean, gu_std = gu_runs.mean(0), gu_runs.std(0)
        tr_pos = int(np.mean(trpos_runs))

        print("[kw-abl] {:>4} {:>7} |   {:.3f} +/- {:.3f}     "
              "|   {:.3f} +/- {:.3f}".format(
                  k, tr_pos, gl_mean[2], gl_std[2], gu_mean[2], gu_std[2]))

        curve_rows.append({
            "n_removed": k, "train_pos": tr_pos, "runs": args.runs,
            "glitch_precision_mean": gl_mean[0], "glitch_precision_std": gl_std[0],
            "glitch_recall_mean": gl_mean[1], "glitch_recall_std": gl_std[1],
            "glitch_f1_mean": gl_mean[2], "glitch_f1_std": gl_std[2],
            "guard_precision_mean": gu_mean[0], "guard_precision_std": gu_std[0],
            "guard_recall_mean": gu_mean[1], "guard_recall_std": gu_std[1],
            "guard_f1_mean": gu_mean[2], "guard_f1_std": gu_std[2],
        })

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
            w.writeheader()
            w.writerows(curve_rows)
        print(f"\n[kw-abl] curves -> {args.out}")
    if args.detail_out and detail_rows:
        with open(args.detail_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)
        print(f"[kw-abl] per-run detail -> {args.detail_out}")


if __name__ == "__main__":
    main()