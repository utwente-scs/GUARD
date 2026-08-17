
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import GUARD, LEAF_COL, get_embedder, load_dataset  # noqa: E402
from splits import get_groups, inner_splits, make_partition, mask_labels  # noqa: E402

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

CSV_PATH = "../../resources/datasets/dataset_t3.csv"
RESULTS_PATH = "../../resources/out/sweep_results.csv"

EMBEDDINGS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/average_word_embeddings_glove.6B.300d",
]

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
CW_GRID = [None, "balanced"]

N_SWEEPS = 30
VAL_FRACTION = 0.2
SEED_OFFSET = 0

REFERENCE_THRESHOLD = 0.5  # reporting only, not used for ranking


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    df = load_dataset(CSV_PATH)
    assert df[LEAF_COL].is_unique, "duplicate keys in dataset"
    leaves = df[LEAF_COL].to_numpy()
    y = df["label"].to_numpy()
    print(f"Labeled: {(y != -1).sum()} | pos: {(y == 1).sum()} | neg: {(y == 0).sum()}")

    select_idx, holdout_idx = make_partition(df, y)
    groups = get_groups(df)

    # HOLDOUT is invisible to this script: its rows stay in X (they are part
    # of the unlabeled corpus) but carry no label.
    y_sel = mask_labels(y, holdout_idx)

    folds = list(
        inner_splits(select_idx, y, groups, N_SWEEPS, VAL_FRACTION, SEED_OFFSET)
    )

    rows = []

    for emb_name in EMBEDDINGS:
        print("\n==============================")
        print(emb_name)
        print("==============================")

        try:
            embed = get_embedder(emb_name)
            X_raw = embed(leaves)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {emb_name}: {e}")
            continue

        print(f"Embedding dimension: {X_raw.shape[1]}")

        for C in C_GRID:
            for cw in CW_GRID:
                aps, f1s, precs, recs = [], [], [], []

                for fit_idx, val_idx in folds:
                    guard = GUARD(C=C, class_weight=cw)
                    guard.fit(X_raw, y_sel, train_idx=fit_idx)

                    proba = guard.predict_proba_raw(X_raw[val_idx])
                    y_val = y[val_idx]

                    aps.append(average_precision_score(y_val, proba))

                    pred = (proba >= REFERENCE_THRESHOLD).astype(int)
                    f1s.append(f1_score(y_val, pred, zero_division=0))
                    precs.append(precision_score(y_val, pred, zero_division=0))
                    recs.append(recall_score(y_val, pred, zero_division=0))

                row = {
                    "embedding": emb_name,
                    "C": C,
                    "class_weight": str(cw),
                    "ap_mean": float(np.mean(aps)),
                    "ap_std": float(np.std(aps)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "prec_mean": float(np.mean(precs)),
                    "rec_mean": float(np.mean(recs)),
                }
                rows.append(row)

                print(
                    f"C={C:<6} cw={str(cw):<9} "
                    f"AP={row['ap_mean']:.4f}+/-{row['ap_std']:.4f}  "
                    f"F1@{REFERENCE_THRESHOLD}={row['f1_mean']:.4f}"
                )

    results = pd.DataFrame(rows).sort_values("ap_mean", ascending=False)
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    print("\n===== BEST PER EMBEDDING (by average precision) =====")
    for emb in EMBEDDINGS:
        sub = results[results.embedding == emb]
        if len(sub):
            b = sub.iloc[0]
            print(f"{emb}: C={b.C} cw={b.class_weight} AP={b.ap_mean:.4f}")

    overall = results.iloc[0]
    print("\n===== SELECTED CONFIGURATION =====")
    print(f"embedding    = {overall.embedding}")
    print(f"C            = {overall.C}")
    print(f"class_weight = {overall.class_weight}")

    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()