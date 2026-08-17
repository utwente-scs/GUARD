
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
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
OUTPUT_PATH = "../../resources/out/confirmation_results.csv"
PROBA_PATH = "../../resources/out/holdout_probabilities.csv"


MODEL = "sentence-transformers/all-mpnet-base-v2"
THRESHOLD = 0.40

N_SWEEPS = 30
VAL_FRACTION = 0.2
SEED_OFFSET = 42


def metrics(y_true, proba, threshold=THRESHOLD):
    pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "ap": average_precision_score(y_true, proba),
    }


def main():
    df = load_dataset(CSV_PATH)
    assert df[LEAF_COL].is_unique, "duplicate keys in dataset"
    leaves = df[LEAF_COL].to_numpy()
    y = df["label"].to_numpy()

    select_idx, holdout_idx = make_partition(df, y)
    groups = get_groups(df)

    embed = get_embedder(MODEL)
    X_raw = embed(leaves)

    y_fit = mask_labels(y, holdout_idx)

    guard = GUARD()
    guard.fit(X_raw, y_fit, train_idx=select_idx)

    proba_hold = guard.predict_proba_raw(X_raw[holdout_idx])
    y_hold = y[holdout_idx]

    head = metrics(y_hold, proba_hold)
    pred_hold = (proba_hold >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_hold, pred_hold, labels=[0, 1]).ravel()

    print("\n===== HEADLINE (held-out, single pass) =====")
    print(f"n            = {len(y_hold)}  (pos={int(y_hold.sum())})")
    print(f"threshold    = {THRESHOLD}")
    print(f"F1           = {head['f1']:.4f}")
    print(f"Precision    = {head['precision']:.4f}")
    print(f"Recall       = {head['recall']:.4f}")
    print(f"AP           = {head['ap']:.4f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Saved so the sensitivity curve can be recomputed on HOLDOUT without
    # refitting, and so false positives can be inspected by hand.
    Path(PROBA_PATH).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "key": leaves[holdout_idx],
            "label": y_hold,
            "proba": proba_hold,
            "pred": pred_hold,
        }
    ).to_csv(PROBA_PATH, index=False)

    rows = []
    for seed, (fit_idx, val_idx) in enumerate(
        inner_splits(select_idx, y, groups, N_SWEEPS, VAL_FRACTION, SEED_OFFSET),
        start=SEED_OFFSET,
    ):
        g = GUARD()
        g.fit(X_raw, mask_labels(y, holdout_idx), train_idx=fit_idx)
        proba = g.predict_proba_raw(X_raw[val_idx])
        rows.append({"seed": seed, **metrics(y[val_idx], proba)})

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_PATH, index=False)

    print("\n===== STABILITY (SELECT sub-splits; sd, not a CI) =====")
    print(f"F1        = {res.f1.mean():.4f} +/- {res.f1.std():.4f}")
    print(f"Precision = {res.precision.mean():.4f} +/- {res.precision.std():.4f}")
    print(f"Recall    = {res.recall.mean():.4f} +/- {res.recall.std():.4f}")

    print(f"\nSaved {OUTPUT_PATH}")
    print(f"Saved {PROBA_PATH}")


if __name__ == "__main__":
    main()